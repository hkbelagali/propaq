//!
//! impl for the surrogate/symbolic propagator! This uses the same
//! partitioned engine as the numerical propagators, with a symbolic
//! coefficient type.
//!
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;

use propaq_core::basis_module::BasisCodec;
use propaq_core::bitset::Bitset;
use propaq_core::logger::Logger;
use propaq_core::progress::Progress;
use propaq_core::run::RunGate;
use propaq_core::store::TermBasis;
use propaq_core::traits::AbstractTerm;

use crate::build::{build, extract_param, BuildOutput};
use crate::model::SurrogateModel;
use crate::symcoeff::SymbolicCoeff;
use crate::truncation::FrequencyTruncationPolicy;
use propaq_core::truncators::{resolve_config, resolve_truncation as core_resolve_truncation, Truncator};

pub fn resolve_truncation(truncation: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Truncator>> {
    if let Some(obj) = truncation {
        if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
            return Ok(legacy.decompose());
        }
    }
    core_resolve_truncation(truncation)
}

/// Configuration for a surrogate build.
pub struct SurrogatePropagator<B: TermBasis> {
    pub pool: Arc<rayon::ThreadPool>,

    pub truncators: Vec<Truncator>,
    pub log_filename: Option<String>,
    pub log_every: usize,
    /// Draw a tqdm bar over the build's gate loop.
    pub progress_bar: bool,
    /// Gates between bar ticks.
    pub progress_every: usize,
    _marker: PhantomData<B>,
}

impl<B: TermBasis> SurrogatePropagator<B> {
    pub fn new(
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        let pool = Arc::new(
            builder
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        let (log_filename, log_every) = match &logger {
            Some(obj) => Python::attach(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(SurrogatePropagator {
            pool,
            truncators,
            log_filename,
            log_every,
            progress_bar,
            progress_every: progress_every.max(1),
            _marker: PhantomData,
        })
    }
}

impl<B: BasisCodec> SurrogatePropagator<B>
where
    B::Term: AbstractTerm,
{
    /// Builds a surrogate model on the core run loop.
    ///
    /// `terms` is the observable and `n_units` is its size in the basis's own
    /// units (for example modes). The engine runs with `B::Alg`, the algebra
    /// of the basis.
    pub fn build_model(
        &self,
        py: Python<'_>,
        terms: &[(B::Term, f64)],
        n_units: usize,
        circuit: &Bound<'_, PyAny>,
        initial_state: &Bitset,
    ) -> PyResult<SurrogateModel> {
        use pyo3::exceptions::PyValueError;
        crate::build::check_supported::<B::Alg>()?;
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let engine_units = B::engine_units(n_units)
            .filter(|u| *u <= B::MAX_UNITS)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "propaq: {} {} exceeds the engine's maximum of {}",
                    n_units,
                    B::UNIT_NAME,
                    B::MAX_UNITS
                ))
            })?;
        let layers = extract_layers::<B>(py, circuit)?;
        let obs: Vec<(Vec<u64>, f64)> = terms.iter().map(|(t, c)| (B::to_words(t), *c)).collect();
        let cfg = resolve_config(&self.truncators);
        let pool = Arc::clone(&self.pool);
        let partitions = pool.current_num_threads().max(1);
        let total_gates = layers.iter().map(Vec::len).sum();
        let progress = Progress::new(py, self.progress_bar, total_gates, self.progress_every)?;
        let out = py.detach(|| {
            pool.install(|| {
                build::<B::Alg>(
                    &obs,
                    &layers,
                    engine_units,
                    engine_units * B::BITS_PER_UNIT,
                    partitions,
                    &cfg,
                    initial_state.as_words(),
                    n_params,
                    progress.as_ref(),
                )
            })
        });
        if let Some(p) = progress.as_ref() {
            p.close();
        }
        let BuildOutput {
            model,
            gates,
            merges,
            phases,
        } = out?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "propaq: {} {} exceeds the engine's maximum of {}",
                n_units,
                B::UNIT_NAME,
                B::MAX_UNITS
            ))
        })?;
        write_merge_log(
            self.log_filename.as_deref(),
            self.log_every,
            &gates,
            &merges,
            &phases,
        )?;
        Ok(model)
    }
}

/// Reads a circuit's layers into lowered gates, in application order.
fn extract_layers<B: BasisCodec>(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<RunGate<SymbolicCoeff>>>>
where
    B::Term: AbstractTerm,
{
    let layers: Vec<Vec<Py<PyAny>>> = circuit.getattr("layers")?.extract()?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers.iter().rev() {
        let mut gates = Vec::with_capacity(layer.len());
        for rot_obj in layer.iter().rev() {
            let rot = rot_obj.bind(py);
            let generator = B::extract_term(&rot.getattr("generator")?)?;
            let angle = extract_param(rot)?;
            let qiskit_gate_idx = rot
                .getattr("qiskit_gate_idx")
                .ok()
                .and_then(|v| v.extract::<Option<usize>>().ok())
                .flatten();
            gates.push(RunGate {
                qiskit_gate_idx,
                generator: B::to_words(&generator),
                angle,
            });
        }
        out.push(gates);
    }
    Ok(out)
}

/// Replays the build's collected gate and merge records into the verbose log.
fn write_merge_log(
    log_filename: Option<&str>,
    log_every: usize,
    gates: &[crate::build::GateRecord],
    merges: &[crate::build::MergeRecord],
    phases: &propaq_core::partitioned_termsum::PhaseStats,
) -> PyResult<()> {
    let Some(filename) = log_filename else {
        return Ok(());
    };
    let f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut log = BufWriter::new(f);
    let log = &mut log;
    let every = log_every.max(1);
    for g in gates.iter().filter(|g| g.gate_idx % every == 0) {
        let qki = g
            .qiskit_gate_idx
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let _ = writeln!(
            log,
            r#"{{"event":"gate","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"terms":{},"monomials":{},"ms_per_gate":{:.3e}}}"#,
            g.gate_idx, g.layer_idx, qki, g.terms, g.monomials, g.ms_per_gate
        );
    }
    for m in merges.iter().filter(|m| m.gate_idx % every == 0) {
        let discarded = m.terms_before.saturating_sub(m.terms_after);
        let mono_discarded = m.monomials_before.saturating_sub(m.monomials_after);
        let qki = m
            .qiskit_gate_idx
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let freq = m
            .frequency
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let weight = m
            .weight
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let coeff = m
            .coefficient
            .map_or_else(|| "null".to_string(), |v| format!("{v:.6e}"));
        let _ = writeln!(
            log,
            r#"{{"event":"surrogate_merge","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"trigger":"{}","terms_before":{},"terms_after":{},"terms_discarded":{},"monomials_before":{},"monomials_after":{},"monomials_discarded":{},"frequency":{},"weight":{},"coefficient":{},"elapsed_ms":{:.3e}}}"#,
            m.gate_idx,
            m.layer_idx,
            qki,
            m.trigger,
            m.terms_before,
            m.terms_after,
            discarded,
            m.monomials_before,
            m.monomials_after,
            mono_discarded,
            freq,
            weight,
            coeff,
            m.elapsed_ms,
        );
    }
    write_phase_event(log, phases);
    Ok(())
}

/// One closing record with the run's phase split and kernel counters.
fn write_phase_event(log: &mut impl Write, p: &propaq_core::partitioned_termsum::PhaseStats) {
    let occupancy = |busy: f64, wall: f64| busy / (wall.max(1e-12) * p.partitions as f64);
    let share = |part: u64, whole: u64| part as f64 / whole.max(1) as f64;
    let _ = writeln!(
        log,
        r#"{{"event":"engine_phases","partitions":{},"scan_s":{:.6e},"absorb_s":{:.6e},"claims_s":{:.6e},"scan_occupancy":{:.4},"absorb_occupancy":{:.4},"claims_occupancy":{:.4},"terms":{},"inline_positions":{},"overflow_rows":{},"overflow_share":{:.4},"visited":{},"emitted":{},"declined":{},"emitted_share":{:.4},"declined_share":{:.4},"exchange_hits":{},"exchange_hit_share":{:.4}}}"#,
        p.partitions,
        p.scan_seconds,
        p.absorb_seconds,
        p.claims_seconds,
        occupancy(p.scan_busy_seconds, p.scan_seconds),
        occupancy(p.absorb_busy_seconds, p.absorb_seconds),
        occupancy(p.claims_busy_seconds, p.claims_seconds),
        p.terms,
        p.inline_positions,
        p.overflow_rows,
        p.overflow_rows as f64 / p.terms.max(1) as f64,
        p.visited,
        p.emitted,
        p.declined,
        share(p.emitted, p.visited),
        share(p.declined, p.visited),
        p.exchange_hits,
        share(p.exchange_hits, p.emitted),
    );
}

