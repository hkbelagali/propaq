///
/// impl for the Majorana propagator, which works with observables
/// represented in the Majorana operator basis. The propagator is
/// a thin wrapper over the partitioned engine, incorporating
/// the Majorana algebra via `MajoranaBasis`.
///
use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::results::PropagationResult;
use propaq_core::run_config::RunConfig;
use propaq_core::term_io::save_terms_to_file;
use propaq_core::truncators::{reject_surrogate_only, resolve_truncation};

use std::io::Write;

use crate::engine;
use crate::monomial::MajoranaMonomial;
use crate::termsum::{MajoranaTermSum, Storage};

/// Replays a run's per-gate records into the verbose log.
fn write_gate_log(
    log_filename: Option<&str>,
    log_every: usize,
    cfg: &propaq_core::truncators::ResolvedConfig,
    gates: &[engine::GateRecord],
    phases: &propaq_core::partitioned_termsum::PhaseStats,
) -> PyResult<()> {
    let Some(filename) = log_filename else {
        return Ok(());
    };
    let f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut log = std::io::BufWriter::new(f);
    let every = log_every.max(1);
    let wc = cfg
        .weight
        .map_or_else(|| "null".to_string(), |w| w.to_string());
    let cc = cfg.coefficient.unwrap_or(0.0);
    for g in gates.iter().filter(|g| g.gate_idx % every == 0) {
        let qki = g
            .qiskit_gate_idx
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let _ = writeln!(
            log,
            r#"{{"event":"gate","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"terms":{},"ms_per_gate":{:.3e}}}"#,
            g.gate_idx, g.layer_idx, qki, g.terms_after, g.elapsed_ms
        );
        let _ = writeln!(
            log,
            r#"{{"event":"truncation","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"trigger":"emit","truncation_model":"emit_gate","terms_before":{},"terms_after":{},"terms_gained":{},"terms_discarded":{},"discarded_coeff_l1":{:.6e},"discarded_coeff_max":{:.6e},"weight_cutoff":{},"coeff_cutoff":{:.6e},"elapsed_ms":{:.3e}}}"#,
            g.gate_idx,
            g.layer_idx,
            qki,
            g.terms_before,
            g.terms_after,
            g.terms_gained,
            g.declined,
            g.discarded_coeff_l1,
            g.discarded_coeff_max,
            wc,
            cc,
            g.elapsed_ms
        );
    }
    write_phase_event(&mut log, phases);
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

/// Back-propagates Majorana observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: A list of truncators
///         (WeightTruncator, CoefficientTruncator, TermBudget), a single such
///         truncator, a legacy TruncationPolicy (decomposed), or None. The
///         symbolic-only FrequencyTruncator is rejected.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     logger: Optional Logger for verbose JSON Lines event logging.
///     pin_threads: Bind each worker to its own CPU.
///     progress_bar: Draw a tqdm bar over the gate loop.
///     progress_every: Gates between progress bar ticks. Defaults to 1.
///
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaPropagator {
    inner: RunConfig,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl MajoranaPropagator {
    /// Initialize the Majorana propagator. See the class docstring for arguments.
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, logger=None, pin_threads=true, progress_bar=false, progress_every=1))]
    fn new(
        noise: Option<Py<PyAny>>,
        truncation: Option<Bound<'_, PyAny>>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        pin_threads: bool,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_surrogate_only(&truncators)?;
        Ok(MajoranaPropagator {
            inner: RunConfig::new(
                noise,
                truncators,
                n_threads,
                logger,
                pin_threads,
                progress_bar,
                progress_every,
            )?,
        })
    }

    /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
    ///
    /// Arguments:
    ///     observable: The Majorana observable to back-propagate.
    ///     circuit: A MajoranaCircuit whose rotations are applied in reverse.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, filename=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        filename: Option<String>,
    ) -> PyResult<MajoranaTermSum> {
        // The engine declines only a width past its dispatch ladder, which the
        // error below reports; there is no other backend to fall through to.
        {
            let cfg = propaq_core::truncators::resolve_config(&self.inner.truncators);
            let (n, pool) = (observable.n_units(), &self.inner.pool);
            let threads = pool.current_num_threads().into();
            let noise = self.inner.noise.as_ref().map(|x| x.bind(py));
            let log = self.inner.log_filename.is_some();
            let mut recs: Vec<engine::GateRecord> = Vec::new();
            let mut phases = propaq_core::partitioned_termsum::PhaseStats::default();
            let evolved = match &observable.inner {
                Storage::F64(src) => {
                    let terms: Vec<(MajoranaMonomial, f64)> =
                        crate::termsum::materialize(src).into_iter().collect();
                    engine::run::<f64>(
                        py,
                        &terms,
                        circuit,
                        None,
                        n,
                        &cfg,
                        pool,
                        threads,
                        noise,
                        false,
                        true,
                        log,
                        self.inner.progress_bar,
                        self.inner.progress_every,
                    )?
                    .and_then(|o| {
                        recs = o.gates;
                        phases = o.phases;
                        o.terms
                    })
                    .map(|t| {
                        let store = crate::termsum::term_sum_from_pairs(t, n);
                        (
                            crate::termsum::materialize(&store),
                            MajoranaTermSum::from_store(store),
                        )
                    })
                }
                Storage::F32(src) => {
                    let terms: Vec<(MajoranaMonomial, f64)> =
                        crate::termsum::materialize(src).into_iter().collect();
                    engine::run::<f32>(
                        py,
                        &terms,
                        circuit,
                        None,
                        n,
                        &cfg,
                        pool,
                        threads,
                        noise,
                        false,
                        true,
                        log,
                        self.inner.progress_bar,
                        self.inner.progress_every,
                    )?
                    .and_then(|o| {
                        recs = o.gates;
                        phases = o.phases;
                        o.terms
                    })
                    .map(|t| {
                        let store = crate::termsum::term_sum_from_pairs(t, n);
                        (
                            crate::termsum::materialize(&store),
                            MajoranaTermSum::from_store_f32(store),
                        )
                    })
                }
            };
            write_gate_log(
                self.inner.log_filename.as_deref(),
                self.inner.log_every,
                &cfg,
                &recs,
                &phases,
            )?;
            if let Some((map, evolved)) = evolved {
                if let Some(path) = filename.as_deref() {
                    save_terms_to_file(&map, path)?;
                }
                return Ok(evolved);
            }
        }

        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "propaq: {} modes exceeds the engine's maximum of {}",
            observable.n_units(),
            engine::MAX_DISPATCH_SITES,
        )))
    }

    /// Compute the expectation value of *observable* in the state prepared by *circuit*.
    ///
    /// Arguments:
    ///     observable: The Majorana observable.
    ///     circuit: A MajoranaCircuit applied to the reference state.
    ///     initial_state: Fock state as a bitstring integer.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, initial_state=None, filename=None))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
        filename: Option<String>,
    ) -> PyResult<PropagationResult> {
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.n_units())?,
            None => Bitset::zero(),
        };
        // `filename` asks for the evolved operator on disk as well as the
        // expectation value, so the run has to hand its terms back for it.
        let want_terms = filename.is_some();
        {
            let cfg = propaq_core::truncators::resolve_config(&self.inner.truncators);
            let run = |terms: &[(MajoranaMonomial, f64)], f32_storage: bool| {
                let (n, pool) = (observable.n_units(), &self.inner.pool);
                let threads = pool.current_num_threads().into();
                let words = initial_state.as_words();
                let log = self.inner.log_filename.is_some();
                let noise = self.inner.noise.as_ref().map(|n| n.bind(py));
                if f32_storage {
                    engine::run::<f32>(
                        py,
                        terms,
                        circuit,
                        Some(words),
                        n,
                        &cfg,
                        pool,
                        threads,
                        noise,
                        true,
                        want_terms,
                        log,
                        self.inner.progress_bar,
                        self.inner.progress_every,
                    )
                    .map(|o| {
                        o.map(|o| {
                            (
                                o.result,
                                o.gates,
                                o.phases,
                                o.terms.map(|t| {
                                    crate::termsum::materialize(
                                        &crate::termsum::term_sum_from_pairs(t, n),
                                    )
                                }),
                            )
                        })
                    })
                } else {
                    engine::run::<f64>(
                        py,
                        terms,
                        circuit,
                        Some(words),
                        n,
                        &cfg,
                        pool,
                        threads,
                        noise,
                        true,
                        want_terms,
                        log,
                        self.inner.progress_bar,
                        self.inner.progress_every,
                    )
                    .map(|o| {
                        o.map(|o| {
                            (
                                o.result,
                                o.gates,
                                o.phases,
                                o.terms.map(|t| {
                                    crate::termsum::materialize(
                                        &crate::termsum::term_sum_from_pairs(t, n),
                                    )
                                }),
                            )
                        })
                    })
                }
            };
            let served = match &observable.inner {
                Storage::F64(s) => run(
                    &crate::termsum::materialize(s)
                        .into_iter()
                        .collect::<Vec<_>>(),
                    false,
                )?,
                Storage::F32(s) => run(
                    &crate::termsum::materialize(s)
                        .into_iter()
                        .collect::<Vec<_>>(),
                    true,
                )?,
            };
            if let Some((result, recs, phases, terms)) = served {
                write_gate_log(
                    self.inner.log_filename.as_deref(),
                    self.inner.log_every,
                    &cfg,
                    &recs,
                    &phases,
                )?;
                if let (Some(path), Some(map)) = (filename.as_deref(), terms.as_ref()) {
                    save_terms_to_file(map, path)?;
                }
                return Ok(result);
            }
        }

        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "propaq: {} modes exceeds the engine's maximum of {}",
            observable.n_units(),
            engine::MAX_DISPATCH_SITES,
        )))
    }

    #[getter]
    fn noise(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.inner.noise.as_ref().map(|n| n.clone_ref(py))
    }

    #[pyo3(signature = (noise=None))]
    fn set_noise(&mut self, noise: Option<Py<PyAny>>) {
        self.inner.noise = noise;
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .truncators
            .iter()
            .map(|t| t.to_object(py))
            .collect()
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`).
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_surrogate_only(&truncators)?;
        self.inner.truncators = truncators;
        Ok(())
    }
}
