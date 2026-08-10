///
/// impl for the Pauli propagator, which works with observables
/// represented in the Pauli operator basis. The propagator is
/// a thin wrapper over the partitioned engine, incorporating
/// the Pauli algebra via `PauliBasis`.
///
use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::propagator::{save_terms_to_file, PropagationResult};
use propaq_core::run_config::RunConfig;
use propaq_core::truncators::{reject_surrogate_only, resolve_truncation, FlushSchedule};

use std::io::Write;

use crate::engine;
use crate::string::PauliString;
use crate::termsum::{PauliTermSum, Storage};


/// Replays a run's per-gate records into the verbose log.
///
/// The schema is the one the merge-then-sweep engine wrote, kept so existing
/// `LogParser` consumers keep
/// working, but two of its fields no longer mean quite what they did and the
/// event says so:
///
/// * `terms_discarded` counted terms a sweep *removed* from the live set. This
///   engine has no sweep; it refuses a branch before the term exists, so the
///   field counts branches never created. `truncation_model` names which of the
///   two a reader is looking at.
/// * `outbox_terms` measured the emitted-but-not-yet-deduplicated backlog, which
///   only exists in a merge-then-sweep lifecycle. Duplicates are folded on
///   insert here, so it is always zero.
///
/// `discarded_coeff_l1` and `discarded_coeff_max` are reported as zero rather
/// than fabricated: computing them would mean accumulating over every declined
/// branch during the scan, billions of them on a deep circuit, and would measure
/// refused branches rather than removed terms anyway.
fn write_gate_log(
    log_filename: Option<&str>,
    log_every: usize,
    cfg: &propaq_core::truncators::ResolvedConfig,
    gates: &[engine::GateRecord],
    phases: &propaq_core::partitioned::PhaseStats,
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
    let wc = cfg.weight.map_or_else(|| "null".to_string(), |w| w.to_string());
    let cc = cfg.coefficient.unwrap_or(0.0);
    for g in gates.iter().filter(|g| g.gate_idx % every == 0) {
        let qki = g.qiskit_gate_idx.map_or_else(|| "null".to_string(), |v| v.to_string());
        let _ = writeln!(
            log,
            r#"{{"event":"gate","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"map_terms":{},"outbox_terms":0,"avg_ms_per_gate":{:.3e}}}"#,
            g.gate_idx, g.layer_idx, qki, g.terms_after, g.elapsed_ms
        );
        let _ = writeln!(
            log,
            r#"{{"event":"truncation","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"trigger":"emit","truncation_model":"emit_gate","terms_before":{},"terms_after":{},"terms_discarded":{},"discarded_coeff_l1":0.0e0,"discarded_coeff_max":0.0e0,"weight_cutoff":{},"coeff_cutoff":{:.6e},"elapsed_ms":{:.3e}}}"#,
            g.gate_idx, g.layer_idx, qki, g.terms_before, g.terms_after, g.declined, wc, cc, g.elapsed_ms
        );
    }
    write_phase_event(&mut log, phases);
    Ok(())
}

/// One closing record with the run's phase split and kernel counters.
///
/// Occupancy is `busy / (wall * partitions)`: the share of the pool doing work
/// rather than waiting at a barrier or behind a straggler. Release builds inline
/// both phases into the same rayon closure and carry no frame pointers, so this
/// record is the only place the split is visible.
fn write_phase_event(log: &mut impl Write, p: &propaq_core::partitioned::PhaseStats) {
    let occupancy = |busy: f64, wall: f64| busy / (wall.max(1e-12) * p.partitions as f64);
    let share = |part: u64, whole: u64| part as f64 / whole.max(1) as f64;
    let _ = writeln!(
        log,
        r#"{{"event":"engine_phases","partitions":{},"scan_s":{:.6e},"absorb_s":{:.6e},"claims_s":{:.6e},"scan_occupancy":{:.4},"absorb_occupancy":{:.4},"terms":{},"inline_positions":{},"overflow_rows":{},"overflow_share":{:.4},"visited":{},"emitted":{},"declined":{},"emitted_share":{:.4},"declined_share":{:.4},"exchange_hits":{},"exchange_hit_share":{:.4}}}"#,
        p.partitions,
        p.scan_seconds,
        p.absorb_seconds,
        p.claims_seconds,
        occupancy(p.scan_busy_seconds, p.scan_seconds),
        occupancy(p.absorb_busy_seconds, p.absorb_seconds),
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

/// Back-propagates Pauli observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: A list of truncators (WeightTruncator, CoefficientTruncator, TermBudget), a single such
///         truncator, a legacy TruncationPolicy (decomposed), or None. The
///         symbolic-only FrequencyTruncator is rejected.
///     schedule: Optional FlushSchedule controlling the lossless merge cadence.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
///     pin_threads: Bind each worker to its own CPU. On by default, and worth
///         8x to 32x on the partitioned engine, because a partition's store fits
///         a core's private cache only if the same core keeps serving it.
///         **Turn it off when the process also runs threaded BLAS.** A pinned
///         worker cannot step around a spinning BLAS thread, and qiskit loads
///         two OpenBLAS copies that each start one spinner per core: measured
///         449ms against 184ms at 64 threads. Setting OPENBLAS_NUM_THREADS=1
///         before numpy imports is the better fix.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliPropagator {
    inner: RunConfig,
}

#[pymethods]
impl PauliPropagator {
    /// Initialize the Pauli propagator. See the class docstring for arguments.
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None, pin_threads=true))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
        pin_threads: bool,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        reject_surrogate_only(&truncators)?;
        Ok(PauliPropagator {
            inner: RunConfig::new(noise, schedule, truncators, n_threads, progress_bar, logger, pin_threads)?,
        })
    }

    /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A PauliCircuit whose rotations are applied in reverse.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, filename=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        filename: Option<String>,
    ) -> PyResult<PauliTermSum> {
        // The engine declines only a width past its dispatch ladder, which the
        // error below reports; there is no other backend to fall through to.
        {
            let cfg = propaq_core::truncators::resolve_config(&self.inner.truncators);
            let (n, pool) = (observable.n_units(), &self.inner.pool);
            let threads = pool.current_num_threads().into();
            let noise = self.inner.noise.as_ref().map(|x| x.bind(py));
            let noise = noise.as_ref().map(|b| &**b);
            let log = self.inner.log_filename.is_some();
            let mut recs: Vec<engine::GateRecord> = Vec::new();
            let mut phases = propaq_core::partitioned::PhaseStats::default();
            let evolved = match &observable.inner {
                Storage::F64(src) => {
                    let terms: Vec<(PauliString, f64)> =
                        crate::termsum::materialize(src).into_iter().collect();
                    engine::run::<f64>(py, &terms, circuit, None, n, &cfg, pool, threads, noise, false, true, log)?
                        .map(|o| { recs = o.gates; phases = o.phases; o.terms })
                        .flatten()
                        .map(|t| {
                            let store = crate::termsum::term_sum_from_pairs(t, n);
                            (crate::termsum::materialize(&store), PauliTermSum::from_store(store))
                        })
                }
                Storage::F32(src) => {
                    let terms: Vec<(PauliString, f64)> =
                        crate::termsum::materialize(src).into_iter().collect();
                    engine::run::<f32>(py, &terms, circuit, None, n, &cfg, pool, threads, noise, false, true, log)?
                        .map(|o| { recs = o.gates; phases = o.phases; o.terms })
                        .flatten()
                        .map(|t| {
                            let store = crate::termsum::term_sum_from_pairs(t, n);
                            (crate::termsum::materialize(&store), PauliTermSum::from_store_f32(store))
                        })
                }
            };
            write_gate_log(self.inner.log_filename.as_deref(), self.inner.log_every, &cfg, &recs, &phases)?;
            if let Some((map, evolved)) = evolved {
                if let Some(path) = filename.as_deref() {
                    save_terms_to_file(&map, path)?;
                }
                return Ok(evolved);
            }
        }

        // Nothing else can serve it. The engine only declines a width past its
        // dispatch ladder, so this is a hard limit rather than a fallback.
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "propaq: {} qubits exceeds the engine's maximum of {}",
            observable.n_units(),
            engine::MAX_DISPATCH_QUBITS,
        )))
    }

    /// Compute the expectation value of *observable* in the state prepared by *circuit*.
    ///
    /// Arguments:
    ///     observable: The Pauli observable.
    ///     circuit: A PauliCircuit applied to the reference state.
    ///     initial_state: Computational basis reference state as a bitstring integer.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, initial_state=None, filename=None))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
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
            // `materialize` widens whatever the storage holds to f64 and the
            // engine narrows it back through `CoeffRepr::from_real`, so one term
            // list serves both widths; only the instantiation differs.
            let run = |terms: &[(PauliString, f64)], f32_storage: bool| {
                let (n, pool) = (observable.n_units(), &self.inner.pool);
                let threads = pool.current_num_threads().into();
                let words = initial_state.as_words();
                let log = self.inner.log_filename.is_some();
                let noise = self.inner.noise.as_ref().map(|n| n.bind(py));
                if f32_storage {
                    engine::run::<f32>(py, terms, circuit, Some(words), n, &cfg, pool, threads, noise.as_ref().map(|b| &**b), true, want_terms, log)
                        .map(|o| o.map(|o| (o.result, o.gates, o.phases, o.terms.map(|t| {
                            crate::termsum::materialize(&crate::termsum::term_sum_from_pairs(t, n))
                        }))))
                } else {
                    engine::run::<f64>(py, terms, circuit, Some(words), n, &cfg, pool, threads, noise.as_ref().map(|b| &**b), true, want_terms, log)
                        .map(|o| o.map(|o| (o.result, o.gates, o.phases, o.terms.map(|t| {
                            crate::termsum::materialize(&crate::termsum::term_sum_from_pairs(t, n))
                        }))))
                }
            };
            let served = match &observable.inner {
                Storage::F64(s) => {
                    run(&crate::termsum::materialize(s).into_iter().collect::<Vec<_>>(), false)?
                }
                Storage::F32(s) => {
                    run(&crate::termsum::materialize(s).into_iter().collect::<Vec<_>>(), true)?
                }
            };
            if let Some((result, recs, phases, terms)) = served {
                write_gate_log(self.inner.log_filename.as_deref(), self.inner.log_every, &cfg, &recs, &phases)?;
                if let (Some(path), Some(map)) = (filename.as_deref(), terms.as_ref()) {
                    save_terms_to_file(map, path)?;
                }
                return Ok(result);
            }
        }

        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "propaq: {} qubits exceeds the engine's maximum of {}",
            observable.n_units(),
            engine::MAX_DISPATCH_QUBITS,
        )))
    }

    /// The noise model used during propagation, if any.
    #[getter]
    fn noise(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner.noise.as_ref().map(|n| n.clone_ref(py))
    }

    /// Set the noise model for this propagator.
    #[pyo3(signature = (noise=None))]
    fn set_noise(&mut self, noise: Option<PyObject>) {
        self.inner.noise = noise;
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
    }

    /// The flush/merge schedule.
    #[getter]
    fn schedule(&self) -> FlushSchedule {
        self.inner.schedule.clone()
    }

    #[setter]
    fn set_schedule(&mut self, schedule: FlushSchedule) {
        self.inner.schedule = schedule;
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`); the current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        reject_surrogate_only(&truncators)?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}
