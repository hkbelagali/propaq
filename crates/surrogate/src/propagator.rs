///
/// impl for the surrogate/symbolic propagator!
///
/// This propagator runs on the same columnar SoA engine as the numerical
/// propagators (`propaq_core::soa`): every gate application is a
/// `soa::kernels::apply_rotation` call over a `SoaTermSum<SymbolicCoeff>`;
/// merging/truncation reuse `soa::kernels::merge`/`map_retain` the same way
/// the numerical propagators do.
///
/// `SymbolicCoeff` (`crate::symcoeff`) represents a coefficient as a
/// persistent DAG (`Scalar`/`Add`/`Scale`/`Cos`/`Sin` nodes, built via `Arc`),
/// not an expanded monomial list — every gate application and every merge is
/// O(1) regardless of how large a coefficient's prior history already is, no
/// monomial ever touched on the hot path. This replaced an earlier CSR/trie
/// design (interned support/exponent tables reconciled at every flush) after
/// real profiling showed that design's per-gate cost scaling with live
/// monomial count was the dominant source of allocator/thread-contention
/// overhead in practice (see `propaq.MD`/project memory for the diagnosis
/// and the ProPauli reference design this port is based on).
///
/// Monomial-level truncation (`FrequencyTruncator`/`CoefficientTruncator`,
/// both of which need an expanded per-monomial view a lazy DAG doesn't have)
/// is not supported — configuring one raises a clear error rather than being
/// silently ignored. Only term-level truncation (`WeightTruncator`,
/// `TermBudget`) is honored; there is no monomial-count-based budget
/// (`MonomialBudget` was removed — judged unnecessary given `WeightTruncator`/
/// `TermBudget` plus the now-default eager merge cadence).
///
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;

use propaq_core::coeff::CoeffRepr;
use propaq_core::logger::Logger;
use propaq_core::propagator::{close_progress_bar, make_progress_bar, tick_progress_bar};
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::{GateParam, SymbolicCoeff};
use crate::truncation::FrequencyTruncationPolicy;
use crate::model::{SurrogateModel, SurrogateTerm, PauliSurrogateModel, MajoranaSurrogateModel};
use propaq_core::truncators::{
    resolve_config, resolve_truncation as core_resolve_truncation, FlushSchedule, ResolvedConfig,
    Truncator,
};

/// `FrequencyTruncator`/`CoefficientTruncator` both need a per-monomial view
/// of a symbolic coefficient's history, which the DAG representation doesn't
/// expose. Rejected explicitly rather than silently ignored, mirroring
/// `propaq_core::truncators::reject_surrogate_only`'s pattern for the
/// numerical propagator's analogous rejection of surrogate-only operators.
fn reject_phase_a_unsupported(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(|t| matches!(t, Truncator::Frequency(_) | Truncator::Coefficient(_))) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "FrequencyTruncator and CoefficientTruncator are not yet supported by the surrogate \
             propagator (monomial-level truncation is being redesigned around a new coefficient \
             representation); use WeightTruncator / TermBudget for now.",
        ));
    }
    Ok(())
}

/// Resolve the flexible `truncation` constructor argument into `(FlushSchedule,
/// [Truncator])`. The surrogate additionally accepts the legacy
/// `FrequencyTruncationPolicy` (decomposed here); everything else, such as a list, a
/// single truncator, a core `TruncationPolicy`, or `None`, is delegated to the
/// shared `propaq_core` resolver. Either path rejects Phase-A-unsupported
/// (monomial-level) truncators -- see `reject_phase_a_unsupported`.
fn resolve_truncation(
    truncation: Option<&Bound<'_, PyAny>>,
    schedule: Option<FlushSchedule>,
) -> PyResult<(FlushSchedule, Vec<Truncator>)> {
    if let Some(obj) = truncation {
        if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
            let (decomposed, ops) = legacy.decompose();
            reject_phase_a_unsupported(&ops)?;
            return Ok((schedule.unwrap_or(decomposed), ops));
        }
    }
    let (schedule, ops) = core_resolve_truncation(truncation, schedule)?;
    reject_phase_a_unsupported(&ops)?;
    Ok((schedule, ops))
}

/// Surrogate propagator: drives a `SoaTermSum<SymbolicCoeff>` directly via
/// `soa::kernels`, generic over the basis (`PauliBasis`/`MajoranaBasis`).
/// Holds its own thread pool rather than wrapping `AbstractPropagator` — the
/// same shape as the numerical `SoaPropagator<B>`, plus the surrogate-only
/// interning/monomial-tracking state.
pub struct SurrogatePropagator<B: SoaBasis> {
    pool: Arc<rayon::ThreadPool>,
    /// Flush/merge cadence (when to truncate), separate from the operators.
    pub schedule: FlushSchedule,
    /// The truncation pipeline: operators applied (after the always-on dedup)
    /// at every flush, in list order.
    pub truncators: Vec<Truncator>,
    progress_bar: bool,
    verbose_log: Option<BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
    current_qiskit_gate_idx: Option<usize>,
    /// Total monomial count across all live coefficients. Like the live term
    /// count, this is only refreshed at flush points (recomputing it every
    /// gate would require a full O(total_terms) pass, unlike the O(1)
    /// term-count read via `SoaTermSum::len`).
    total_monomials: usize,
    _marker: PhantomData<B>,
}

impl<B: SoaBasis> SurrogatePropagator<B>
where
    B::Term: AbstractTerm + for<'py> FromPyObject<'py>,
{
    pub fn new(
        schedule: FlushSchedule,
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
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
            Some(obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(SurrogatePropagator {
            pool,
            schedule,
            truncators,
            progress_bar,
            verbose_log: None,
            log_filename,
            log_every,
            last_log_instant: None,
            last_log_gate_idx: 0,
            current_qiskit_gate_idx: None,
            total_monomials: 0,
            _marker: PhantomData,
        })
    }

    fn open_log(&mut self) -> PyResult<()> {
        if let Some(ref filename) = self.log_filename {
            let f = OpenOptions::new()
                .create(true).write(true).truncate(true)
                .open(filename)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.verbose_log = Some(BufWriter::new(f));
        }
        self.last_log_instant = None;
        self.last_log_gate_idx = 0;
        self.current_qiskit_gate_idx = None;
        Ok(())
    }

    fn flush_and_maybe_truncate(
        &mut self,
        evolved: &mut SoaTermSum<SymbolicCoeff>,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();
        let pool = Arc::clone(&self.pool);

        pool.install(|| kernels::merge::<B, SymbolicCoeff>(evolved));
        // Only needed for the verbose log line below; skip the O(total_terms)
        // pass entirely when logging is off.
        let monomials_before = if self.verbose_log.is_some() {
            pool.install(|| kernels::sum_coeffs(evolved, |c| c.monomial_count()))
        } else {
            0
        };

        let cfg = resolve_config(&self.truncators);
        let outcome = pool.install(|| apply_truncation_policy::<B>(evolved, &cfg));
        self.total_monomials = outcome.monomials_after;

        if self.verbose_log.is_some() {
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let qki = match self.current_qiskit_gate_idx {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            };
            let mf_str = outcome.frequency.map_or_else(|| "null".to_string(), |v| v.to_string());
            let wc_str = outcome.weight.map_or_else(|| "null".to_string(), |v| v.to_string());
            let mas_str = outcome.coefficient.map_or_else(|| "null".to_string(), |v| format!("{v:.3e}"));
            let terms_discarded = outcome.total_before - outcome.total_after;
            let monomials_discarded = monomials_before - outcome.monomials_after;
            let (total_before, total_after, monomials_after) =
                (outcome.total_before, outcome.total_after, outcome.monomials_after);
            if let Some(ref mut log) = self.verbose_log {
                let _ = writeln!(
                    log,
                    r#"{{"event":"surrogate_flush","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{terms_discarded},"monomials_before":{monomials_before},"monomials_after":{monomials_after},"monomials_discarded":{monomials_discarded},"frequency":{mf_str},"weight":{wc_str},"coefficient":{mas_str},"elapsed_ms":{elapsed_ms:.3e}}}"#
                );
            }
        }
    }

    /// Run surrogate propagation and return the compiled model.
    ///
    /// `evolved` is the observable's `SoaTermSum<SymbolicCoeff>` (seeded from
    /// the numerical observable via `SoaTermSum::map_coeffs` by the caller);
    /// `circuit` is a `SurrogatePauliCircuit` / `SurrogateMajoranaCircuit` Python object;
    /// `initial_state` is the Fock state for structural filtering;
    /// `n_params` is the total parameter count (determines lut size at evaluate time).
    pub fn run_build(
        &mut self,
        py: Python<'_>,
        evolved: &mut SoaTermSum<SymbolicCoeff>,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
        n_params: usize,
    ) -> PyResult<SurrogateModel<B::Term>> {
        self.open_log()?;

        // Resolve the truncation pipeline once (Copy config). The flush trigger
        // (`max_terms`) and the `min_terms` gate come from the `TermBudget`
        // operator; the merge cadence from the schedule.
        let cfg = resolve_config(&self.truncators);

        let n_units = evolved.n_units;
        let stride = evolved.stride;

        // Extract circuit data from Python: each rotation is either
        // symbolic (`param_index`) or numeric (`angle`), see `GateParam`.
        // The generator's plane words are extracted once here (rather than
        // re-extracting the Python object every gate application), same as
        // the numerical `SoaPropagator`.
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let circuit_data: Vec<Vec<(Vec<u64>, Vec<u64>, GateParam, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<_> {
                    let rot = rot_obj.bind(py);
                    let generator: B::Term = rot.getattr("generator")?.extract()?;
                    let param = SymbolicCoeff::extract_gate_param(rot)?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    let mut gen0 = vec![0u64; stride];
                    let mut gen1 = vec![0u64; stride];
                    B::term_into_planes(&generator, n_units, [&mut gen0, &mut gen1]);
                    Ok((gen0, gen1, param, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Each symbolic rotation carries its parameter index directly, written
        // into every branching monomial's factor run, no gate numbering or
        // `gate -> param` table is needed (a parameter reused across gates
        // accumulates into a trig power at build time; see `SymbolicCoeff`).
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();

        let (pbar, postfix) = make_progress_bar(py, self.progress_bar, total_rotations)?;

        // Flush trigger from the budget; merge cadence from the schedule.
        let max_terms: Option<usize> = cfg.max_terms;
        let merge_max_terms: Option<usize> = self.schedule.merge_max_terms;

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let mut pending_monomials: usize = 0;
        let mut deferred_threshold_trigger: Option<&'static str> = None;
        // Deduplicated count as of the last merge, for the gate log's
        // `map_terms`/`outbox_terms` split (mirrors the numerical `SoaPropagator`
        // — there's no physical partition/outbox distinction in a flat SoA
        // array, only "merged" vs "appended since the last merge").
        let mut merged_len = evolved.len();

        let pool = Arc::clone(&self.pool);
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            // Note: no per-layer noise here. The old engine's
            // `self.inner.uniform_damping(py)` always saw `noise: None` for the
            // surrogate propagator (neither `PauliSurrogatePropagator` nor
            // `MajoranaSurrogatePropagator` ever exposed a way to set it — no
            // `noise`/`set_noise` on either pyclass), so that branch was
            // unreachable dead code; removed rather than carried forward.

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (gen0, gen1, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let gen = [gen0.as_slice(), gen1.as_slice()];
                let before = evolved.len();
                let added = py.allow_threads(|| {
                    pool.install(|| kernels::apply_rotation::<B, SymbolicCoeff>(evolved, gen, param, false))
                });
                let added_monomials: usize =
                    evolved.coeffs[before..before + added].iter().map(|c| c.size_hint()).sum();
                pending += added;
                pending_monomials += added_monomials;

                self.current_qiskit_gate_idx = *qiskit_gate_idx;

                if self.verbose_log.is_some() && gate_idx % self.log_every == 0 {
                    let now = std::time::Instant::now();
                    let avg_ms_str = match self.last_log_instant {
                        Some(last) => {
                            let gates = (gate_idx - self.last_log_gate_idx).max(1);
                            format!("{:.6e}", last.elapsed().as_secs_f64() * 1000.0 / gates as f64)
                        }
                        None => "null".to_string(),
                    };
                    self.last_log_instant = Some(now);
                    self.last_log_gate_idx = gate_idx;
                    let outbox_terms = evolved.len() - merged_len;
                    // Live count: the last-flush total plus monomials added since
                    // (both O(1) reads, no O(total_terms) pass). `pending_monomials`
                    // survives a lossless merge, so this stays accurate between
                    // truncation flushes too.
                    let monomials = self.total_monomials + pending_monomials;
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{merged_len},"outbox_terms":{outbox_terms},"monomials":{monomials},"avg_ms_per_gate":{avg_ms_str}}}"#
                        );
                    }
                }

                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, _, ni, _)| *ni);
                let terms_trigger = max_terms.map_or(false, |max| evolved.len() >= max);
                let threshold_trigger = terms_trigger.then_some("threshold");
                let pending_trigger = deferred_threshold_trigger.or(threshold_trigger);
                if let Some(trigger) = pending_trigger {
                    if !next_is_intermediate {
                        py.allow_threads(|| self.flush_and_maybe_truncate(evolved, gate_idx, layer_idx, trigger));
                        pending = 0;
                        pending_monomials = 0;
                        deferred_threshold_trigger = None;
                        merged_len = evolved.len();
                    } else if threshold_trigger.is_some() && deferred_threshold_trigger.is_none() {
                        deferred_threshold_trigger = Some(trigger);
                        if self.verbose_log.is_some() {
                            let live_terms = evolved.len();
                            let live_monomials = self.total_monomials + pending_monomials;
                            let qki = self
                                .current_qiskit_gate_idx
                                .map_or_else(|| "null".to_string(), |v| v.to_string());
                            if let Some(ref mut log) = self.verbose_log {
                                let _ = writeln!(
                                    log,
                                    r#"{{"event":"surrogate_flush_deferred","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms":{live_terms},"monomials":{live_monomials},"reason":"intermediate_boundary"}}"#
                                );
                            }
                        }
                    }
                } else if !next_is_intermediate
                    && merge_max_terms.map_or(false, |m| pending >= m)
                {
                    let terms_before = evolved.len();
                    py.allow_threads(|| pool.install(|| kernels::merge::<B, SymbolicCoeff>(evolved)));
                    pending = 0;
                    self.total_monomials = py.allow_threads(|| {
                        pool.install(|| kernels::sum_coeffs(evolved, |c| c.monomial_count()))
                    });
                    pending_monomials = 0;
                    merged_len = evolved.len();
                    if self.verbose_log.is_some() {
                        let terms_after = evolved.len();
                        let monomials_after = self.total_monomials;
                        let qki = self
                            .current_qiskit_gate_idx
                            .map_or_else(|| "null".to_string(), |v| v.to_string());
                        if let Some(ref mut log) = self.verbose_log {
                            let _ = writeln!(
                                log,
                                r#"{{"event":"surrogate_merge","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"terms_before":{terms_before},"terms_after":{terms_after},"monomials_after":{monomials_after}}}"#
                            );
                        }
                    }
                }

                if let Some(ref pf) = postfix {
                    // Live count: last-flush total plus monomials added since (O(1) reads).
                    pf.bind(py).set_item("monomials", self.total_monomials + pending_monomials)?;
                }
                tick_progress_bar(py, &pbar, &postfix, evolved.len())?;
                gate_idx += 1;
            }
        }

        close_progress_bar(py, &pbar)?;

        py.allow_threads(|| self.flush_and_maybe_truncate(evolved, gate_idx, circuit_data.len(), "final"));

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        // Compile: collect terms with nonzero structural overlap, reading
        // straight out of the final SoA columns (no hashmap drain — the
        // container is already flat). Each surviving coefficient's DAG is
        // compiled once here into a flat evaluation tape (`CompiledCoeff`) —
        // this is the only point in Phase A where a coefficient's structure
        // is walked/flattened.
        let n = evolved.len();
        let mut raw: Vec<SurrogateTerm<B::Term>> = Vec::new();
        for i in 0..n {
            let overlap = B::trace(evolved.term_planes(i), evolved.n_units, initial_state);
            if overlap.abs() > 1e-15 {
                // Take rather than clone: `evolved` isn't reused after this build.
                let coeff = std::mem::take(&mut evolved.coeffs[i]).compile();
                let term = B::term_from_planes(evolved.term_planes(i), evolved.n_units);
                raw.push(SurrogateTerm { term, overlap, coeff });
            }
        }

        Ok(SurrogateModel::new(raw, n_params))
    }
}

/// Result of one `apply_truncation_policy` call, for logging/reporting.
pub struct TruncationOutcome {
    pub total_before: usize,
    pub total_after: usize,
    pub monomials_after: usize,
    pub frequency: Option<usize>,
    pub weight: Option<u32>,
    pub coefficient: Option<f64>,
}

/// Run the truncation pipeline against `evolved`'s current live state.
///
/// Term-level only: `WeightTruncator` (operator weight) and the always-on
/// lossless dedup (`SymbolicCoeff::add_assign`/`is_empty` via `merge`,
/// already applied by the caller before this runs). Monomial-level
/// truncation (`FrequencyTruncator`, symbolic `CoefficientTruncator`) is
/// rejected at construction time (`reject_phase_a_unsupported`); there is no
/// monomial-count budget at all (`MonomialBudget` was removed as
/// unnecessary).
pub fn apply_truncation_policy<B: SoaBasis>(
    evolved: &mut SoaTermSum<SymbolicCoeff>,
    cfg: &ResolvedConfig,
) -> TruncationOutcome {
    let total_before = evolved.len();
    let min_terms = cfg.min_terms.unwrap_or(0);
    let n_units = evolved.n_units;

    // Deferred like the numerical propagator: below min_terms, skip the lossy
    // filters.
    let apply_lossy = total_before >= min_terms;

    let monomials_after = kernels::map_retain::<B, SymbolicCoeff, _, _>(
        evolved,
        |_c: &mut SymbolicCoeff| {},
        |term: [&[u64]; 2], c: &SymbolicCoeff| {
            let weight_ok = !apply_lossy || cfg.weight.map_or(true, |w| B::weight(term, n_units) <= w);
            weight_ok && !c.is_empty()
        },
    );

    let total_after = evolved.len();
    TruncationOutcome {
        total_before,
        total_after,
        monomials_after,
        frequency: cfg.frequency,
        weight: cfg.weight,
        coefficient: cfg.coefficient,
    }
}

use propaq_pauli::string::PauliBasis;
use propaq_pauli::termsum::PauliTermSum;
use propaq_majorana::monomial::MajoranaBasis;
use propaq_majorana::termsum::MajoranaTermSum;

/// Back-propagates Pauli observables symbolically, producing a compiled model
/// that can be re-evaluated for any parameter assignment.
///
/// Arguments:
///     truncation: A list of truncator objects
///         (FrequencyTruncator, CoefficientTruncator, WeightTruncator,
///         TermBudget) applied at each flush, a single such truncator, a
///         legacy FrequencyTruncationPolicy (decomposed automatically), or None.
///     schedule: Optional FlushSchedule controlling flush/merge cadence. Omitted
///         -> sensible defaults when any truncator is given, or "flush only at the
///         end" when truncation is also None. A legacy policy supplies its own
///         schedule unless one is passed explicitly here.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliBasis>,
}

#[pymethods]
impl PauliSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        Ok(PauliSurrogatePropagator {
            inner: SurrogatePropagator::new(schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    /// Compile the observable back-propagated through the circuit into a SurrogateModel.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A SurrogatePauliCircuit.
    ///     initial_state: Fock state as a bitstring integer (default 0).
    #[pyo3(signature = (observable, circuit, initial_state=0))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
    ) -> PyResult<PauliSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let mut evolved = observable.inner.map_coeffs(|c| SymbolicCoeff::from_real(*c));
        let model = self.inner.run_build(py, &mut evolved, circuit, initial_state, n_params)?;
        Ok(PauliSurrogateModel { inner: model })
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
    /// constructor's `truncation`). The current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}

/// Back-propagates Majorana observables symbolically.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogatePropagator {
    inner: SurrogatePropagator<MajoranaBasis>,
}

#[pymethods]
impl MajoranaSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    #[pyo3(signature = (observable, circuit, initial_state=0))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
    ) -> PyResult<MajoranaSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let mut evolved = observable.inner.map_coeffs(|c| SymbolicCoeff::from_real(*c));
        let model = self.inner.run_build(py, &mut evolved, circuit, initial_state, n_params)?;
        Ok(MajoranaSurrogateModel { inner: model })
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
    /// constructor's `truncation`). The current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}

#[cfg(test)]
mod numeric_history_dedup_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;
    use propaq_pauli::string::PauliString;
    use std::collections::HashMap;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// Read every live term's key (via `PauliBasis::term_from_planes`) and its
    /// `f64` coefficient into a map, for comparing two independently-evolved
    /// `SoaTermSum`s by term identity rather than by row order (`merge`/
    /// `apply_rotation` don't guarantee the two representations end up in the
    /// same row order even when driven by an identical gate sequence).
    fn f64_values(terms: &SoaTermSum<f64>) -> HashMap<PauliString, f64> {
        (0..terms.len())
            .map(|i| (PauliBasis::term_from_planes(terms.term_planes(i), terms.n_units), *terms.coeff(i)))
            .collect()
    }

    fn symbolic_values(terms: &SoaTermSum<SymbolicCoeff>, lut: &[f64]) -> HashMap<PauliString, f64> {
        (0..terms.len())
            .map(|i| {
                let key = PauliBasis::term_from_planes(terms.term_planes(i), terms.n_units);
                (key, terms.coeff(i).compile().evaluate(lut))
            })
            .collect()
    }

    /// A purely-numeric gate history: `SymbolicCoeff`'s `Scale` nodes are
    /// exactly the same arithmetic as `f64`'s own `apply_rotation` (see
    /// `symcoeff.rs`'s `apply_rotation_numeric_scalar_matches_f64_apply_rotation`
    /// unit test for the single-gate version of this same property). This
    /// drives the real kernels (`apply_rotation`/`merge`) through a whole
    /// brick-wall circuit on both a plain-`f64` `SoaTermSum` and a
    /// `SymbolicCoeff` one and checks every surviving term's compiled/evaluated
    /// value agrees with the trusted numeric engine's own accumulation --
    /// i.e. that `add_assign`/`Scale` compose correctly end-to-end, not just
    /// for one gate in isolation.
    #[test]
    fn numeric_gate_history_matches_the_plain_f64_engine() {
        const N_QUBITS: usize = 8;
        let mut numeric: SoaTermSum<f64> = SoaTermSum::new(N_QUBITS, 1);
        let mut symbolic: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);

        let (gx, gz) = planes_of(0, 1, 1);
        numeric.push([&gx, &gz], 1.0);
        symbolic.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        let mut gate_idx = 0u32;
        for round in 0..24 {
            let offset = round % 2;
            for q in (offset..N_QUBITS - 1).step_by(2) {
                let (genx, genz) = planes_of((1 << q) | (1 << (q + 1)), 0, 1);
                let angle = 0.3 + 0.1 * gate_idx as f64;
                kernels::apply_rotation::<PauliBasis, f64>(&mut numeric, [&genx, &genz], &angle, false);
                kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                    &mut symbolic,
                    [&genx, &genz],
                    &GateParam::Numeric { angle },
                    false,
                );
                gate_idx += 1;
            }
            kernels::merge::<PauliBasis, f64>(&mut numeric);
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut symbolic);
        }

        let expected = f64_values(&numeric);
        let got = symbolic_values(&symbolic, &[]);
        assert_eq!(got.len(), expected.len(), "live term sets diverged between the two engines");
        assert!(expected.len() > 1, "test did not exercise any branching");
        for (key, &want) in &expected {
            let have = got.get(key).expect("term missing from symbolic result");
            assert!(
                (have - want).abs() < 1e-9 * want.abs().max(1.0),
                "symbolic {have} vs f64 reference {want}",
            );
        }
    }
}

#[cfg(test)]
mod shared_parameter_dedup_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;
    use propaq_pauli::string::PauliString;
    use std::collections::HashMap;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    fn f64_values(terms: &SoaTermSum<f64>) -> HashMap<PauliString, f64> {
        (0..terms.len())
            .map(|i| (PauliBasis::term_from_planes(terms.term_planes(i), terms.n_units), *terms.coeff(i)))
            .collect()
    }

    fn symbolic_values(terms: &SoaTermSum<SymbolicCoeff>, lut: &[f64]) -> HashMap<PauliString, f64> {
        (0..terms.len())
            .map(|i| {
                let key = PauliBasis::term_from_planes(terms.term_planes(i), terms.n_units);
                (key, terms.coeff(i).compile().evaluate(lut))
            })
            .collect()
    }

    /// Symbolic gates reusing a handful of shared parameters across many gates
    /// must evaluate identically to running the same circuit with each
    /// parameter's angle fixed as a concrete number throughout -- exercising
    /// `Cos`/`Sin`/`Add` composing correctly under heavy subtree sharing (the
    /// same parameter's `Arc<Node>` reused by dozens of gates), not just a
    /// bound on monomial count.
    #[test]
    fn shared_parameter_history_matches_f64_engine_under_fixed_angles() {
        const N_QUBITS: usize = 8;
        const N_PARAMS: usize = 3;
        let fixed_angles: [f64; N_PARAMS] = [0.41, 1.13, 2.02];

        let mut numeric: SoaTermSum<f64> = SoaTermSum::new(N_QUBITS, 1);
        let mut symbolic: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);

        let (gx, gz) = planes_of(0, 1, 1);
        numeric.push([&gx, &gz], 1.0);
        symbolic.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        let mut gate_idx: u32 = 0;
        for round in 0..30 {
            let offset = round % 2;
            for q in (offset..N_QUBITS - 1).step_by(2) {
                let (genx, genz) = planes_of((1 << q) | (1 << (q + 1)), 0, 1);
                let param = gate_idx as usize % N_PARAMS;
                kernels::apply_rotation::<PauliBasis, f64>(
                    &mut numeric,
                    [&genx, &genz],
                    &fixed_angles[param],
                    false,
                );
                kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                    &mut symbolic,
                    [&genx, &genz],
                    &GateParam::Symbolic { param: param as u32 },
                    false,
                );
                gate_idx += 1;
            }
            kernels::merge::<PauliBasis, f64>(&mut numeric);
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut symbolic);
        }

        let lut: Vec<f64> = fixed_angles.iter().flat_map(|&t| [t.cos(), t.sin()]).collect();
        let expected = f64_values(&numeric);
        let got = symbolic_values(&symbolic, &lut);
        assert_eq!(got.len(), expected.len(), "live term sets diverged between the two engines");
        assert!(expected.len() > 1, "test did not exercise any branching");
        assert!(gate_idx as usize > 4 * N_PARAMS, "test should reuse each parameter many times");
        for (key, &want) in &expected {
            let have = got.get(key).expect("term missing from symbolic result");
            assert!(
                (have - want).abs() < 1e-8 * want.abs().max(1.0),
                "symbolic {have} vs f64 reference {want}",
            );
        }
    }
}
