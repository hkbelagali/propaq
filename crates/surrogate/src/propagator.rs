///
/// impl for the surrogate/symbolic propagator!
///
/// This propagator now runs on the same columnar SoA engine as the
/// numerical propagators (`propaq_core::soa`), rather than the
/// hash-partition/outbox engine (`propaq_core::propagator::AbstractPropagator`)
/// it originally shared with them — this crate was the last consumer of
/// that engine, which has since been retired. Every gate application is a
/// `soa::kernels::apply_rotation` call over a `SoaTermSum<SymbolicCoeff>`;
/// merging/truncation reuse `soa::kernels::merge`/new generic `map_retain`/
/// `fold_coeffs`/`sum_coeffs` primitives instead of the old engine's
/// per-partition equivalents. None of the symbolic coefficient math below
/// changed — `SymbolicCoeff`/`Generation`/`SupportTrie`/`ExponentDict`
/// already did their interning work at flush time (not per-gate), so they
/// needed no redesign to sit under a columnar container instead of a
/// sharded hashmap.
///
/// The major difference between symbolic and numerical propagation
/// is the combinatorial explosion of terms in the symbolic propagator.
/// This is because we can merge the same term with different numerical
/// coefficients, but naively merging symbolic coefficients is not possible
/// because they might consist of different paths.
///
/// Symbolic coefficients can be compactly represented as
///             \sum_i c_i \prod_j sin(\theta_j)^{a_j} cos(\theta_j)^{b_j}
///
/// The coefficients c_i are numerical, and the monomial attributes
/// j, a_j and b_j fit into a u32. These are stored in a SoA
/// (structure of arrays) format, and coefficients lie in a shared arena.
///
/// In order to alleviate the combinatorial explosion of distinct paths,
/// the monomials are factored into their support (parameters they touch)
/// and exponents. This reveals an invariant - the support is always
/// an ascending list of parameter indices, and ideally stored as a
/// trie. This mitigates the combinatorial explosion of distinct paths,
/// and allows for efficient merging. The support is interned into
/// the trie, and the exponents are stored in pairs in a separate
/// container. Therefore, a global trie is maintained for the entire
/// propagation, shared across threads. It is updated at every flush,
/// during which the monomials are reconciled into a new generation of the
/// trie. This must be done serially, but the subsequent deduplication of
/// the coefficients is done in parallel.
///
/// For circuits with primarily numerical parameters and a few symbolic
/// parameters, deep-copying the symbolic history of the coefficients
/// is wasteful and unnecessary, since the history is unchanged from the
/// action of an anticommuting numerical gate. Therefore, the propagator
/// involves a deferred realization scheme, in which the symbolic history
/// is only realized in memory when a term anticommutes with a symbolic
/// gate. By doing so, the propagator can avoid significant memory overhead.
///
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;
use rayon::prelude::*;

use propaq_core::coeff::CoeffRepr;
use propaq_core::logger::Logger;
use propaq_core::propagator::{close_progress_bar, make_progress_bar, tick_progress_bar};
use propaq_core::soa::kernels;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::{GateParam, SymbolicCoeff};
use crate::interning::Generation;
use crate::truncation::FrequencyTruncationPolicy;
use crate::model::{SurrogateModel, SurrogateTerm, PauliSurrogateModel, MajoranaSurrogateModel};
use propaq_core::truncators::{
    resolve_config, resolve_truncation as core_resolve_truncation, FlushSchedule, ResolvedConfig,
    Truncator,
};

/// Resolve the flexible `truncation` constructor argument into `(FlushSchedule,
/// [Truncator])`. The surrogate additionally accepts the legacy
/// `FrequencyTruncationPolicy` (decomposed here); everything else, such as a list, a
/// single truncator, a core `TruncationPolicy`, or `None`, is delegated to the
/// shared `propaq_core` resolver.
fn resolve_truncation(
    truncation: Option<&Bound<'_, PyAny>>,
    schedule: Option<FlushSchedule>,
) -> PyResult<(FlushSchedule, Vec<Truncator>)> {
    if let Some(obj) = truncation {
        if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
            let (decomposed, ops) = legacy.decompose();
            return Ok((schedule.unwrap_or(decomposed), ops));
        }
    }
    core_resolve_truncation(truncation, schedule)
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
    /// The current frozen support/exponent interning generation. Coefficients'
    /// base ids reference it between flushes; `reconcile` advances it (folding
    /// each live coefficient's extension into a fresh generation) at flush
    /// barriers, and it is handed to the compiled model at the end.
    generation: Generation,
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
            generation: Generation::new(),
            _marker: PhantomData,
        })
    }

    /// Advance the interning generation: fold every live coefficient's extension
    /// into a fresh generation, replacing base ids and clearing extensions.
    fn reconcile(&mut self, evolved: &mut SoaTermSum<SymbolicCoeff>) {
        let old = std::mem::take(&mut self.generation);
        let mut new = Generation::new();
        let n = evolved.len();
        // Serial: interning mutates one shared generation.
        for c in &mut evolved.coeffs[..n] {
            c.reconcile_into_deferred(&old, &mut new);
        }
        self.generation = new;
        // Parallel: each coefficient's dedup is independent and reads no shared
        // state (it never dereferences the generation), so lift it off the
        // serial interning critical path onto all worker threads.
        self.pool.install(|| evolved.coeffs[..n].par_iter_mut().for_each(|c| c.deduplicate()));
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

        // Compact the survivors: fold their extensions into a fresh interning
        // generation. Runs after truncation so only surviving structure is
        // re-interned (dead nodes are dropped with the old generation).
        self.reconcile(evolved);

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
        // Fresh interning generation for this build (the previous build's was
        // moved into its model, but reset defensively in case one errored out).
        self.generation = Generation::new();

        // Resolve the truncation pipeline once (Copy config). The flush triggers
        // (`max_terms`/`max_monomials`) and the `min_terms` gate come from the
        // `TermBudget`/`MonomialBudget` operators; the merge cadence from the
        // schedule.
        let cfg = resolve_config(&self.truncators);

        // Look-ahead frequency-pruning cap for symbolic rotations. Skipping a
        // doomed sin-branch monomial at generation time is exactly equivalent to
        // trimming it at the next flush only when *every* flush applies the
        // frequency cap, i.e. when the `min_terms` gate is 0/None so
        // `apply_lossy` in `apply_truncation_policy` is always true. With a
        // nonzero `min_terms`, trimming is deferred, so eager pruning could
        // diverge; fall back to `None` (no look-ahead) there. Requires a
        // `FrequencyTruncator` in the pipeline to supply the cap.
        let prune_freq: Option<u32> = if cfg.min_terms.unwrap_or(0) == 0 {
            cfg.frequency.map(|f| f as u32)
        } else {
            None
        };

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
                    // Inject the look-ahead cap into symbolic params
                    let param = match SymbolicCoeff::extract_gate_param(rot)? {
                        GateParam::Symbolic { param, .. } => {
                            GateParam::Symbolic { param, prune_freq }
                        }
                        numeric => numeric,
                    };
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

        // Flush triggers from the budgets; merge cadence from the schedule.
        let max_terms: Option<usize> = cfg.max_terms;
        let max_monomials: Option<usize> = cfg.max_monomials;
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
                // Term count is a poor proxy for a symbolic coefficient's actual
                // size: a handful of terms can carry the overwhelming majority
                // of monomials while term count barely moves. Watch monomial
                // count directly so a flush still fires in that case.
                let monomials_trigger = max_monomials
                    .map_or(false, |max| self.total_monomials + pending_monomials >= max);
                let threshold_trigger = if terms_trigger || monomials_trigger {
                    Some(if monomials_trigger && !terms_trigger {
                        "monomial_threshold"
                    } else {
                        "threshold"
                    })
                } else {
                    None
                };
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
        // container is already flat).
        let n = evolved.len();
        let mut raw: Vec<SurrogateTerm<B::Term>> = Vec::new();
        for i in 0..n {
            let overlap = B::trace(evolved.term_planes(i), evolved.n_units, initial_state);
            if overlap.abs() > 1e-15 {
                // Take rather than clone: `evolved` isn't reused after this build.
                let mut coeff = std::mem::take(&mut evolved.coeffs[i]);
                coeff.deduplicate();
                let term = B::term_from_planes(evolved.term_planes(i), evolved.n_units);
                raw.push(SurrogateTerm { term, overlap, coeff });
            }
        }

        // The final `flush_and_maybe_truncate("final")` above reconciled the
        // survivors, so every coefficient's base ids reference `self.generation`.
        // Hand it to the model.
        let generation = std::mem::take(&mut self.generation);
        Ok(SurrogateModel::with_generation(raw, n_params, generation))
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

/// Apply the resolved truncation config to `propagator`'s current live state.
#[inline]
fn monomial_removal_budget(monomials_after: usize, max: usize) -> usize {
    monomials_after.saturating_sub(max)
}

/// Elementwise-add two frequency histograms, reconciling lengths. Combine step
/// for the parallel `fold_coeffs` that builds the global histogram.
fn combine_histograms(mut a: Vec<u64>, b: Vec<u64>) -> Vec<u64> {
    if a.len() < b.len() {
        a.resize(b.len(), 0);
    }
    for (slot, v) in a.iter_mut().zip(b.iter()) {
        *slot += *v;
    }
    a
}

/// Walking a frequency histogram from the highest frequency down, find the
/// boundary frequency `f*` at which a cumulative removal of `budget` monomials
/// lands, and how many must be removed from within `f*` (the remainder after
/// fully removing every higher-frequency bucket). `remove_in_boundary` is always
/// `<= hist[f*]`; it equals `hist[f*]` exactly when the whole boundary bucket is
/// consumed. Returns `(0, 0)` only when `budget` meets or exceeds every monomial
/// present (caller then removes everything at/above frequency 0).
fn boundary_from_histogram(hist: &[u64], budget: usize) -> (usize, usize) {
    let mut remaining = budget as u64;
    for f in (0..hist.len()).rev() {
        let cnt = hist[f];
        if cnt == 0 {
            continue;
        }
        if remaining <= cnt {
            return (f, remaining as usize);
        }
        remaining -= cnt;
    }
    (0, 0)
}

/// Run the truncation pipeline against `evolved`'s current live state.
pub fn apply_truncation_policy<B: SoaBasis>(
    evolved: &mut SoaTermSum<SymbolicCoeff>,
    cfg: &ResolvedConfig,
) -> TruncationOutcome {
    let total_before = evolved.len();
    let min_terms = cfg.min_terms.unwrap_or(0);
    let n_units = evolved.n_units;

    // Deferred like the numerical propagator: below min_terms, skip the lossy
    // filters and only run the lossless dedup.
    let apply_lossy = total_before >= min_terms;

    let mut monomials_after = kernels::map_retain::<B, SymbolicCoeff, _, _>(
        evolved,
        |c: &mut SymbolicCoeff| {
            if apply_lossy {
                if let Some(mf) = cfg.frequency {
                    c.trim_high_frequency(mf);
                }
            }
            c.deduplicate();
            if apply_lossy {
                if let Some(coeff) = cfg.coefficient {
                    c.trim_small_scalars(coeff);
                }
            }
        },
        |term: [&[u64]; 2], c: &SymbolicCoeff| {
            let weight_ok = !apply_lossy || cfg.weight.map_or(true, |w| B::weight(term, n_units) <= w);
            weight_ok && !c.is_empty()
        },
    );

    if let Some(max) = cfg.max_monomials {
        debug_assert!(
            max >= cfg.min_monomials.unwrap_or(0),
            "MonomialBudget misconfigured: min_monomials must not exceed max_monomials",
        );
        if monomials_after > max {
            let budget = monomial_removal_budget(monomials_after, max);
            if budget > 0 {
                // 1. Global frequency histogram in one parallel fold. Frequency
                //    is a small bounded int, so a per-worker `Vec<u64>` keyed by
                //    frequency is cheap and exact.
                let hist: Vec<u64> = kernels::fold_coeffs(
                    evolved,
                    Vec::new,
                    |mut h, c: &SymbolicCoeff| {
                        c.add_freq_histogram(&mut h);
                        h
                    },
                    combine_histograms,
                );

                // 2. Boundary frequency f* (importance = frequency desc) and how
                //    many monomials must come out of it after every higher bucket
                //    is removed in full.
                let (f_star, remove_in_boundary) = boundary_from_histogram(&hist, budget);

                // 3. Secondary key |scalar| asc, applied only within the boundary
                //    bucket. If the whole bucket is consumed, s* = INFINITY drops
                //    all of it; otherwise a single `select_nth` over the boundary
                //    bucket's |scalar| picks the cutoff, and the exact-s* ties are
                //    budget-limited so the cut lands precisely at `max`.
                let full_bucket = f_star < hist.len() && remove_in_boundary as u64 >= hist[f_star];
                let (s_star, tie_budget) = if remove_in_boundary == 0 || full_bucket {
                    (f64::INFINITY, 0usize)
                } else {
                    let mut scalars: Vec<f64> = kernels::fold_coeffs(
                        evolved,
                        Vec::new,
                        |mut v, c: &SymbolicCoeff| {
                            c.collect_boundary_scalars(f_star, &mut v);
                            v
                        },
                        |mut a, mut b| {
                            a.append(&mut b);
                            a
                        },
                    );
                    let k = remove_in_boundary.min(scalars.len());
                    scalars.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
                    let s = scalars[k - 1];
                    let n_below = scalars.iter().filter(|&&x| x < s).count();
                    (s, k.saturating_sub(n_below))
                };

                // 4. One importance-ranked removal pass.
                let tie_remaining = std::sync::atomic::AtomicUsize::new(tie_budget);
                monomials_after = kernels::map_retain::<B, SymbolicCoeff, _, _>(
                    evolved,
                    |c: &mut SymbolicCoeff| {
                        c.remove_by_rank_budgeted(f_star, s_star, &tie_remaining);
                    },
                    |_term, c: &SymbolicCoeff| !c.is_empty(),
                );
            }
        }
    }

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
///         MonomialBudget) applied at each flush, a single such truncator, a
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
mod monomial_removal_budget_tests {
    use super::{boundary_from_histogram, combine_histograms, monomial_removal_budget};

    #[test]
    fn boundary_removes_higher_buckets_first_then_partial() {
        // freq0=5, freq1=3, freq2=2. Budget 4 = all of freq2 (2) + 2 of freq1.
        let hist = vec![5, 3, 2];
        assert_eq!(boundary_from_histogram(&hist, 4), (1, 2));
    }

    #[test]
    fn boundary_consuming_exactly_one_bucket_reports_full_bucket() {
        let hist = vec![5, 3, 2];
        // remove_in_boundary == hist[f*] signals a full-bucket removal to caller.
        assert_eq!(boundary_from_histogram(&hist, 2), (2, 2));
    }

    #[test]
    fn boundary_within_top_bucket() {
        let hist = vec![5, 3, 2];
        assert_eq!(boundary_from_histogram(&hist, 1), (2, 1));
    }

    #[test]
    fn boundary_skips_empty_buckets() {
        // freq3=2 removed in full, then 1 of freq0 (freq1/freq2 empty).
        let hist = vec![5, 0, 0, 2];
        assert_eq!(boundary_from_histogram(&hist, 3), (0, 1));
    }

    #[test]
    fn boundary_budget_exceeds_all_returns_zero_zero() {
        assert_eq!(boundary_from_histogram(&[2, 2], 10), (0, 0));
        assert_eq!(boundary_from_histogram(&[], 5), (0, 0));
    }

    #[test]
    fn combine_histograms_reconciles_lengths() {
        assert_eq!(combine_histograms(vec![1, 2], vec![10, 20, 30]), vec![11, 22, 30]);
        assert_eq!(combine_histograms(vec![1, 2, 3], vec![10]), vec![11, 2, 3]);
    }

    #[test]
    fn budget_always_targets_max() {
        // The budget is exactly `after - max`, landing precisely at max.
        for (after, max) in [(100, 90), (100, 99), (1_000_000, 999_999), (11, 10)] {
            assert_eq!(monomial_removal_budget(after, max), after - max);
        }
    }

    #[test]
    fn exactly_one_over_max_wants_a_budget_of_one() {
        assert_eq!(monomial_removal_budget(91, 90), 1);
    }

    #[test]
    fn max_equal_to_after_wants_zero() {
        assert_eq!(monomial_removal_budget(999, 999), 0);
    }
}

#[cfg(test)]
mod numeric_history_dedup_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// A purely-numeric gate history produces only empty-mask monomials, so the
    /// live monomial count must never exceed the live term count once a merge
    /// has run (each term collapses to a single monomial). This is the regime
    /// the user reported: dozens of terms but the monomial count exploding into
    /// the millions because the lossless merge only deduped term keys, never the
    /// identical monomials `add_assign` piled up inside each coefficient. With
    /// `CoeffRepr::post_merge` calling `deduplicate` at the merge, live
    /// monomials track live terms exactly.
    #[test]
    fn numeric_gates_keep_live_monomials_bounded_by_term_count() {
        const N_QUBITS: usize = 8;
        let mut evolved: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);

        // Seed a single weight-1 Z on qubit 0.
        let (gx, gz) = planes_of(0, 1, 1);
        evolved.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        // Brick-wall of two-qubit generators, all with *numeric* angles, with a
        // lossless merge every few gates (mirrors `merge_max_terms` firing).
        let mut gate_idx = 0u32;
        let mut peak_live_monomials = 0usize;
        for round in 0..24 {
            let offset = round % 2;
            for q in (offset..N_QUBITS - 1).step_by(2) {
                // A weight-2 generator that anticommutes with Z-type terms on
                // these qubits (X components), so branches are actually created.
                let (genx, genz) = planes_of((1 << q) | (1 << (q + 1)), 0, 1);
                let angle = 0.3 + 0.1 * gate_idx as f64;
                kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                    &mut evolved,
                    [&genx, &genz],
                    &GateParam::Numeric { angle },
                    false,
                );
                gate_idx += 1;
            }
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut evolved);

            // After a merge, every coefficient is deduplicated; with only empty
            // masks in play, that is at most one monomial per live term.
            let live_terms = evolved.len();
            let live_monomials = kernels::sum_coeffs(&evolved, |c| c.monomial_count());
            peak_live_monomials = peak_live_monomials.max(live_monomials);
            assert!(
                live_monomials <= live_terms,
                "round {round}: live monomials {live_monomials} exceeded live terms {live_terms} \
                 , numeric-history monomials were not deduplicated at the merge",
            );
        }

        // Sanity floor: the run actually exercised real branching (non-trivial
        // term count), so the bound above wasn't vacuously true on an empty map.
        assert!(peak_live_monomials > 1, "test did not exercise any branching");
    }
}

#[cfg(test)]
mod shared_parameter_dedup_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    #[test]
    fn symbolic_gates_on_few_shared_parameters_keep_monomials_polynomial_not_exponential() {
        const N_QUBITS: usize = 8;
        const N_PARAMS: usize = 3;
        let mut evolved: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);

        let (gx, gz) = planes_of(0, 1, 1);
        evolved.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        // Brick-wall of two-qubit generators, all *symbolic*, cycling through
        // only `N_PARAMS` distinct parameter indices every parameter is
        // reused dozens of times over the run.
        let mut gate_idx: u32 = 0;
        let mut param_counts = [0usize; N_PARAMS];
        let mut peak_live_monomials = 0usize;
        for round in 0..30 {
            let offset = round % 2;
            for q in (offset..N_QUBITS - 1).step_by(2) {
                let (genx, genz) = planes_of((1 << q) | (1 << (q + 1)), 0, 1);
                let param = gate_idx as usize % N_PARAMS;
                kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(
                    &mut evolved,
                    [&genx, &genz],
                    &GateParam::Symbolic { param: param as u32, prune_freq: None },
                    false,
                );
                param_counts[param] += 1;
                gate_idx += 1;
            }
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut evolved);

            let live_terms = evolved.len();
            let live_monomials = kernels::sum_coeffs(&evolved, |c| c.monomial_count());
            peak_live_monomials = peak_live_monomials.max(live_monomials);

            let max_monomials_per_term: usize = param_counts.iter().map(|&k| k + 1).product();
            assert!(
                live_monomials <= max_monomials_per_term * live_terms.max(1),
                "round {round}: live monomials {live_monomials} exceeded the polynomial \
                 bound {max_monomials_per_term} * {live_terms} terms,  same-parameter \
                 branches were not collapsing into trig powers",
            );
        }

        // Sanity floor: real branching happened, and the run went through far
        // more gates than `2^gate_idx` monomials (the old scheme's bound)
        // could ever let fit in memory, yet stayed within the polynomial
        // bound above throughout.
        assert!(peak_live_monomials > 1, "test did not exercise any branching");
        assert!(gate_idx as usize > 4 * N_PARAMS, "test should reuse each parameter many times");
    }
}
