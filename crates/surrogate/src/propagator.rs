use std::io::{BufWriter, Write};
use std::fs::OpenOptions;

use pyo3::prelude::*;

use propaq_core::coeff::CoeffRepr;
use propaq_core::propagator::AbstractPropagator;
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::{GateParam, SymbolicCoeff};
use crate::truncation::{
    CoefficientTruncator, FlushSchedule, FrequencyTruncationPolicy, FrequencyTruncator,
    MonomialBudget, Truncator, WeightTruncator,
};
use crate::model::{SurrogateModel, SurrogateTerm, PauliSurrogateModel, MajoranaSurrogateModel};

/// Resolve the flexible `truncation` constructor argument (which may be a legacy
/// `FrequencyTruncationPolicy`, a Python list of individual truncators, a single
/// truncator, or `None`) together with an optional explicit `schedule` into the
/// internal `(FlushSchedule, [Truncator])` pair the propagator runs.
///
/// - a legacy `FrequencyTruncationPolicy` decomposes into a schedule + operators
///   (an explicit `schedule` argument, if any, overrides the decomposed one);
/// - a list/single truncator uses the explicit `schedule` or the standard
///   defaults;
/// - `None` truncation with no schedule means "flush only at the end" (all
///   triggers off), matching the old `truncation=None` behavior.
fn resolve_truncation(
    truncation: Option<&Bound<'_, PyAny>>,
    schedule: Option<FlushSchedule>,
) -> PyResult<(FlushSchedule, Vec<Truncator>)> {
    let Some(obj) = truncation else {
        return Ok((schedule.unwrap_or_else(FlushSchedule::none), Vec::new()));
    };
    if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
        let (decomposed, ops) = legacy.decompose();
        return Ok((schedule.unwrap_or(decomposed), ops));
    }
    if let Ok(ops) = obj.extract::<Vec<Truncator>>() {
        return Ok((schedule.unwrap_or_default(), ops));
    }
    if let Ok(one) = obj.extract::<Truncator>() {
        return Ok((schedule.unwrap_or_default(), vec![one]));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "truncation must be a FrequencyTruncationPolicy, a truncator \
         (FrequencyTruncator/CoefficientTruncator/WeightTruncator/MonomialBudget), \
         a list of truncators, or None",
    ))
}

/// Generic surrogate propagator wrapping `AbstractPropagator<M, SymbolicCoeff>`.
///
/// Cannot write `impl<M> AbstractPropagator<M, SymbolicCoeff>` here because
/// `AbstractPropagator` is a foreign type; instead we wrap and delegate.
pub struct SurrogatePropagator<M: AbstractTerm> {
    pub inner: AbstractPropagator<M, SymbolicCoeff>,
    /// Flush/merge cadence (when to truncate), separate from the operators.
    pub schedule: FlushSchedule,
    /// The truncation pipeline: operators applied (after the always-on dedup)
    /// at every flush, in list order.
    pub truncators: Vec<Truncator>,
    verbose_log: Option<BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
    current_qiskit_gate_idx: Option<usize>,
    /// Total monomial count across all live coefficients. Like `total_terms`,
    /// this is only refreshed at flush points (recomputing it every gate would
    /// require a full O(total_terms) pass, unlike the O(1) term-count read).
    total_monomials: usize,
}

impl<M: AbstractTerm + for<'py> FromPyObject<'py>> SurrogatePropagator<M> {
    pub fn new(
        schedule: FlushSchedule,
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        use propaq_core::logger::Logger;
        let (log_filename, log_every) = match &logger {
            Some(obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        // Inner propagator carries no noise/truncation — surrogate manages its own.
        let inner = AbstractPropagator::new(None, None, n_threads, progress_bar, logger)?;
        Ok(SurrogatePropagator {
            inner,
            schedule,
            truncators,
            verbose_log: None,
            log_filename,
            log_every,
            last_log_instant: None,
            last_log_gate_idx: 0,
            current_qiskit_gate_idx: None,
            total_monomials: 0,
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
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();

        self.inner.flush_outboxes_to_maps();
        // Only needed for the verbose log line below; skip the O(total_terms)
        // pass entirely when logging is off.
        let monomials_before = if self.verbose_log.is_some() {
            self.inner.sum_coeffs(|c| c.monomial_count())
        } else {
            0
        };

        let outcome = apply_truncation_policy(&mut self.inner, &self.schedule, &self.truncators);
        self.total_monomials = outcome.monomials_after;

        if self.verbose_log.is_some() {
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let qki = match self.current_qiskit_gate_idx {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            };
            let mf_str = outcome.max_frequency.map_or_else(|| "null".to_string(), |v| v.to_string());
            let wc_str = outcome.weight_cutoff.map_or_else(|| "null".to_string(), |v| v.to_string());
            let mas_str = outcome.min_abs_scalar.map_or_else(|| "null".to_string(), |v| format!("{v:.3e}"));
            let terms_discarded = outcome.total_before - outcome.total_after;
            let monomials_discarded = monomials_before - outcome.monomials_after;
            let (total_before, total_after, monomials_after) =
                (outcome.total_before, outcome.total_after, outcome.monomials_after);
            if let Some(ref mut log) = self.verbose_log {
                let _ = writeln!(
                    log,
                    r#"{{"event":"surrogate_flush","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{terms_discarded},"monomials_before":{monomials_before},"monomials_after":{monomials_after},"monomials_discarded":{monomials_discarded},"max_frequency":{mf_str},"weight_cutoff":{wc_str},"min_abs_scalar":{mas_str},"elapsed_ms":{elapsed_ms:.3e}}}"#
                );
            }
        }
    }

    /// Run surrogate propagation and return the compiled model.
    ///
    /// `evolved` is the observable map (contains the initial coefficients);
    /// `circuit` is a `SurrogatePauliCircuit` / `SurrogateMajoranaCircuit` Python object;
    /// `initial_state` is the Fock state for structural filtering;
    /// `n_params` is the total parameter count (determines lut size at evaluate time).
    pub fn run_build(
        &mut self,
        py: Python<'_>,
        evolved: &propaq_core::termsum::AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
        n_params: usize,
    ) -> PyResult<SurrogateModel<M>> {
        self.open_log()?;

        // Look-ahead frequency-pruning cap for symbolic rotations. Skipping a
        // doomed sin-branch monomial at generation time is exactly equivalent
        // to trimming it at the next flush only when *every* flush applies
        // `max_frequency` — i.e. when the schedule's `min_terms` gate is 0/None,
        // so `apply_lossy` in `apply_truncation_policy` is always true. With a
        // nonzero `min_terms`, trimming is deferred until the term count is high,
        // so eager pruning could diverge; fall back to `None` (no look-ahead)
        // there and let the flush path decide. Requires a `FrequencyTruncator` in
        // the pipeline to supply the cap.
        let prune_freq: Option<u32> = if self.schedule.min_terms.unwrap_or(0) == 0 {
            self.truncators.iter().find_map(|t| match t {
                Truncator::Frequency(f) => Some(f.max_frequency as u32),
                _ => None,
            })
        } else {
            None
        };

        // Extract circuit data from Python: each rotation is either
        // symbolic (`param_index`) or numeric (`angle`) — see `GateParam`.
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let mut circuit_data: Vec<Vec<(M, GateParam, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, GateParam, bool, Option<usize>)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    // Inject the look-ahead cap into symbolic params; numeric
                    // rotations never change frequency, so they're left as-is.
                    // `gate_idx` is a placeholder here — assigned in propagation
                    // order in the pass below.
                    let param = match SymbolicCoeff::extract_gate_param(rot)? {
                        GateParam::Symbolic { gate_idx, param, .. } => {
                            GateParam::Symbolic { gate_idx, param, prune_freq }
                        }
                        numeric => numeric,
                    };
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    Ok((generator, param, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Assign each gate its propagation-order index — the bit-pair position
        // it writes into every branching monomial's mask — and build the
        // circuit-wide `gate -> param` table `evaluate` resolves masks against.
        // Propagation applies layers in reverse, and gates within a layer in
        // reverse (see the gate loop below), so gates are numbered in exactly
        // that order: gate 0 is the first one applied. Numbering in propagation
        // order keeps a monomial's gate indices monotonic, so masks only grow
        // at the tail and early gates cost few mask words. Numeric gates get an
        // index too (they still occupy a propagation slot), but a `u32::MAX`
        // param sentinel — their positions are never written into any mask.
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let mut gate_to_param: Vec<u32> = vec![u32::MAX; total_rotations];
        let mut next_gate_idx: u32 = 0;
        for layer in circuit_data.iter_mut().rev() {
            for gate in layer.iter_mut().rev() {
                match &mut gate.1 {
                    GateParam::Symbolic { gate_idx, param, .. } => {
                        *gate_idx = next_gate_idx;
                        gate_to_param[next_gate_idx as usize] = *param;
                    }
                    GateParam::Numeric { gate_idx, .. } => {
                        *gate_idx = next_gate_idx;
                    }
                }
                next_gate_idx += 1;
            }
        }

        // Uniform noise support only (symbolic coefficients can carry damping as scalar).
        let damping = self.inner.uniform_damping(py);

        let (pbar, postfix) = self.inner.make_progress_bar(py, total_rotations)?;

        self.inner.initialize_from(evolved);

        let max_terms: Option<usize> = self.schedule.max_terms;
        let max_monomials: Option<usize> = self.schedule.max_monomials;
        // Finer, lossless merge cadence (decoupled from truncation): collapse
        // duplicate Pauli strings out of the outboxes once this many terms
        // accumulate, without truncating. Default-on via the schedule.
        let merge_max_terms: Option<usize> = self.schedule.merge_max_terms;

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let mut pending_monomials: usize = 0;

        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            // Apply uniform noise before the layer (mirrors numerical propagator order).
            if let Some(d) = damping {
                py.allow_threads(|| self.inner.apply_uniform_noise_inplace(d));
            }

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (generator, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let (added, added_monomials) = py.allow_threads(|| self.inner.apply_gate_inplace(generator, *param));
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
                    let outbox_terms = self.inner.n_outbox_terms();
                    let map_terms = self.inner.total_terms();
                    // Stale between flushes, like map_terms: cheap O(1) read, not
                    // recomputed every gate (that would require an O(total_terms) pass).
                    let monomials = self.total_monomials;
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{map_terms},"outbox_terms":{outbox_terms},"monomials":{monomials},"avg_ms_per_gate":{avg_ms_str}}}"#
                        );
                    }
                }

                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni, _)| *ni);
                let terms_trigger = max_terms.map_or(false, |max| self.inner.total_terms() + pending >= max);
                // Term count is a poor proxy for a symbolic coefficient's actual
                // size: a handful of terms can carry the overwhelming majority
                // of monomials while term count barely moves. Watch monomial
                // count directly so a flush still fires in that case.
                let monomials_trigger = max_monomials
                    .map_or(false, |max| self.total_monomials + pending_monomials >= max);
                if !next_is_intermediate && (terms_trigger || monomials_trigger) {
                    let trigger = if monomials_trigger && !terms_trigger {
                        "monomial_threshold"
                    } else {
                        "threshold"
                    };
                    py.allow_threads(|| self.flush_and_maybe_truncate(gate_idx, layer_idx, trigger));
                    pending = 0;
                    pending_monomials = 0;
                } else if !next_is_intermediate
                    && merge_max_terms.map_or(false, |m| pending >= m)
                {
                    // Finer lossless merge: transpose outboxes into the maps,
                    // collapsing duplicate Pauli strings, without the lossy
                    // truncation pass. This refreshes `total_terms` (so the
                    // truncation trigger sees the unique-term count, not the
                    // path count) and resets `pending`. `pending_monomials` is
                    // intentionally NOT reset: a merge is lossless, so those
                    // monomials stay live and uncounted in `total_monomials`
                    // (which only a truncation flush refreshes) until then.
                    let terms_before = self.inner.total_terms() + pending;
                    py.allow_threads(|| self.inner.flush_outboxes_to_maps());
                    pending = 0;
                    if self.verbose_log.is_some() {
                        let terms_after = self.inner.total_terms();
                        let qki = self
                            .current_qiskit_gate_idx
                            .map_or_else(|| "null".to_string(), |v| v.to_string());
                        if let Some(ref mut log) = self.verbose_log {
                            let _ = writeln!(
                                log,
                                r#"{{"event":"surrogate_merge","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"terms_before":{terms_before},"terms_after":{terms_after}}}"#
                            );
                        }
                    }
                }

                if let Some(ref pf) = postfix {
                    // Cached at flush time, like total_terms; cheap O(1) read here.
                    pf.bind(py).set_item("monomials", self.total_monomials)?;
                }
                AbstractPropagator::<M, SymbolicCoeff>::tick_progress_bar(
                    py, &pbar, &postfix, self.inner.total_terms(),
                )?;
                gate_idx += 1;
            }
        }

        AbstractPropagator::<M, SymbolicCoeff>::close_progress_bar(py, &pbar)?;

        py.allow_threads(|| self.flush_and_maybe_truncate(gate_idx, circuit_data.len(), "final"));

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        // Compile: collect terms with nonzero structural overlap. Drains
        // `self.inner`'s partition maps rather than cloning out of them —
        // `self.inner` is re-initialized from scratch on the next `build()`
        // call, so nothing is lost by moving instead of copying here.
        let raw: Vec<SurrogateTerm<M>> = self.inner.drain_collect_terms(|term, mut coeff| {
            let overlap = term.trace_with_fock_state(initial_state);
            if overlap.abs() > 1e-15 {
                coeff.deduplicate();
                Some(SurrogateTerm { term, overlap, coeff })
            } else {
                None
            }
        });

        Ok(SurrogateModel::new(raw, n_params, gate_to_param))
    }
}

/// Result of one `apply_truncation_policy` call, for logging/reporting.
pub struct TruncationOutcome {
    pub total_before: usize,
    pub total_after: usize,
    pub monomials_after: usize,
    pub max_frequency: Option<usize>,
    pub weight_cutoff: Option<u32>,
    pub min_abs_scalar: Option<f64>,
}

/// The distinct truncation operations resolved from a pipeline. The list is
/// collapsed into at-most-one of each kind (last occurrence wins) — the pure
/// filters commute, so order among them is immaterial, and the monomial budget
/// is always applied last regardless of position since it must rebalance the
/// post-filter state.
#[derive(Default)]
struct ResolvedOps {
    max_frequency: Option<usize>,
    weight_cutoff: Option<u32>,
    min_abs_scalar: Option<f64>,
    monomial_budget: Option<(usize, usize)>,
}

fn resolve_ops(truncators: &[Truncator]) -> ResolvedOps {
    let mut r = ResolvedOps::default();
    for t in truncators {
        match t {
            Truncator::Frequency(FrequencyTruncator { max_frequency }) => {
                r.max_frequency = Some(*max_frequency);
            }
            Truncator::Coefficient(CoefficientTruncator { min_abs_scalar }) => {
                r.min_abs_scalar = Some(*min_abs_scalar);
            }
            Truncator::Weight(WeightTruncator { weight_cutoff }) => {
                r.weight_cutoff = Some(*weight_cutoff);
            }
            Truncator::MonomialBudget(MonomialBudget { min_monomials, max_monomials }) => {
                r.monomial_budget = Some((*min_monomials, *max_monomials));
            }
        }
    }
    r
}

/// Apply `policy`'s truncation rules to `propagator`'s current live state.
/// Assumes the caller has already flushed outboxes into partition maps
/// (`AbstractPropagator::flush_outboxes_to_maps`) if that's needed — this
/// only touches what's already live in `propagator`'s maps.
///
/// Two independent stages:
///
/// 1. Dedup, plus (once term count reaches `truncation_range.0`) an
///    optional `max_frequency` trim and `weight_cutoff` term retain.
/// 2. Independently, if a `monomial_range` is configured and the live
///    monomial count still exceeds `monomial_range.1` (`max`) after stage 1,
///    remove monomials at the single highest frequency currently present
///    — never anything lower. The *target* is `max`, not `monomial_range.0`
///    (`min`): the top-frequency bucket is removed in full only if doing so
///    doesn't remove more than needed to reach `max`; otherwise only enough
///    of it is removed (an arbitrary subset of the tied top frequency,
///    since frequency alone — not scalar magnitude — is this crate's
///    existing pruning signal; see `trim_high_frequency`) to land exactly
///    at `max`. `min` is not a target here — see `monomial_removal_budget`
///    — it's a floor that only matters on a misconfigured policy
///    (`min > max`). If the whole top-frequency bucket is itself smaller
///    than what's needed to reach `max`, it's still removed in full
///    (falling short of `max`, left to a subsequent flush — see below). Not
///    gated behind stage 1's term-count floor: monomial count can explode
///    with comparatively few live terms, so it needs its own trigger.
///
///    This deliberately only ever erodes one frequency level per call
///    rather than searching for a cutoff that reaches the floor in one
///    shot: a fast-growing run may need several consecutive flushes to
///    fully erode a deep distribution, each cheap and predictable, rather
///    than one large adaptive cut that might remove a lot of only
///    moderately-high-frequency data along with the truly extreme end.
///
/// Extracted as a standalone function (rather than inlined in
/// `SurrogatePropagator::flush_and_maybe_truncate`) so tooling that drives
/// an `AbstractPropagator` directly — without going through the
/// PyO3-circuit-driven `build()` entrypoint, e.g. `bin/cluster_bench` —
/// replicates the exact same flush behavior instead of a hand-rolled copy
/// that could drift out of sync with it.
///
/// Maximum number of monomials the monomial-range stage may remove from the
/// current top-frequency bucket this call, given `monomials_after > max`.
///
/// Targets landing at `max` (`monomials_after - max`), not `min`: a
/// top-frequency bucket bigger than that amount only ever gets a partial,
/// budgeted removal (see the caller's `budget >= n_top` branch), never
/// discarded in full, so truncation stops at `max` instead of continuing to
/// erode down toward `min`. `min` only becomes the binding constraint (via
/// `want.min(floor)` below) on a misconfigured policy with `min > max`;
/// under a sane one (`max >= min`), `monomials_after - max` is always `<=
/// monomials_after - min`, so this always reduces to `monomials_after -
/// max` and `min` has no effect. Kept as an explicit floor regardless,
/// rather than relying on callers never passing `min > max`.
#[inline]
fn monomial_removal_budget(monomials_after: usize, min: usize, max: usize) -> usize {
    let want = monomials_after.saturating_sub(max);
    let floor = monomials_after.saturating_sub(min);
    want.min(floor)
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

/// Run the truncation pipeline against `propagator`'s current live state (the
/// caller must have flushed outboxes into maps first). Order of operations:
///
/// 1. A single fused per-coefficient pass: optional `max_frequency` trim, the
///    always-on lossless dedup (merge identical monomials, drop exact zeros),
///    then the optional `min_abs_scalar` coefficient trim (run *after* dedup so
///    it sees merged scalars); plus a per-term `weight_cutoff` retain. These
///    lossy operators are gated by the schedule's `min_terms` — below it, only
///    dedup runs.
/// 2. If a `MonomialBudget` operator is present and the live monomial count
///    still exceeds its `max`, an importance-ranked removal keyed by
///    `(frequency desc, |scalar| asc)` down to `max`. Not gated by `min_terms`:
///    a monomial explosion with few terms still needs cutting.
///
/// Standalone (not inlined into `flush_and_maybe_truncate`) so tooling that
/// drives an `AbstractPropagator` directly (e.g. `bin/cluster_bench`) reuses the
/// exact same flush behavior instead of a hand-rolled copy that could drift.
pub fn apply_truncation_policy<M: AbstractTerm>(
    propagator: &mut AbstractPropagator<M, SymbolicCoeff>,
    schedule: &FlushSchedule,
    truncators: &[Truncator],
) -> TruncationOutcome {
    let total_before = propagator.total_terms();
    let ops = resolve_ops(truncators);
    let min_terms = schedule.min_terms.unwrap_or(0);

    // Deferred like the numerical propagator's TruncationPolicy: below
    // min_terms, skip the lossy filters and only run the lossless dedup.
    let apply_lossy = total_before >= min_terms;

    let mut monomials_after = propagator.map_and_retain_coeffs_inplace(
        |_, c: &mut SymbolicCoeff| {
            if apply_lossy {
                if let Some(mf) = ops.max_frequency {
                    c.trim_high_frequency(mf);
                }
            }
            c.deduplicate();
            if apply_lossy {
                if let Some(mas) = ops.min_abs_scalar {
                    c.trim_small_scalars(mas);
                }
            }
        },
        |t: &M, c: &SymbolicCoeff| {
            let weight_ok = !apply_lossy || ops.weight_cutoff.map_or(true, |w| t.weight() <= w);
            weight_ok && !c.is_empty()
        },
    );

    if let Some((min, max)) = ops.monomial_budget {
        if monomials_after > max {
            let budget = monomial_removal_budget(monomials_after, min, max);
            if budget > 0 {
                // 1. Global frequency histogram in one parallel fold. Frequency
                //    is a small bounded int, so a per-worker `Vec<u64>` keyed by
                //    frequency is cheap and exact; the per-coefficient scan is
                //    serial per coeff but the fold spreads coeffs across workers.
                let hist: Vec<u64> = propagator.fold_coeffs(
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
                    let mut scalars: Vec<f64> = propagator.fold_coeffs(
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

                // 4. One importance-ranked removal pass. Per-entry parallelism in
                //    `map_and_retain_coeffs_inplace` keeps a giant coefficient
                //    from stalling its partition.
                let tie_remaining = std::sync::atomic::AtomicUsize::new(tie_budget);
                monomials_after = propagator.map_and_retain_coeffs_inplace(
                    |_, c: &mut SymbolicCoeff| {
                        c.remove_by_rank_budgeted(f_star, s_star, &tie_remaining);
                    },
                    |_, c: &SymbolicCoeff| !c.is_empty(),
                );
            }
        }
    }

    let total_after = propagator.total_terms();
    TruncationOutcome {
        total_before,
        total_after,
        monomials_after,
        max_frequency: ops.max_frequency,
        weight_cutoff: ops.weight_cutoff,
        min_abs_scalar: ops.min_abs_scalar,
    }
}

use propaq_pauli::string::PauliString;
use propaq_pauli::termsum::PauliTermSum;
use propaq_majorana::monomial::MajoranaMonomial;
use propaq_majorana::termsum::MajoranaTermSum;

/// Back-propagates Pauli observables symbolically, producing a compiled model
/// that can be re-evaluated for any parameter assignment.
///
/// Arguments:
///     truncation: The truncation pipeline — a list of truncator objects
///         (FrequencyTruncator, CoefficientTruncator, WeightTruncator,
///         MonomialBudget) applied at each flush, a single such truncator, a
///         legacy FrequencyTruncationPolicy (decomposed automatically), or None.
///     schedule: Optional FlushSchedule controlling flush/merge cadence. Omitted
///         → sensible defaults when any truncator is given, or "flush only at the
///         end" when truncation is also None. A legacy policy supplies its own
///         schedule unless one is passed explicitly here.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliString>,
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
        let evolved = observable.inner.copy();
        let model = self.inner.run_build(py, &evolved, circuit, initial_state, n_params)?;
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
    inner: SurrogatePropagator<MajoranaMonomial>,
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
        let evolved = observable.inner.copy();
        let model = self.inner.run_build(py, &evolved, circuit, initial_state, n_params)?;
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
    fn targets_max_when_bucket_would_be_more_than_enough() {
        // A bucket of any size > 10 should only ever be allowed to remove
        // 10 (landing exactly at max=90), never eroded further toward
        // min=50 just because a bigger bucket happens to be available.
        assert_eq!(monomial_removal_budget(100, 50, 90), 10);
    }

    #[test]
    fn sane_policy_ignores_min_entirely() {
        // want = after - max is always <= floor = after - min when
        // max >= min, so min should never change the result.
        for (after, min, max) in [(100, 50, 90), (100, 0, 99), (1_000_000, 1, 999_999), (11, 10, 10)] {
            assert_eq!(monomial_removal_budget(after, min, max), after - max);
        }
    }

    #[test]
    fn misconfigured_min_greater_than_max_clamps_to_the_min_floor() {
        // min > max: the floor (after - min) is tighter than the max-based
        // want (after - max), so it's the binding constraint.
        assert_eq!(monomial_removal_budget(100, 90, 50), 10);
    }

    #[test]
    fn exactly_one_over_max_wants_a_budget_of_one() {
        assert_eq!(monomial_removal_budget(91, 50, 90), 1);
    }

    #[test]
    fn zero_min_and_max_equal_to_after_minus_one() {
        assert_eq!(monomial_removal_budget(1000, 0, 999), 1);
    }
}
