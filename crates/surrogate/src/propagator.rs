///
/// impl for the surrogate/symbolic propagator!
///
/// This propagator runs on the same columnar SoA engine as the numerical
/// propagators (`propaq_core::soa`): every gate application is a
/// `soa::kernels::apply_rotation` call over a `SoaTermSum<SymbolicCoeff>`;
/// merging/truncation reuse `soa::kernels::merge`/`map_retain` the same way
/// the numerical propagators do.
///
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;
use rayon::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::coeff::CoeffRepr;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::logger::Logger;
use propaq_core::propagator::{close_progress_bar, make_progress_bar, tick_progress_bar};
use propaq_core::soa::kernels;
use propaq_core::soa::propagator::CLIFFORD_COS_EPS;
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::{simplify_sharded, CompiledCoeff, GateParam, SymbolicCoeff};
use crate::truncation::FrequencyTruncationPolicy;
use crate::model::{SurrogateModel, SurrogateTerm, PauliSurrogateModel, MajoranaSurrogateModel};
use propaq_core::truncators::{
    reject_numerical_only, resolve_config, resolve_truncation as core_resolve_truncation, FlushSchedule,
    ResolvedConfig, Truncator,
};

/// Shard count (relative to thread count) for both `compile_surviving_terms`
/// and `apply_truncation_policy`'s `Simplify` pass -- shared by both since
/// they're the same tradeoff: a subtree shared *across* shard boundaries is
/// processed once per shard instead of once globally, bounding duplication
/// at `(thread count) x (oversubscription factor)` instead of unboundedly.
/// See `compile_surviving_terms`'s doc comment for the full rationale
/// (oversubscription specifically, not exactly one shard per thread).
const SHARD_OVERSUBSCRIPTION: usize = 4;

/// Resolve the flexible `truncation` constructor argument into `(FlushSchedule,
/// [Truncator])`. The surrogate additionally accepts the legacy
/// `FrequencyTruncationPolicy` (decomposed here); everything else, such as a list, a
/// single truncator, a core `TruncationPolicy`, or `None`, is delegated to the
/// shared `propaq_core` resolver. Every truncator the resolved list can
/// produce is honored by `apply_truncation_policy` below -- no rejection step.
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
    /// term-count read via `SoaTermSum::len`). `u128`, not `usize`: an
    /// individual coefficient's monomial count can legitimately saturate a
    /// `u64` on a real workload (see `Node::add`'s doc comment).
    total_monomials: u128,
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
        pending_monomials: u128,
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
        // Cheap running estimate (not the exact recount above), for
        // `MonomialBudget`'s `min_monomials` floor -- deliberately a
        // different name/value from `monomials_before`, which is an exact,
        // verbose-logging-only recompute. `saturating_add`: both operands can
        // legitimately be near `u128::MAX` once truly huge (see
        // `Node::add`'s doc comment) -- plain `+` wrapping here would let a
        // `MonomialBudget` floor/ceiling silently stop comparing correctly.
        let monomials_before_estimate = self.total_monomials.saturating_add(pending_monomials);

        let cfg = resolve_config(&self.truncators);
        let outcome =
            pool.install(|| apply_truncation_policy::<B>(evolved, &cfg, monomials_before_estimate));
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
            // `saturating_sub`: truncation only ever removes monomials, so
            // `monomials_after <= monomials_before` in exact arithmetic, but
            // defensive here too now that both sides can independently
            // saturate at `u128::MAX` -- a plain `-` would underflow-panic
            // (debug) / wrap to a huge bogus value (release) in any edge case
            // this reasoning missed.
            let monomials_discarded = monomials_before.saturating_sub(outcome.monomials_after);
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
        initial_state: &[u64],
        n_params: usize,
    ) -> PyResult<SurrogateModel> {
        self.open_log()?;

        // Resolve the truncation pipeline once (Copy config). The flush
        // triggers (`max_terms`/`max_monomials`) and the `min_terms`/
        // `min_monomials` floors come from `TermBudget`/`MonomialBudget`;
        // the merge cadence from the schedule.
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

        // Flush trigger from the budget(s); merge cadence from the schedule.
        let max_terms: Option<usize> = cfg.max_terms;
        let max_monomials: Option<u128> = cfg.max_monomials;
        let merge_max_terms: Option<usize> = self.schedule.merge_max_terms;

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let mut pending_monomials: u128 = 0;
        let mut deferred_threshold_trigger: Option<&'static str> = None;
        let mut merged_len = evolved.len();

        let pool = Arc::clone(&self.pool);
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (gen0, gen1, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let gen = [gen0.as_slice(), gen1.as_slice()];
                let before = evolved.len();
                // Same Clifford in-place fast path the numerical `SoaPropagator`
                // already takes for a `pi/2`-angle gate: without this, every such
                // gate (numeric basis-change/Clifford rotations included) appends
                // a new row instead of overwriting in place, needlessly doubling
                // the affected rows' monomial `count` once a later merge
                // recombines them -- see `SymbolicCoeff::is_clifford_param`.
                let clifford_inplace = SymbolicCoeff::is_clifford_param(param, CLIFFORD_COS_EPS);
                let added = py.allow_threads(|| {
                    pool.install(|| {
                        kernels::apply_rotation::<B, SymbolicCoeff>(evolved, gen, param, clifford_inplace)
                    })
                });
                // `apply_rotation` returns the touched-row count in both
                // branches, but only the append branch actually grows
                // `evolved` -- the in-place branch overwrites `added` rows
                // scattered across the *existing* `[0, n)` range instead of
                // appending a contiguous run at `[before, before+added)`.
                // Reading that slice unconditionally would (once
                // `clifford_inplace` can be `true` at all) read stale/unused
                // buffer contents past the live region rather than the rows
                // this gate actually touched. It's also unnecessary: the
                // in-place branch only ever wraps a coefficient in a `Scale`
                // node (a numeric Clifford angle is never `GateParam::Symbolic`),
                // and `Node::scale`'s `count` is the inner node's `count`
                // unchanged, so an in-place gate never changes monomial count
                // -- there is no delta to add.
                if !clifford_inplace {
                    // `saturating_add` throughout: `size_hint()` values (and
                    // their running accumulation into `pending_monomials`) can
                    // legitimately be near `u128::MAX` on a real workload -- see
                    // `Node::add`'s doc comment for why plain `+`/`.sum()` here
                    // would silently wrap instead.
                    let added_monomials: u128 = evolved.coeffs[before..before + added]
                        .iter()
                        .fold(0u128, |acc, c| acc.saturating_add(c.size_hint()));
                    pending += added;
                    pending_monomials = pending_monomials.saturating_add(added_monomials);
                }

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
                    let monomials = self.total_monomials.saturating_add(pending_monomials);
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
                let monomials_trigger = max_monomials
                    .map_or(false, |max| self.total_monomials.saturating_add(pending_monomials) >= max);
                // If both ceilings cross on the same gate, `terms_trigger` wins
                // in the logged/latched trigger name (harmless -- a flush still
                // fires either way, this only affects which string gets logged).
                let threshold_trigger: Option<&'static str> = if terms_trigger {
                    Some("threshold")
                } else if monomials_trigger {
                    Some("monomial_threshold")
                } else {
                    None
                };
                let pending_trigger = deferred_threshold_trigger.or(threshold_trigger);
                if let Some(trigger) = pending_trigger {
                    if !next_is_intermediate {
                        py.allow_threads(|| {
                            self.flush_and_maybe_truncate(evolved, gate_idx, layer_idx, trigger, pending_monomials)
                        });
                        pending = 0;
                        pending_monomials = 0;
                        deferred_threshold_trigger = None;
                        merged_len = evolved.len();
                    } else if threshold_trigger.is_some() && deferred_threshold_trigger.is_none() {
                        deferred_threshold_trigger = Some(trigger);
                        if self.verbose_log.is_some() {
                            let live_terms = evolved.len();
                            let live_monomials = self.total_monomials.saturating_add(pending_monomials);
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
                    pf.bind(py)
                        .set_item("monomials", self.total_monomials.saturating_add(pending_monomials))?;
                }
                tick_progress_bar(py, &pbar, &postfix, evolved.len())?;
                gate_idx += 1;
            }
        }

        close_progress_bar(py, &pbar)?;

        py.allow_threads(|| {
            self.flush_and_maybe_truncate(evolved, gate_idx, circuit_data.len(), "final", pending_monomials)
        });

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        // Compile: collect terms with nonzero structural overlap and compile
        // survivors into one shared tape, sharded across `pool`'s workers --
        // see `compile_surviving_terms`'s doc comment for the full
        // rationale (this used to be a `compile()` call per term, which
        // caused a reported multi-hundred-GB OOM under heavy parameter
        // reuse).
        let (tape, raw) = py.allow_threads(|| {
            pool.install(|| compile_surviving_terms::<B>(evolved, n_units, stride, initial_state))
        });

        Ok(SurrogateModel::new(raw, tape, n_params))
    }
}

/// Compile all of `evolved`'s surviving (nonzero structural overlap) terms
/// into ONE shared tape, sharded across `rayon::current_num_threads()`
/// contiguous chunks (call from within the pool whose thread count should
/// determine the shard count, e.g. inside `pool.install`). Each shard's
/// coefficients are compiled together via `SymbolicCoeff::compile_batch`,
/// sharing one memo/tape per shard rather than one `compile()` call per
/// term: compiling term-by-term re-walks and re-emits any DAG subtree
/// shared across many surviving terms once per referencing term, which
/// under heavy parameter reuse approaches (surviving terms) x (distinct
/// nodes) total compiled output -- the diagnosed cause of the OOM this
/// replaced, and a source of redundant work in every subsequent
/// `evaluate` too. Sharding bounds duplication of any shared node at
/// (shard count) instead of (terms referencing it), independent of term
/// count -- see `propaq.MD`'s "Evaluate & persistence" section.
/// `CompiledCoeff::merge_shards` does one cheap, purely-arithmetic pass
/// concatenating the shards' tapes into the single tape returned here,
/// alongside a flat `Vec<SurrogateTerm>` (`(overlap, root)` per surviving
/// term -- the term itself is never reconstructed, only ever needed to
/// compute `overlap`).
///
/// Deliberately free of `pyo3`/`py` -- this is the whole reason it's split
/// out of `run_build`, so it can be exercised directly by a Rust unit test
/// against a hand-built `SoaTermSum<SymbolicCoeff>` without fabricating a
/// Python circuit object.
///
/// Shard count is **oversubscribed** relative to thread count
/// (`SHARD_OVERSUBSCRIPTION` shards per worker, not exactly one) so rayon's
/// work-stealing scheduler actually has something to rebalance: with exactly
/// one shard per thread, once every thread has claimed its one chunk there
/// is nothing left in the queue for an idle thread to steal, so a single
/// unusually heavy shard (a few terms with much deeper coefficient DAGs than
/// the rest -- plausible under skewed parameter reuse) stalls the whole pass
/// on that one worker while every other thread sits idle. Leaving several
/// unclaimed shards in the queue at all times lets a thread that finishes
/// early immediately pick up the next one instead of idling. This raises the
/// shared-node duplication bound from `(thread count)` to `(thread count) x
/// (oversubscription factor)` -- still a small constant independent of term
/// count, so the OOM-prevention property this function exists for is
/// unaffected, just with a slightly larger (and tunable) constant.
fn compile_surviving_terms<B: SoaBasis>(
    evolved: &mut SoaTermSum<SymbolicCoeff>,
    n_units: usize,
    stride: usize,
    initial_state: &[u64],
) -> (CompiledCoeff, Vec<SurrogateTerm>) {
    let n = evolved.len();
    let target_shards = (rayon::current_num_threads() * SHARD_OVERSUBSCRIPTION).max(1);
    let chunk = n.div_ceil(target_shards).max(1);

    let (shard_tapes, shard_terms): (Vec<CompiledCoeff>, Vec<Vec<(f64, usize)>>) = {
        let SoaTermSum { planes, coeffs, .. } = evolved;
        coeffs[..n]
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(chunk_idx, shard)| {
                // `par_chunks_mut` doesn't hand us the chunk's own absolute
                // row offset -- `chunk_idx * chunk` is it, since chunking
                // always partitions front-to-back in fixed-size runs (the
                // same assumption `SurrogateModel::save`'s own `par_chunks`
                // sharding already relies on).
                let base = chunk_idx * chunk;
                let mut overlaps: Vec<f64> = Vec::new();
                let mut survivors: Vec<SymbolicCoeff> = Vec::new();
                for (j, c) in shard.iter_mut().enumerate() {
                    let global_row = base + j;
                    let s = global_row * stride;
                    let term = [&planes[0][s..s + stride], &planes[1][s..s + stride]];
                    let overlap = B::trace(term, n_units, initial_state);
                    // Always take, even when the row won't survive: a
                    // structurally-zero-overlap term contributes nothing to
                    // any expectation value regardless of parameters, so its
                    // entire coefficient DAG is pure waste once we know
                    // that -- leaving it in `evolved` (untouched) would keep
                    // it resident until the whole `build()` call unwinds,
                    // well past when the compiled tape is already built.
                    // Taking it here drops it (or, if some part is still
                    // shared with a surviving term's own coefficient via
                    // `Arc`, correctly keeps just that shared part alive)
                    // immediately instead.
                    let coeff = std::mem::take(c);
                    if overlap.abs() > 1e-15 {
                        overlaps.push(overlap);
                        survivors.push(coeff);
                    }
                }
                let (tape, local_roots) = SymbolicCoeff::compile_batch(survivors);
                let terms: Vec<(f64, usize)> = overlaps.into_iter().zip(local_roots).collect();
                (tape, terms)
            })
            .unzip()
    };

    let (tape, offsets) = CompiledCoeff::merge_shards(shard_tapes);
    let mut raw = Vec::with_capacity(shard_terms.iter().map(|s| s.len()).sum());
    for (terms, offset) in shard_terms.into_iter().zip(offsets) {
        for (overlap, local_root) in terms {
            let root = if local_root == usize::MAX { usize::MAX } else { local_root + offset };
            raw.push(SurrogateTerm { overlap, root });
        }
    }
    (tape, raw)
}

/// Result of one `apply_truncation_policy` call, for logging/reporting.
pub struct TruncationOutcome {
    pub total_before: usize,
    pub total_after: usize,
    pub monomials_after: u128,
    pub frequency: Option<usize>,
    pub weight: Option<u32>,
    pub coefficient: Option<f64>,
}

/// Run the truncation pipeline against `evolved`'s current live state.
///
/// `WeightTruncator` (operator weight) is a stream-compaction filter, same
/// as the numerical propagator. `FrequencyTruncator`/`CoefficientTruncator`
/// go through `SymbolicCoeff::prune`, which drops monomials structurally
/// (no expansion) -- see `prune`'s doc comment. `TermBudget`/`MonomialBudget`
/// only decide *when* this function gets called (see `run_build`'s trigger
/// logic) and additionally gate `apply_lossy` below via their `min_terms`/
/// `min_monomials` floors. The always-on lossless dedup
/// (`SymbolicCoeff::add_assign`/`is_empty` via `merge`) has already run by
/// the time this is called.
///
/// `Simplify`, when present in the pipeline, runs *before* everything else
/// here (including the `apply_lossy` gate below -- it's lossless, unlike
/// `prune`/`WeightTruncator`, so there's no correctness reason to defer it
/// during warmup) and *before* `prune` specifically: `prune`'s
/// `coeff_cutoff` decision uses `upper_scale`, a safe bound computed
/// independently per derivation path, so two still-unmerged paths to the
/// same canonical monomial are each judged by their own (possibly small)
/// bound -- pruning them independently first can permanently discard a
/// monomial whose *combined* magnitude would have cleared the cutoff.
/// Running `Simplify` first means `prune` always sees the true, merged
/// magnitude, sharpening (never loosening) `CoefficientTruncator`'s
/// accuracy. This is a real, user-visible behavior change versus running
/// `CoefficientTruncator` without `Simplify`: the same configured cutoff can
/// retain a different (more accurate) survivor set. `FrequencyTruncator` is
/// unaffected -- merging same-canonical-monomial siblings never changes
/// frequency.
///
/// `monomials_before_estimate` is the caller's cheap running estimate
/// (`self.total_monomials.saturating_add(pending_monomials)`), not an exact
/// recount -- there's no O(1) exact way to get one, unlike `total_before`
/// (`evolved.len()`).
pub fn apply_truncation_policy<B: SoaBasis>(
    evolved: &mut SoaTermSum<SymbolicCoeff>,
    cfg: &ResolvedConfig,
    monomials_before_estimate: u128,
) -> TruncationOutcome {
    let total_before = evolved.len();
    let min_terms = cfg.min_terms.unwrap_or(0);
    let min_monomials = cfg.min_monomials.unwrap_or(0);
    let n_units = evolved.n_units;

    if cfg.simplify {
        let n_shards = (rayon::current_num_threads() * SHARD_OVERSUBSCRIPTION).max(1);
        simplify_sharded(&mut evolved.coeffs[..total_before], n_shards);
    }

    // Deferred like the numerical propagator: below min_terms/min_monomials,
    // skip the lossy filters. Both floors must clear (AND, not OR) -- each
    // is an independent "don't lossy-prune something too young/small yet"
    // veto, and `min_terms` already gates the whole bundle (including
    // `WeightTruncator`, which isn't really "about" term count either), so
    // there's precedent that a floor means "wait for everything to warm up,"
    // not a per-operator-scoped gate.
    let apply_lossy = total_before >= min_terms && monomials_before_estimate >= min_monomials;
    // Saturating cast: a `usize` cap beyond `u32::MAX` is indistinguishable
    // from "no cap" in practice, but should clamp rather than wrap.
    let max_frequency = cfg.frequency.map(|f| f.min(u32::MAX as usize) as u32);

    let monomials_after = kernels::map_retain::<B, SymbolicCoeff, _, _>(
        evolved,
        |c: &mut SymbolicCoeff| {
            if apply_lossy {
                c.prune(max_frequency, cfg.coefficient);
            }
        },
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
        reject_numerical_only(&truncators)?;
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
    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PauliSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let mut evolved = observable.inner.map_coeffs(|c| SymbolicCoeff::from_real(*c));
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.inner.n_units)?,
            None => Bitset::zero(),
        };
        let model = self.inner.run_build(py, &mut evolved, circuit, initial_state.as_words(), n_params)?;
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
        reject_numerical_only(&truncators)?;
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
        reject_numerical_only(&truncators)?;
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<MajoranaSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let mut evolved = observable.inner.map_coeffs(|c| SymbolicCoeff::from_real(*c));
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.inner.n_units)?,
            None => Bitset::zero(),
        };
        let model = self.inner.run_build(py, &mut evolved, circuit, initial_state.as_words(), n_params)?;
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
        reject_numerical_only(&truncators)?;
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

#[cfg(test)]
mod sharded_compile_tests {
    use super::*;
    use crate::model::SurrogateModel;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// `compile_surviving_terms` (the sharded-batch replacement for a
    /// per-term `compile()` loop) exercised end-to-end against a circuit
    /// with heavy cross-term sharing (many shared parameters, many rounds --
    /// same shape as `shared_parameter_history_matches_f64_engine_under_fixed_angles`),
    /// asserting the resulting `SurrogateModel::evaluate` output matches the
    /// trusted plain-`f64` numeric engine's own expectation value. This is
    /// the multi-shard path's end-to-end regression guard: the unit tests in
    /// `symcoeff.rs` cover `compile_batch`/`merge_shards` in isolation, this
    /// covers the shard/offset bookkeeping that glues them together in
    /// `run_build`.
    #[test]
    fn compile_surviving_terms_matches_f64_engine_under_heavy_parameter_sharing() {
        const N_QUBITS: usize = 8;
        const N_PARAMS: usize = 3;
        const FOCK: u64 = 0;
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
        assert!(gate_idx as usize > 4 * N_PARAMS, "test should reuse each parameter many times");

        let expected: f64 = (0..numeric.len())
            .map(|i| *numeric.coeff(i) * PauliBasis::trace(numeric.term_planes(i), numeric.n_units, &[FOCK]))
            .sum();

        let n_units = symbolic.n_units;
        let stride = symbolic.stride;
        let (tape, terms) = compile_surviving_terms::<PauliBasis>(&mut symbolic, n_units, stride, &[FOCK]);
        assert!(!terms.is_empty(), "test did not exercise any surviving terms");
        let model = SurrogateModel::new(terms, tape, N_PARAMS);

        let got = model.evaluate(&fixed_angles);
        assert!(
            (got - expected).abs() < 1e-8 * expected.abs().max(1.0),
            "sharded-batch-compiled model {got} vs f64 reference {expected}",
        );
    }

    /// Regression test for a real memory-retention gap: `compile_surviving_terms`
    /// used to only `mem::take` a row's coefficient inside the
    /// `overlap.abs() > 1e-15` branch, so a structurally-zero-overlap row's
    /// entire coefficient DAG (pure waste -- it contributes nothing to any
    /// expectation value, for any parameters) stayed fully resident in
    /// `evolved` until the caller's `SoaTermSum` eventually dropped, well
    /// after the compiled tape was already built. Every row's coefficient
    /// must now be taken (freeing it immediately unless still shared with a
    /// surviving term's own coefficient via `Arc`), not just survivors'.
    #[test]
    fn zero_overlap_rows_have_their_coefficient_cleared_too() {
        const N_QUBITS: usize = 4;
        const FOCK: u64 = 0;
        let mut ts: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);

        // Row 0: pure-Z string -- structurally survives (nonzero trace
        // against the all-zero Fock state).
        let (gx0, gz0) = planes_of(0, 0b1, 1);
        ts.push([&gx0, &gz0], SymbolicCoeff::from_real(1.0));

        // Row 1: has X support -- structurally zero overlap regardless of
        // its coefficient's value or any parameter assignment.
        let (gx1, gz1) = planes_of(0b1, 0, 1);
        ts.push([&gx1, &gz1], SymbolicCoeff::from_real(3.0));

        let (_tape, terms) = compile_surviving_terms::<PauliBasis>(&mut ts, N_QUBITS, 1, &[FOCK]);
        assert_eq!(terms.len(), 1, "only the pure-Z row should survive");

        assert!(ts.coeff(0).is_empty(), "surviving row's coefficient should have been taken");
        assert!(
            ts.coeff(1).is_empty(),
            "zero-overlap row's coefficient should ALSO have been taken/dropped, not left resident",
        );
    }
}

#[cfg(test)]
mod monomial_budget_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// `apply_truncation_policy`'s `apply_lossy` gate must AND `min_terms`
    /// and `min_monomials` together -- both floors must clear before any
    /// lossy operator (here, `CoefficientTruncator`) runs, not either one
    /// alone. Uses a `CoefficientTruncator` cutoff that would prune the sole
    /// live term's coefficient down to nothing (removing the row entirely
    /// via `map_retain`'s `!c.is_empty()` check) as the observable signal
    /// for whether pruning actually ran.
    #[test]
    fn min_terms_and_min_monomials_floors_are_anded_together() {
        const N_QUBITS: usize = 4;

        let build_one_term = || {
            let mut ts: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);
            let (gx, gz) = planes_of(0, 0b1, 1);
            ts.push([&gx, &gz], SymbolicCoeff::from_real(0.001));
            ts
        };
        let cutoff_cfg = |min_terms: Option<usize>, min_monomials: Option<u128>| ResolvedConfig {
            coefficient: Some(1.0), // upper_scale 0.001 << 1.0: prunes to nothing if lossy runs
            min_terms,
            min_monomials,
            ..Default::default()
        };

        // min_terms not met (1 < 5): apply_lossy must be false regardless of
        // min_monomials, so the term survives untouched.
        let mut ts = build_one_term();
        apply_truncation_policy::<PauliBasis>(&mut ts, &cutoff_cfg(Some(5), None), 1);
        assert_eq!(ts.len(), 1, "min_terms floor should have suppressed pruning");

        // min_terms met (1 >= 1) but min_monomials not met (1 < 5): still no
        // pruning -- both floors must clear, not just one.
        let mut ts = build_one_term();
        apply_truncation_policy::<PauliBasis>(&mut ts, &cutoff_cfg(Some(1), Some(5)), 1);
        assert_eq!(ts.len(), 1, "min_monomials floor should have suppressed pruning even though min_terms cleared");

        // Both floors clear: lossy pruning proceeds, the sole term's
        // coefficient is pruned to nothing and the row is dropped.
        let mut ts = build_one_term();
        apply_truncation_policy::<PauliBasis>(&mut ts, &cutoff_cfg(Some(1), Some(1)), 1);
        assert_eq!(ts.len(), 0, "with both floors cleared, the coefficient truncator should have pruned the row away");
    }
}

#[cfg(test)]
mod simplify_truncator_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// Regression guard for `apply_truncation_policy`'s `Simplify`-before-
    /// `prune` ordering (see that function's doc comment): two derivation
    /// paths to the *same* canonical monomial (here, the trivial empty-run
    /// "constant" monomial), each individually below a `CoefficientTruncator`
    /// cutoff, but whose combined magnitude clears it. `prune`'s
    /// `coeff_cutoff` check is per-derivation-path (`upper_scale`), so
    /// without merging first it "salami-slices" the true contribution away;
    /// `Simplify` merges the two 0.3-scalar paths into one 0.6-scalar
    /// monomial before `prune` ever runs, so the true magnitude is what
    /// gets checked against the cutoff.
    #[test]
    fn simplify_runs_before_prune_and_sharpens_coefficient_cutoff() {
        const N_QUBITS: usize = 4;

        let build_two_paths_to_same_monomial = || {
            let mut ts: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);
            let (gx, gz) = planes_of(0, 0b1, 1);
            let mut coeff = SymbolicCoeff::from_real(0.3);
            coeff.add_assign(SymbolicCoeff::from_real(0.3));
            ts.push([&gx, &gz], coeff);
            ts
        };

        // 0.3 < 0.5 (each path, individually) < 0.6 (the true combined sum).
        let cutoff_cfg = |simplify: bool| ResolvedConfig {
            coefficient: Some(0.5),
            simplify,
            ..Default::default()
        };

        let mut ts = build_two_paths_to_same_monomial();
        apply_truncation_policy::<PauliBasis>(&mut ts, &cutoff_cfg(false), 1);
        assert_eq!(
            ts.len(), 0,
            "without Simplify, two salami-sliced paths (0.3 each) are incorrectly pruned \
             even though their true combined magnitude (0.6) clears the 0.5 cutoff",
        );

        let mut ts = build_two_paths_to_same_monomial();
        apply_truncation_policy::<PauliBasis>(&mut ts, &cutoff_cfg(true), 1);
        assert_eq!(
            ts.len(), 1,
            "with Simplify, the merged 0.6-scalar monomial's true magnitude clears the cutoff and survives",
        );
    }

    /// `simplify` must default to `false` (`ResolvedConfig::default()`) and
    /// change nothing when unset -- guards against `Simplify` being
    /// accidentally wired in as always-on. The full existing test suite
    /// (none of which sets `cfg.simplify`) passing unchanged is the primary
    /// evidence for this; this test additionally pins the exact default
    /// behavior on the same salami-slicing shape used above, so a future
    /// change to the default is caught here specifically, not just
    /// incidentally elsewhere.
    #[test]
    fn simplify_disabled_by_default_leaves_existing_pruning_behavior_unchanged() {
        const N_QUBITS: usize = 4;

        let mut ts: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);
        let (gx, gz) = planes_of(0, 0b1, 1);
        let mut coeff = SymbolicCoeff::from_real(0.3);
        coeff.add_assign(SymbolicCoeff::from_real(0.3));
        ts.push([&gx, &gz], coeff);

        let cfg = ResolvedConfig { coefficient: Some(0.5), ..Default::default() };
        assert!(!cfg.simplify, "ResolvedConfig::default() must leave simplify off");

        apply_truncation_policy::<PauliBasis>(&mut ts, &cfg, 1);
        assert_eq!(ts.len(), 0, "default (Simplify-absent) behavior must match the pre-Simplify pruning outcome");
    }
}

#[cfg(test)]
mod clifford_inplace_regression_tests {
    use super::*;
    use propaq_core::soa::SoaTermSum;

    fn planes_of(x: u64, z: u64, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut gx = vec![0u64; stride];
        let mut gz = vec![0u64; stride];
        gx[0] = x;
        gz[0] = z;
        (gx, gz)
    }

    /// Regression test for the bug this file's `run_build` used to have:
    /// every numeric gate (including a real circuit's Clifford, `pi/2`-angle
    /// basis-change/routing rotations -- e.g. a transpiled `swap`) was applied
    /// with `clifford_inplace` hardcoded to `false`, so an anticommuting
    /// Clifford gate always appended a new row instead of overwriting in
    /// place. A single anticommuting term run through `N` such gates (each
    /// followed by a `merge`, mirroring `run_build`'s periodic merge cadence)
    /// then has its monomial `count` double per gate even though only one
    /// live term (and one *real* trig factor) is ever present -- exactly the
    /// "few parameters, astronomically large monomial count" shape reported
    /// against a real workload (see `CHANGELOG.md`: "3969 terms, ~15
    /// quintillion monomials"). Computing `clifford_inplace` via
    /// `SymbolicCoeff::is_clifford_param` (this fix) keeps both the term
    /// count and the monomial count flat at 1 throughout, since a `pi/2`
    /// rotation's cos branch is genuinely negligible and the kernel can
    /// overwrite in place instead of branching at all.
    #[test]
    fn clifford_angle_numeric_gates_no_longer_inflate_monomial_count() {
        const N_QUBITS: usize = 4;
        const ROUNDS: usize = 40;
        let angle = std::f64::consts::FRAC_PI_2;

        let mut evolved: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);
        let (gx, gz) = planes_of(0, 0b1, 1); // Z on qubit 0
        evolved.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        // Generator anticommutes with the live Z term at every round (X on
        // the same qubit), so every round is a genuine branch opportunity.
        let (genx, genz) = planes_of(0b1, 0, 1);
        let gen = [genx.as_slice(), genz.as_slice()];
        let param = GateParam::Numeric { angle };

        for _ in 0..ROUNDS {
            let clifford_inplace = SymbolicCoeff::is_clifford_param(&param, CLIFFORD_COS_EPS);
            assert!(clifford_inplace, "a pi/2 numeric angle must be recognized as Clifford");
            kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(&mut evolved, gen, &param, clifford_inplace);
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut evolved);
        }

        assert_eq!(evolved.len(), 1, "the in-place branch must never grow the live term count");
        let total_monomials = kernels::sum_coeffs(&evolved, |c| c.monomial_count());
        assert_eq!(
            total_monomials, 1,
            "monomial count must stay flat across Clifford gates, not double per round"
        );
    }

    /// Same circuit as above but forcing the pre-fix `clifford_inplace =
    /// false` behavior, documenting the growth the fix eliminates: the live
    /// term count stabilizes at 2 (the branch's two possible Pauli strings,
    /// each gate's cos-branch and sin-branch always landing on one of the
    /// same two keys, so `merge` finds a duplicate to fold every round), but
    /// the monomial `count` those merges accumulate via `Node::add` still
    /// doubles every round even though only one real trig factor (`pi/2`) is
    /// ever involved.
    #[test]
    fn without_clifford_inplace_monomial_count_doubles_every_round() {
        const N_QUBITS: usize = 4;
        const ROUNDS: usize = 20;
        let angle = std::f64::consts::FRAC_PI_2;

        let mut evolved: SoaTermSum<SymbolicCoeff> = SoaTermSum::new(N_QUBITS, 1);
        let (gx, gz) = planes_of(0, 0b1, 1);
        evolved.push([&gx, &gz], SymbolicCoeff::from_real(1.0));

        let (genx, genz) = planes_of(0b1, 0, 1);
        let gen = [genx.as_slice(), genz.as_slice()];
        let param = GateParam::Numeric { angle };

        for _ in 0..ROUNDS {
            kernels::apply_rotation::<PauliBasis, SymbolicCoeff>(&mut evolved, gen, &param, false);
            kernels::merge::<PauliBasis, SymbolicCoeff>(&mut evolved);
        }

        assert_eq!(evolved.len(), 2, "term count stabilizes at the branch's 2 possible Pauli strings");
        let total_monomials = kernels::sum_coeffs(&evolved, |c| c.monomial_count());
        assert_eq!(
            total_monomials,
            1u128 << ROUNDS,
            "pre-fix behavior: monomial count doubles every round despite only one real trig factor",
        );
    }
}
