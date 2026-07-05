use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use propaq_core::coeff::CoeffRepr;

use crate::interning::Generation;

/// Below this monomial count, use the simple sort-based merge. Benchmarking
/// showed the hash-based merge only reliably wins with substantial exact
/// factor-pattern duplication (~100x+ repeats); with little/no duplication it
/// is comparable or slower at every scale tested, and gets relatively worse
/// as the count grows (larger hashtables mean more cache misses, while sort's
/// O(log k) term grows very slowly). Set conservatively high so only the rare
/// pathological terms (which dominate wall-clock time, and are more likely to
/// carry real duplication after surviving many merges) take the hash path,
/// while the common case keeps the well-tested sort with no regression risk.
const HASH_MERGE_THRESHOLD: usize = 100_000;

/// A trig factor is packed into a single `u32`:
/// `[param:16 | cos_pow:8 | sin_pow:8]`. The **parameter index** lives in the
/// high 16 bits, so a plain ascending `u32` sort orders factors by parameter —
/// which is what makes a monomial's factor run canonical (see `SymbolicCoeff`).
const PARAM_SHIFT: u32 = 16;
const COS_SHIFT: u32 = 8;
const POW_MASK: u32 = 0xff;

/// Pack `(param, cos_pow, sin_pow)` into a factor word. `pub(crate)` so the
/// interning layer (`interning.rs`) can materialize decoded runs.
#[inline]
pub(crate) fn make_factor(param: u32, cos_pow: u32, sin_pow: u32) -> u32 {
    (param << PARAM_SHIFT) | (cos_pow << COS_SHIFT) | sin_pow
}

#[inline]
pub(crate) fn factor_param(f: u32) -> u32 {
    f >> PARAM_SHIFT
}

#[inline]
pub(crate) fn factor_cos(f: u32) -> u32 {
    (f >> COS_SHIFT) & POW_MASK
}

#[inline]
pub(crate) fn factor_sin(f: u32) -> u32 {
    f & POW_MASK
}

/// Minimum monomials before `evaluate` splits a single term's monomial list
/// across threads. Higher than `apply_gate_inplace`'s gate threshold because
/// per-monomial evaluate cost is near-uniform (no coefficient-size skew), so the
/// per-split overhead only pays off for genuinely large terms. Overridable once
/// at process start via `PROPAQ_EVALUATE_PAR_MIN_LEN` for threshold sweeps (read
/// cached in a `LazyLock`; one relaxed load per `evaluate` call).
///
/// Set to 8192 from a sweep (28-thread Xeon, `grown_coeff` inputs in
/// `benches/surrogate_bench.rs`): at this `min_len` a ~4k-monomial coefficient
/// stays serial (≈184µs vs ≈237µs when split) while a ~64k one parallelizes
/// with coarse, cache-friendly chunks (≈1.13ms vs ≈1.73ms at min_len 1024 and
/// ≈3.42ms serial). Cluster-specific tuning may differ — hence the env override.
const EVALUATE_PAR_MIN_LEN_DEFAULT: usize = 8192;

#[inline]
fn evaluate_par_min_len() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("PROPAQ_EVALUATE_PAR_MIN_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(EVALUATE_PAR_MIN_LEN_DEFAULT)
    });
    *V
}

/// Whether numeric-branch coefficient sharing (copy-on-write via `Arc`) is on.
/// Default on; set `PROPAQ_DISABLE_COEFF_SHARING=1` to force an immediate copy at
/// every numeric branch (the pre-sharing baseline) — used to A/B the memory win
/// and as a safety escape hatch. Read once, cached.
#[inline]
fn coeff_sharing_enabled() -> bool {
    static V: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(std::env::var("PROPAQ_DISABLE_COEFF_SHARING").as_deref(), Ok("1") | Ok("true"))
    });
    *V
}

/// A monomial's "frequency" (symbolic branch degree): the total trig power it
/// carries, `Σ (cos_pow + sin_pow)` over its factors. This is the number of
/// symbolic gates that branched the path (multiplicities included), matching the
/// `(1/2)^ℓ` average-magnitude physics that drives frequency truncation.
#[inline]
fn run_frequency(factors: &[u32]) -> usize {
    factors.iter().map(|&f| (factor_cos(f) + factor_sin(f)) as usize).sum()
}

/// The highest parameter index recorded in a run (`0` for an empty run). Runs
/// are sorted ascending by parameter, so this is just the last factor's param.
#[inline]
#[cfg(test)]
fn run_last_param(factors: &[u32]) -> u32 {
    factors.last().map_or(0, |&f| factor_param(f))
}

/// Whether a run is canonical: parameters strictly ascending and every factor
/// carries a nonzero total power. Two monomials with the same `(param -> cos/sin
/// powers)` map therefore produce identical `u32` slices, so bit-for-bit slice
/// equality is semantic equality. Referenced by the `debug_assert!` in
/// `push_monomial` (evaluated only in debug builds) and by tests, so it stays
/// compiled in release without being a runtime cost there.
fn run_is_canonical(factors: &[u32]) -> bool {
    let mut prev: Option<u32> = None;
    for &f in factors {
        let p = factor_param(f);
        if let Some(pp) = prev {
            if p <= pp {
                return false;
            }
        }
        if factor_cos(f) + factor_sin(f) == 0 {
            return false;
        }
        prev = Some(p);
    }
    true
}

/// Copy `run` into `dst`, folding one more `cos`/`sin` branch on parameter
/// `param` into it. `run` is sorted ascending by parameter; if `param` is
/// already present its matching power is incremented, otherwise a fresh factor
/// is inserted at the sorted position. `O(#factors)`, and the result stays
/// canonical. This increment on a *repeated* parameter is the collapse mechanism
/// that keeps parameter-reusing ansätze (UCJ/LUCJ) from exploding: two paths
/// that reach the same `(param -> powers)` map produce identical runs and merge
/// under `deduplicate`.
#[inline]
fn write_incremented(dst: &mut Vec<u32>, run: &[u32], param: u32, is_sin: bool) {
    let mut i = 0usize;
    while i < run.len() && factor_param(run[i]) < param {
        dst.push(run[i]);
        i += 1;
    }
    if i < run.len() && factor_param(run[i]) == param {
        let f = run[i];
        let (mut cos_pow, mut sin_pow) = (factor_cos(f), factor_sin(f));
        if is_sin {
            sin_pow += 1;
        } else {
            cos_pow += 1;
        }
        debug_assert!(
            cos_pow <= POW_MASK && sin_pow <= POW_MASK,
            "trig power exceeds the 8-bit cap (255) for param {param}; \
             a single parameter branched >255 times on one path"
        );
        dst.push(make_factor(param, cos_pow, sin_pow));
        i += 1;
    } else {
        let (cos_pow, sin_pow) = if is_sin { (0, 1) } else { (1, 0) };
        dst.push(make_factor(param, cos_pow, sin_pow));
    }
    dst.extend_from_slice(&run[i..]);
}

/// Merge two canonical (ascending-param, params-distinct) factor runs into one
/// canonical run in `dst`, summing the cos/sin powers of any param that appears
/// in both. Used at reconciliation to fold a monomial's extension back into its
/// decoded base (a param branched during the window that was already in the base
/// lands in both halves and must recombine into a single factor).
fn merge_runs(dst: &mut Vec<u32>, a: &[u32], b: &[u32]) {
    let (mut i, mut j) = (0usize, 0usize);
    while i < a.len() && j < b.len() {
        let (pa, pb) = (factor_param(a[i]), factor_param(b[j]));
        if pa < pb {
            dst.push(a[i]);
            i += 1;
        } else if pb < pa {
            dst.push(b[j]);
            j += 1;
        } else {
            let cos_pow = factor_cos(a[i]) + factor_cos(b[j]);
            let sin_pow = factor_sin(a[i]) + factor_sin(b[j]);
            debug_assert!(
                cos_pow <= POW_MASK && sin_pow <= POW_MASK,
                "merged trig power exceeds the 8-bit cap (255) for param {pa}"
            );
            dst.push(make_factor(pa, cos_pow, sin_pow));
            i += 1;
            j += 1;
        }
    }
    dst.extend_from_slice(&a[i..]);
    dst.extend_from_slice(&b[j..]);
}

/// Per-monomial header for the support⊗exponent factored representation.
///
/// A monomial's full factor run is split into a **base** (the history interned
/// into a frozen [`Generation`](crate::interning::Generation) before the last
/// flush) plus an **extension** (the factors appended since, stored inline in
/// the owning coefficient's `factors` arena). The full run is
/// `generation.decode(base_support, base_exp) ++ factors[start..end]`.
///
/// `scalar` is real, not complex: `apply_rotation` is only ever invoked on
/// anticommuting (generator, term) pairs, and for Hermitian, involutory
/// operators the commutator phase in that case is always purely imaginary
/// (`±i`); multiplying by the explicit `i` in `apply_rotation` cancels it,
/// leaving a real result at every step. Given a real (Hermitian) seed
/// observable, every monomial's scalar stays real by induction.
///
/// `base_support`/`base_exp` are opaque ids into the current frozen generation's
/// support trie / exponent dictionary (`(0, 0)` = empty base, before any
/// reconciliation). `base_freq` caches the base's `Σ` powers so the hot path
/// recovers a monomial's frequency as `base_freq + Σ extension powers` **without
/// dereferencing the tables**. `end` is a u64 factor-count offset of this
/// monomial's *extension* in the arena (its start is the previous header's
/// `end`, or 0 for the first).
#[derive(Clone, Copy, Debug)]
struct MonoHead {
    scalar: f64,
    base_support: u32,
    base_exp: u32,
    base_freq: u32,
    end: u64,
}

impl MonoHead {
    /// A monomial whose base is empty (all factors live in the extension).
    #[inline]
    fn flat(scalar: f64, end: u64) -> Self {
        MonoHead { scalar, base_support: 0, base_exp: 0, base_freq: 0, end }
    }
}

/// Per-thread free-list of previously-live `(heads, factors)` buffer pairs,
/// recycled by `apply_rotation`'s branch construction and `deduplicate`'s
/// rebuild instead of round-tripping through the global allocator — both are
/// the hottest per-gate/per-flush allocation sites for `SymbolicCoeff`. Scoped
/// per OS thread (not passed explicitly) so this needs no change to
/// `CoeffRepr` or any caller: `AbstractPropagator`'s worker threads are
/// long-lived for a whole `build()` run, so buffers recycle across many
/// gates/flushes on the same thread.
const BUFFER_POOL_CAP: usize = 64;

thread_local! {
    static COEFF_BUFFER_POOL: RefCell<Vec<(Vec<MonoHead>, Vec<u32>)>> =
        const { RefCell::new(Vec::new()) };
}

/// Check out a buffer pair from this thread's pool, or a fresh empty pair if
/// none is available. Callers must `reserve` as needed before use.
fn take_pooled_buffers() -> (Vec<MonoHead>, Vec<u32>) {
    COEFF_BUFFER_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

/// Return a no-longer-needed buffer pair to this thread's pool for reuse, or
/// drop it normally if the pool is already at capacity.
fn return_pooled_buffers(mut heads: Vec<MonoHead>, mut factors: Vec<u32>) {
    heads.clear();
    factors.clear();
    COEFF_BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < BUFFER_POOL_CAP {
            pool.push((heads, factors));
        }
    });
}

thread_local! {
    static ORDER_POOL: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
}

/// Check out a sort-permutation buffer for `deduplicate`, or a fresh empty one.
fn take_order() -> Vec<u32> {
    ORDER_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

/// Return a sort-permutation buffer (clearing it, keeping capacity) for reuse.
fn return_order(mut order: Vec<u32>) {
    order.clear();
    ORDER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < BUFFER_POOL_CAP {
            pool.push(order);
        }
    });
}

/// A sum of monomials `scalar * product(trig factors)`: a symbolic
/// coefficient accumulated during surrogate propagation.
///
/// Each monomial stores its branch factors in **parameter space**: one `u32`
/// per distinct parameter it touched, packed `[param:16 | cos_pow:8 |
/// sin_pow:8]` and sorted ascending by parameter. A symbolic gate on parameter
/// `p` multiplies the monomial by `cos(θ_p)` (or `sin(θ_p)`), which increments
/// `p`'s cos/sin power (inserting the factor if `p` is new) — so a parameter
/// reused across many gates accumulates into powers rather than spawning a
/// distinct factor per gate. A commuting gate contributes nothing; a
/// numeric-angle gate is folded into the scalar and stores nothing. Storage is
/// O(#distinct params) per monomial.
///
/// The key property over a gate-indexed scheme: two paths that pick up the same
/// parameters to the same total powers (e.g. `cos(θ_p)·sin(θ_p)` reached via
/// gate-A-cos/gate-B-sin vs gate-A-sin/gate-B-cos on the same parameter)
/// produce **identical** runs, so `deduplicate` merges them. For ansätze that
/// reuse parameters this collapses the monomial count dramatically.
///
/// Stored in CSR/SoA form — one 16-byte header per monomial plus a single
/// shared `u32` arena — instead of one owning object per monomial. At the design
/// scale (hundreds of millions of monomials) the per-monomial representation
/// was the dominant cost: every clone/grow/merge did one allocator round-trip
/// *per monomial*. Here every operation is a streaming pass over two flat
/// buffers with at most one buffer rebuild per call.
///
/// Runs are canonical (ascending params, nonzero powers), so two monomials with
/// the same parameter/power map compare equal word-for-word. `add_assign` simply
/// appends monomials (and flags the coefficient dirty); call `deduplicate` to
/// merge identical runs and drop near-zero terms before evaluation.

/// An immutable, already-deduplicated monomial arena, shared behind an `Arc` by
/// the parent and child of a numeric branch. See `SymbolicCoeff::shared`.
#[derive(Default)]
struct Inner {
    heads: Vec<MonoHead>,
    factors: Vec<u32>,
}

#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    heads: Vec<MonoHead>,
    factors: Vec<u32>,
    /// Whether identical runs may exist across monomials. Only `add_assign`
    /// can introduce duplicates (rotations preserve pairwise distinctness,
    /// scaling doesn't touch factors), so `deduplicate` skips clean coefficients
    /// entirely — the common case at a flush, where most live terms received no
    /// inbox merges since the last one.
    dirty: bool,
    /// Copy-on-write structural sharing. When `Some((inner, mult))`, the owned
    /// `heads`/`factors` are empty and the logical value is `mult · inner` — a
    /// shared, immutable, canonical arena. Created only at a **numeric** branch
    /// (`apply_rotation_numeric`), where parent and child are the same arena up
    /// to a scalar; this makes numeric branching O(1) instead of copying the
    /// whole arena. Any *structural* mutation (`realize`) materializes it back to
    /// the owned representation, folding `mult` into each scalar. A shared value
    /// is always clean (its `inner` was deduplicated when wrapped), so
    /// `deduplicate`/`post_merge` are no-ops on it — it survives the merge cadence
    /// unrealized, which is the whole point for numeric-heavy circuits.
    shared: Option<(Arc<Inner>, f64)>,
}

impl SymbolicCoeff {
    /// Single scalar monomial with an empty run (no params branched); used to
    /// seed from the observable.
    pub fn from_scalar(c: f64) -> Self {
        SymbolicCoeff {
            heads: vec![MonoHead::flat(c, 0)],
            factors: Vec::new(),
            dirty: false,
            shared: None,
        }
    }

    /// `(heads, factors, mult)` view over the effective monomials — the owned
    /// buffers with `mult = 1`, or the shared arena with its multiplier. Read-only
    /// operations use this so they transparently handle a shared value without
    /// realizing it (which would collapse the sharing).
    #[inline]
    fn view(&self) -> (&[MonoHead], &[u32], f64) {
        match &self.shared {
            Some((inner, mult)) => (&inner.heads, &inner.factors, *mult),
            None => (self.heads.as_slice(), self.factors.as_slice(), 1.0),
        }
    }

    /// Identity of the shared arena (its `Arc` pointer address), or `None` if the
    /// value is owned. Distinct terms that share one arena report the same id;
    /// used by `cluster_bench` to measure the terms-per-arena ratio.
    pub fn arena_ptr(&self) -> Option<usize> {
        self.shared.as_ref().map(|(inner, _)| Arc::as_ptr(inner) as usize)
    }

    /// Materialize a shared value into the owned representation (folding `mult`
    /// into each scalar) so a mutating op can proceed. No-op on an owned value.
    /// `pub` so the compile step can flatten every term before building the model.
    pub fn realize(&mut self) {
        if let Some((inner, mult)) = self.shared.take() {
            // The owned buffers are empty while shared; refill them in place.
            self.factors.extend_from_slice(&inner.factors);
            self.heads.reserve(inner.heads.len());
            self.heads.extend(
                inner.heads.iter().map(|h| MonoHead { scalar: h.scalar * mult, ..*h }),
            );
            self.dirty = false; // inner was deduplicated when shared
        }
    }

    /// Ensure `self` is a shared value and return a clone of its `Arc<Inner>`.
    /// Wrapping an owned value deduplicates it first, upholding the invariant that
    /// a shared arena is always canonical.
    fn ensure_shared(&mut self) -> Arc<Inner> {
        if let Some((inner, _)) = &self.shared {
            return Arc::clone(inner);
        }
        self.deduplicate(); // no-op when already clean
        let inner = Arc::new(Inner {
            heads: std::mem::take(&mut self.heads),
            factors: std::mem::take(&mut self.factors),
        });
        self.shared = Some((Arc::clone(&inner), 1.0));
        self.dirty = false;
        inner
    }

    /// Start offset of monomial `i`'s factor run.
    #[inline]
    fn start(&self, i: usize) -> usize {
        if i == 0 { 0 } else { self.heads[i - 1].end as usize }
    }

    /// Extension factor run of monomial `i` (the factors appended since the last
    /// reconciliation; the full run is `decode(base) ++ this`).
    #[inline]
    fn factor_run(&self, i: usize) -> &[u32] {
        &self.factors[self.start(i)..self.heads[i].end as usize]
    }

    /// Identity key of monomial `i` for dedup: `(base_support, base_exp)`. Two
    /// monomials are the same iff these match *and* their extension runs match.
    #[inline]
    fn mono_base(&self, i: usize) -> (u32, u32) {
        (self.heads[i].base_support, self.heads[i].base_exp)
    }

    pub fn monomial_count(&self) -> usize {
        self.view().0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.view().0.is_empty()
    }

    /// Iterate `(scalar, extension run)` per monomial, in storage order. Scalars
    /// are scaled by the shared multiplier when the value is shared.
    ///
    /// This yields only the **extension** — the factors appended since the last
    /// reconciliation. When the base is empty (every monomial `(0, 0)`, e.g. a
    /// freshly-built or deserialized-flat coefficient) that *is* the full run,
    /// which is how the tests and the flat serialization path use it. Callers
    /// that must see the full run of a base-populated coefficient decode the
    /// base against the generation explicitly (see `evaluate` / reconciliation).
    pub fn iter_monomials(&self) -> impl Iterator<Item = (f64, &[u32])> + '_ {
        let (heads, factors, mult) = self.view();
        let mut start = 0usize;
        heads.iter().map(move |h| {
            let end = h.end as usize;
            let run = &factors[start..end];
            start = end;
            (h.scalar * mult, run)
        })
    }

    /// Iterate `(scalar, base_support, base_exp, extension run)` per monomial,
    /// in storage order, scaling scalars by the shared multiplier. The
    /// serialization path uses this to write each monomial's base ids (which
    /// reference the model's generation) plus any residual extension.
    pub(crate) fn iter_factored(&self) -> impl Iterator<Item = (f64, u32, u32, &[u32])> + '_ {
        let (heads, factors, mult) = self.view();
        let mut start = 0usize;
        heads.iter().map(move |h| {
            let end = h.end as usize;
            let run = &factors[start..end];
            start = end;
            (h.scalar * mult, h.base_support, h.base_exp, run)
        })
    }

    /// Append one monomial with an empty base. `run` must be a canonical factor
    /// run (params ascending, powers nonzero) — this is the deserialization/test
    /// construction entry point, and it writes the whole run into the extension.
    pub fn push_monomial(&mut self, scalar: f64, run: &[u32]) {
        debug_assert!(run_is_canonical(run), "factor run must be canonical (ascending params)");
        self.realize();
        self.factors.extend_from_slice(run);
        self.heads.push(MonoHead::flat(scalar, self.factors.len() as u64));
    }

    /// Append one monomial that already carries a base `(support, exp, freq)`
    /// plus a (possibly empty) canonical extension run. Used when reconstructing
    /// factored coefficients (model load) and by the reconciliation rewrite.
    pub(crate) fn push_factored(
        &mut self,
        scalar: f64,
        base_support: u32,
        base_exp: u32,
        base_freq: u32,
        extension: &[u32],
    ) {
        debug_assert!(run_is_canonical(extension), "extension run must be canonical");
        self.realize();
        self.factors.extend_from_slice(extension);
        self.heads.push(MonoHead {
            scalar,
            base_support,
            base_exp,
            base_freq,
            end: self.factors.len() as u64,
        });
    }

    /// Reserve for `n_monomials` headers and `n_factors` arena factors.
    pub fn reserve(&mut self, n_monomials: usize, n_factors: usize) {
        self.heads.reserve(n_monomials);
        self.factors.reserve(n_factors);
    }

    /// Advance this coefficient into a new generation: fold every monomial's
    /// full run (its base decoded against `old` merged with its extension) into
    /// `new`'s interning tables, replacing the base ids and clearing the
    /// extension. `old` must be the generation the current base ids reference.
    ///
    /// After this, every monomial has an empty extension and a base id pair into
    /// `new`. Because re-interning is content-addressed, two monomials that
    /// reached the same full run via different base/extension splits now share
    /// one base id, so a `deduplicate` merges them (the cross-lineage merge the
    /// mid-window dedup defers). Realizes a shared value first.
    pub(crate) fn reconcile_into(&mut self, old: &Generation, new: &mut Generation) {
        self.realize();
        if self.heads.is_empty() {
            return;
        }
        let (mut new_heads, unused) = take_pooled_buffers();
        new_heads.reserve(self.heads.len());
        let mut full: Vec<u32> = Vec::new();
        let mut base_dec: Vec<u32> = Vec::new();
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            let ext = &self.factors[start..end];
            start = end;
            // Full canonical run = decode(old base) merged with the extension.
            let merged: &[u32] = if h.base_support == 0 && h.base_exp == 0 {
                ext
            } else {
                base_dec.clear();
                old.decode_into(h.base_support, h.base_exp, &mut base_dec);
                full.clear();
                merge_runs(&mut full, &base_dec, ext);
                &full
            };
            let (support, exp, freq) = new.intern_run(merged);
            new_heads.push(MonoHead { scalar: h.scalar, base_support: support, base_exp: exp, base_freq: freq, end: 0 });
        }
        let old_heads = std::mem::replace(&mut self.heads, new_heads);
        self.factors.clear();
        return_pooled_buffers(old_heads, unused);
        // Cross-lineage duplicates now share base ids; collapse them.
        self.dirty = true;
        self.deduplicate();
    }

    /// Drop monomials with frequency (total trig power) > max_freq, compacting
    /// the arena in place (no allocation).
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.compact(|freq, _| freq <= max_freq);
    }

    /// Drop every monomial whose frequency equals exactly `freq`, in place.
    pub fn remove_at_frequency(&mut self, freq: usize) {
        self.compact(|freq_i, _| freq_i != freq);
    }

    /// Drop monomials whose scalar prefactor is smaller in magnitude than
    /// `min_abs`, compacting in place. Valid as a contribution bound because the
    /// symbolic trig product is `<= 1` in magnitude (see `CoefficientTruncator`);
    /// intended to run *after* `deduplicate` so it sees merged scalars.
    pub fn trim_small_scalars(&mut self, min_abs: f64) {
        self.compact(|_, scalar| scalar.abs() >= min_abs);
    }

    /// In-place compaction keeping monomials for which `keep(frequency,
    /// scalar)` holds. Writes never overtake reads (removal only shrinks), so
    /// both buffers are rewritten in one forward pass with zero allocation.
    /// Realizes a shared value first (monomial-level removal must mutate the
    /// arena; this is where a shared value's sharing is lost under monomial-level
    /// truncation — term-level truncation drops whole terms instead).
    fn compact(&mut self, mut keep: impl FnMut(usize, f64) -> bool) {
        self.realize();
        let mut w_head = 0usize;
        let mut w_fac = 0usize;
        let mut start = 0usize;
        for i in 0..self.heads.len() {
            let head = self.heads[i];
            let end = head.end as usize;
            let freq = head.base_freq as usize + run_frequency(&self.factors[start..end]);
            if keep(freq, head.scalar) {
                let len = end - start;
                if w_fac != start {
                    self.factors.copy_within(start..end, w_fac);
                }
                w_fac += len;
                self.heads[w_head] = MonoHead { end: w_fac as u64, ..head };
                w_head += 1;
            }
            start = end;
        }
        self.heads.truncate(w_head);
        self.factors.truncate(w_fac);
    }

    /// Merge monomials with identical runs and drop near-zero results. Skips
    /// clean coefficients outright (see the `dirty` field docs); a consequence
    /// is that near-zero scalars are only pruned on dirty coefficients — which
    /// matches where they can arise, since destructive cancellation requires a
    /// merge in the first place.
    ///
    /// Below `HASH_MERGE_THRESHOLD` monomials, sorts an index permutation
    /// (comparing arena slices) and merges adjacent equal runs (`O(k log k)`,
    /// cheap for small `k`). Above it, accumulates scalars in a hashmap keyed
    /// by *borrowed* arena slices (`O(k)` amortized). Either path rebuilds the
    /// two flat buffers once; there is no per-monomial allocation.
    pub fn deduplicate(&mut self) {
        if !self.dirty || self.heads.len() <= 1 {
            self.dirty = false;
            return;
        }
        self.dirty = false;

        if self.heads.len() < HASH_MERGE_THRESHOLD {
            let mut order = take_order();
            order.extend(0..self.heads.len() as u32);
            // Order by (base_support, base_exp) then the extension run, so all
            // identical monomials (same base *and* extension) are adjacent.
            order.sort_unstable_by(|&a, &b| {
                self.mono_base(a as usize)
                    .cmp(&self.mono_base(b as usize))
                    .then_with(|| self.factor_run(a as usize).cmp(self.factor_run(b as usize)))
            });

            let (mut heads, mut factors) = take_pooled_buffers();
            heads.reserve(self.heads.len());
            factors.reserve(self.factors.len());
            let mut i = 0usize;
            while i < order.len() {
                let head = self.heads[order[i] as usize];
                let base = (head.base_support, head.base_exp);
                let run = self.factor_run(order[i] as usize);
                let mut scalar = head.scalar;
                let mut j = i + 1;
                while j < order.len()
                    && self.mono_base(order[j] as usize) == base
                    && self.factor_run(order[j] as usize) == run
                {
                    scalar += self.heads[order[j] as usize].scalar;
                    j += 1;
                }
                if scalar.abs() > 1e-15 {
                    factors.extend_from_slice(run);
                    heads.push(MonoHead { scalar, end: factors.len() as u64, ..head });
                }
                i = j;
            }
            let old_heads = std::mem::replace(&mut self.heads, heads);
            let old_factors = std::mem::replace(&mut self.factors, factors);
            return_pooled_buffers(old_heads, old_factors);
            return_order(order);
            return;
        }

        // Key on (base_support, base_exp, extension run); value carries the
        // summed scalar and the group's base_freq (identical across a group).
        let mut acc: FxHashMap<(u32, u32, &[u32]), (f64, u32)> = FxHashMap::default();
        acc.reserve(self.heads.len());
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            let slot = acc
                .entry((h.base_support, h.base_exp, &self.factors[start..end]))
                .or_insert((0.0, h.base_freq));
            slot.0 += h.scalar;
            start = end;
        }
        let (mut heads, mut factors) = take_pooled_buffers();
        heads.reserve(acc.len());
        factors.reserve(self.factors.len());
        for ((base_support, base_exp, run), (scalar, base_freq)) in acc {
            if scalar.abs() > 1e-15 {
                factors.extend_from_slice(run);
                heads.push(MonoHead {
                    scalar,
                    base_support,
                    base_exp,
                    base_freq,
                    end: factors.len() as u64,
                });
            }
        }
        let old_heads = std::mem::replace(&mut self.heads, heads);
        let old_factors = std::mem::replace(&mut self.factors, factors);
        return_pooled_buffers(old_heads, old_factors);
    }

    /// Evaluate against a flat LUT indexed by `2 * param` (`cos`) /
    /// `2 * param + 1` (`sin`). Each monomial walks its factor run once, raising
    /// each parameter's `cos`/`sin` to the recorded powers (`powi`) — the
    /// parameter index is stored directly in the factor, so there is no
    /// gate→param indirection at evaluate time.
    ///
    /// `SurrogateModel::evaluate` already parallelizes across terms, which
    /// covers the common case; but a handful of terms can carry the
    /// overwhelming majority of monomials, leaving other threads idle while one
    /// churns through a huge single-term monomial list serially. `with_min_len`
    /// lets rayon's splitter fall back to a single sequential chunk for
    /// ordinary (small) terms while still splitting (and letting idle threads
    /// steal via the outer per-term `par_iter`) once a term's monomial count is
    /// large enough to be worth it.
    pub fn evaluate(&self, gen: &Generation, lut: &[f64]) -> f64 {
        let (heads, factors, mult) = self.view();
        (0..heads.len())
            .into_par_iter()
            .with_min_len(evaluate_par_min_len())
            .map(|i| {
                let h = heads[i];
                let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                let end = h.end as usize;
                let mut prod = h.scalar * mult;
                // Base contribution (interned history); skipped for an empty base
                // so a flat coefficient needs no table access.
                if h.base_support != 0 || h.base_exp != 0 {
                    prod *= gen.base_product(h.base_support, h.base_exp, lut);
                }
                for &f in &factors[start..end] {
                    let p = factor_param(f) as usize;
                    let cos_pow = factor_cos(f) as i32;
                    let sin_pow = factor_sin(f) as i32;
                    if cos_pow > 0 {
                        prod *= lut[2 * p].powi(cos_pow);
                    }
                    if sin_pow > 0 {
                        prod *= lut[2 * p + 1].powi(sin_pow);
                    }
                }
                prod
            })
            .sum()
    }

    /// Highest frequency present and how many monomials sit at exactly that
    /// frequency; `(0, 0)` if empty. Parallel over monomial chunks (same skew
    /// rationale as `evaluate`) so one giant coefficient doesn't serialize the
    /// truncation pass that calls this per live term.
    pub fn top_frequency_and_count(&self) -> (usize, usize) {
        const PAR_MIN_LEN: usize = 65_536;
        let (heads, factors, _mult) = self.view();
        (0..heads.len())
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || (0usize, 0usize),
                |(mut freq, mut count), i| {
                    let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                    let end = heads[i].end as usize;
                    let len = heads[i].base_freq as usize + run_frequency(&factors[start..end]);
                    match len.cmp(&freq) {
                        std::cmp::Ordering::Greater => { freq = len; count = 1; }
                        std::cmp::Ordering::Equal => count += 1,
                        std::cmp::Ordering::Less => {}
                    }
                    (freq, count)
                },
            )
            .reduce(|| (0, 0), Self::combine_top_frequency)
    }

    /// Merge two `(top frequency, count at that frequency)` aggregates.
    pub fn combine_top_frequency(a: (usize, usize), b: (usize, usize)) -> (usize, usize) {
        match a.0.cmp(&b.0) {
            std::cmp::Ordering::Greater => a,
            std::cmp::Ordering::Less => b,
            std::cmp::Ordering::Equal => (a.0, a.1 + b.1),
        }
    }

    /// Remove monomials whose frequency equals exactly `freq`, claiming
    /// removals from a `remaining` budget shared across every coefficient
    /// processed in the same pass. Returns how many were removed.
    ///
    /// Counts this coefficient's own hits first (a local, read-only scan),
    /// then claims `min(hits, remaining)` in a single compare-exchange loop:
    /// one atomic operation per coefficient that actually has a hit, not per
    /// monomial.
    pub fn remove_at_frequency_budgeted(&mut self, freq: usize, remaining: &AtomicUsize) -> usize {
        self.realize();
        let mut hits = 0usize;
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            if h.base_freq as usize + run_frequency(&self.factors[start..end]) == freq {
                hits += 1;
            }
            start = end;
        }
        if hits == 0 {
            return 0;
        }

        let mut cur = remaining.load(Ordering::Relaxed);
        let claim = loop {
            let take = hits.min(cur);
            if take == 0 {
                return 0;
            }
            match remaining.compare_exchange_weak(cur, cur - take, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break take,
                Err(actual) => cur = actual,
            }
        };

        let mut removed = 0usize;
        self.compact(|freq_i, _| {
            if freq_i == freq && removed < claim {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
    }

    /// Add this coefficient's monomial frequencies into `hist` (`hist[f]` counts
    /// monomials of frequency `f`), growing it as needed. Used to build the
    /// global frequency histogram that drives importance-ranked truncation.
    pub fn add_freq_histogram(&self, hist: &mut Vec<u64>) {
        let (heads, factors, _mult) = self.view();
        let mut start = 0usize;
        for h in heads {
            let end = h.end as usize;
            let f = h.base_freq as usize + run_frequency(&factors[start..end]);
            if f >= hist.len() {
                hist.resize(f + 1, 0);
            }
            hist[f] += 1;
            start = end;
        }
    }

    /// Append `|scalar|` of every monomial at exactly frequency `freq` to `out`.
    /// The gathered values from all coefficients feed a single `select_nth`
    /// that picks the scalar threshold within the boundary-frequency bucket.
    pub fn collect_boundary_scalars(&self, freq: usize, out: &mut Vec<f64>) {
        let (heads, factors, mult) = self.view();
        let mut start = 0usize;
        for h in heads {
            let end = h.end as usize;
            if h.base_freq as usize + run_frequency(&factors[start..end]) == freq {
                out.push((h.scalar * mult).abs());
            }
            start = end;
        }
    }

    /// Importance-ranked removal by key `(frequency desc, |scalar| asc)`:
    ///
    /// - every monomial with `frequency > f_star` is removed (the buckets above
    ///   the boundary, all less important than anything at the boundary);
    /// - within `frequency == f_star`, monomials with `|scalar| < s_star` are
    ///   removed unconditionally (they are the smallest, and — by the
    ///   `select_nth` that produced `s_star` — globally fewer than the boundary
    ///   budget, so removing all of them never overshoots);
    /// - the `|scalar| == s_star` ties are removed only while the shared
    ///   `tie_budget` allows, so the pass lands exactly at the target count.
    ///
    /// Returns the number of monomials removed. Ties are claimed once per
    /// coefficient (count locally, then a single compare-exchange), mirroring
    /// `remove_at_frequency_budgeted`. `s_star = INFINITY` removes the whole
    /// boundary bucket (used when the budget consumes it entirely).
    pub fn remove_by_rank_budgeted(&mut self, f_star: usize, s_star: f64, tie_budget: &AtomicUsize) -> usize {
        self.realize();
        let mut tie_hits = 0usize;
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            if h.base_freq as usize + run_frequency(&self.factors[start..end]) == f_star
                && h.scalar.abs() == s_star
            {
                tie_hits += 1;
            }
            start = end;
        }

        let claim = if tie_hits == 0 {
            0
        } else {
            let mut cur = tie_budget.load(Ordering::Relaxed);
            loop {
                let take = tie_hits.min(cur);
                if take == 0 {
                    break 0;
                }
                match tie_budget.compare_exchange_weak(cur, cur - take, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break take,
                    Err(actual) => cur = actual,
                }
            }
        };

        let mut removed = 0usize;
        let mut ties_removed = 0usize;
        self.compact(|freq, scalar| {
            if freq > f_star {
                removed += 1;
                false
            } else if freq == f_star {
                let a = scalar.abs();
                if a < s_star {
                    removed += 1;
                    false
                } else if a == s_star && ties_removed < claim {
                    ties_removed += 1;
                    removed += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        removed
    }

    /// Symbolic rotation on parameter `param`: records that every monomial's path
    /// branched on this parameter, picking up `cos` (kept in place on `self`) or
    /// `sin` (returned as the new anticommuted term). The parameter index is
    /// written directly into the run — evaluation reads it straight from the
    /// factor, with no gate→param table.
    ///
    /// Both branches stream each monomial's existing factors into a fresh arena,
    /// folding one more `cos`/`sin` power on `param` (see `write_incremented`):
    /// if the monomial already touched `param` its power is bumped in place,
    /// otherwise a factor is inserted at the sorted position. A repeated
    /// parameter therefore grows a power instead of adding a factor — the
    /// collapse mechanism for parameter-reusing ansätze. Both branches remain
    /// pairwise distinct (the increment is injective), so no dedup is forced.
    ///
    /// `prune_freq` enables look-ahead pruning: the sin child's frequency is its
    /// parent's + 1 (one more unit of total power), so a parent already at
    /// `>= cap` would produce a child a lossy `max_frequency` flush discards.
    /// When set, such children are never generated. The cos branch is left
    /// untouched (it stays in an existing term, trimmed at the next flush as
    /// before). The propagator only passes `Some` when this is provably
    /// equivalent to the deferred trim.
    fn apply_rotation_symbolic(&mut self, param: u32, prune_freq: Option<u32>, phase: Complex64) -> Self {
        // sin branch scalar: * (i * phase). `phase` is always ±i here (only
        // called on anticommuting generator/term pairs), so `i * phase` is
        // always real — see the `MonoHead::scalar` doc comment.
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        // A symbolic gate folds a factor into every monomial, so the shared
        // immutable arena can't be reused — materialize first. (Rare for the
        // numeric-heavy workload this sharing targets.)
        self.realize();
        let n = self.heads.len();

        // Sin branch first, while `self`'s arena is still un-rebuilt. Buffers
        // come from this thread's pool instead of a fresh allocation. Each
        // monomial grows by at most one factor word.
        let (mut sin_heads, mut sin_factors) = take_pooled_buffers();
        sin_heads.reserve(n);
        sin_factors.reserve(self.factors.len() + n);

        let mut start = 0usize;
        for head in &self.heads {
            let end = head.end as usize;
            let run = &self.factors[start..end];
            start = end;

            // Look-ahead: the sin child's frequency is this monomial's full
            // frequency (base + extension) + 1; skip emitting it if that would
            // exceed the cap.
            if let Some(cap) = prune_freq {
                if head.base_freq + run_frequency(run) as u32 >= cap {
                    continue;
                }
            }

            write_incremented(&mut sin_factors, run, param, true);
            sin_heads.push(MonoHead {
                scalar: head.scalar * branch_phase,
                end: sin_factors.len() as u64,
                ..*head
            });
        }

        // Cos branch: rebuild `self`, folding this parameter's cos factor into
        // every monomial's run (they all took the cos branch).
        let (mut new_heads, mut new_factors) = take_pooled_buffers();
        new_heads.reserve(n);
        new_factors.reserve(self.factors.len() + n);
        let mut start = 0usize;
        for head in &self.heads {
            let end = head.end as usize;
            let run = &self.factors[start..end];
            start = end;
            write_incremented(&mut new_factors, run, param, false);
            new_heads.push(MonoHead { end: new_factors.len() as u64, ..*head });
        }
        let old_heads = std::mem::replace(&mut self.heads, new_heads);
        let old_factors = std::mem::replace(&mut self.factors, new_factors);
        return_pooled_buffers(old_heads, old_factors);

        // Duplicates in self (if any) are duplicated into the branch too.
        SymbolicCoeff { heads: sin_heads, factors: sin_factors, dirty: self.dirty, shared: None }
    }

    /// Numeric-angle rotation: `cos`/`sin` of `angle` are computed immediately
    /// (mirrors `Complex64::apply_rotation` exactly). Numeric gates carry no
    /// symbolic information — no factor is written — so parent (cos branch) and
    /// child (sin branch) are the *same* monomial arena up to a scalar multiple.
    ///
    /// Rather than deep-copying the arena, this **shares** it: `self` is wrapped
    /// into an immutable `Arc` once (`ensure_shared`), then the cos branch scales
    /// `self`'s multiplier and the sin branch returns a new value referencing the
    /// same arena. Both branches are O(1). The copy is deferred to `realize`,
    /// which only fires on a structural mutation — for a numeric-heavy circuit
    /// that means the *rare* symbolic gate, so a whole numeric sub-tree of
    /// distinct terms shares one arena. See the `shared` field.
    fn apply_rotation_numeric(&mut self, angle: f64, phase: Complex64) -> Self {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        // Mirrors `apply_rotation_symbolic`'s `branch_phase`, scaled by
        // `sin_t`: `phase` is always ±i here, so `sin_t * (i * phase)` is real.
        let branch_phase = Complex64::new(0.0, sin_t) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let inner = self.ensure_shared();
        // Multiplier before this gate: `cos` scales `self`, `sin` seeds the child.
        let mult_pre = self.shared.as_ref().unwrap().1;
        self.shared.as_mut().unwrap().1 = mult_pre * cos_t;

        let mut sin = SymbolicCoeff {
            heads: Vec::new(),
            factors: Vec::new(),
            dirty: false,
            shared: Some((inner, mult_pre * branch_phase)),
        };
        // Baseline / escape hatch: materialize both branches immediately, i.e.
        // the pre-sharing deep-copy behavior.
        if !coeff_sharing_enabled() {
            self.realize();
            sin.realize();
        }
        sin
    }
}

/// Gate parameter for a symbolic rotation: either a symbolic parameter (a slot
/// in the parameter vector, accumulated as a tracked factor in the run and
/// resolved later by `evaluate` against the LUT) or a concrete numeric angle
/// baked in immediately (mirrors `Complex64::apply_rotation`'s math and never
/// touches the run).
///
/// `Symbolic` carries `param` (the parameter index behind this gate — written
/// directly into every branching monomial's run) and the optional look-ahead
/// `prune_freq` cap (injected by the propagator after extraction). `Numeric`
/// carries only its `angle` — a numeric rotation never grows a monomial's
/// frequency and stores nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GateParam {
    Symbolic { param: u32, prune_freq: Option<u32> },
    Numeric { angle: f64 },
}

impl GateParam {
    /// A symbolic gate on parameter `x`, with no look-ahead pruning. Convenience
    /// for the Python extraction path (the propagator injects the real
    /// `prune_freq` afterward), tests, and benchmarks.
    #[inline]
    pub fn symbolic(x: u32) -> Self {
        GateParam::Symbolic { param: x, prune_freq: None }
    }
}

impl CoeffRepr for SymbolicCoeff {
    /// Gate parameter is either a symbolic parameter or a concrete numeric
    /// angle; see `GateParam`.
    type GateParam = GateParam;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        // Seed observables are Hermitian, so their Pauli/Majorana-basis
        // coefficients are real; see the `MonoHead::scalar` doc comment.
        debug_assert!(c.im.abs() < 1e-9, "surrogate seed coefficient must be real: {c:?}");
        SymbolicCoeff::from_scalar(c.re)
    }

    fn add_assign(&mut self, mut other: Self) {
        // `self` is the additive-identity default (empty owned, not a ref): take
        // `other` wholesale, preserving any shared ref it carries. Common at a
        // flush: `map.entry(term).or_default().add_assign(coeff)`.
        if self.shared.is_none() && self.heads.is_empty() {
            *self = other;
            return;
        }
        // `other` is the additive identity: nothing to add.
        if other.shared.is_none() && other.heads.is_empty() {
            return;
        }
        // Same shared arena — the common case for two numeric-branch siblings
        // that merged onto one Pauli string: O(1), just add the multipliers, no
        // realize (the shared arena stays shared).
        if let (Some((ai, am)), Some((bi, bm))) = (&mut self.shared, &other.shared) {
            if Arc::ptr_eq(ai, bi) {
                *am += *bm;
                return;
            }
        }
        // General case: materialize both and concatenate monomials.
        self.realize();
        other.realize();
        let ext_base = self.factors.len() as u64;
        self.factors.append(&mut other.factors);
        self.heads.reserve(other.heads.len());
        self.heads.extend(other.heads.iter().map(|h| MonoHead { end: h.end + ext_base, ..*h }));
        self.dirty = true;
        return_pooled_buffers(other.heads, other.factors);
    }

    /// Dispatches to `apply_rotation_symbolic` (branch factor recorded) for a
    /// symbolic gate or `apply_rotation_numeric` (cos/sin folded into each
    /// scalar) for a concrete angle.
    fn apply_rotation(&mut self, param: &GateParam, phase: Complex64) -> Self {
        match param {
            GateParam::Symbolic { param, prune_freq } => {
                self.apply_rotation_symbolic(*param, *prune_freq, phase)
            }
            GateParam::Numeric { angle } => self.apply_rotation_numeric(*angle, phase),
        }
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        // Scaling a shared value is O(1) — just the multiplier; it stays shared,
        // so uniform noise between symbolic gates never forces a realize.
        if let Some((_, mult)) = &mut self.shared {
            *mult *= factor;
            return;
        }
        for h in &mut self.heads {
            h.scalar *= factor;
        }
    }

    /// Collapse monomials with identical runs that `add_assign` just
    /// juxtaposed. Without this, a periodic outbox merge only dedupes at the
    /// term-key level — runs that happen to coincide (every monomial from a
    /// purely-numeric gate history shares the same empty run) pile up as
    /// separate entries until the next full truncation flush. `deduplicate`
    /// already no-ops on a clean (non-`dirty`) coefficient, so this costs
    /// nothing when `add_assign` had nothing new to fold in.
    #[inline]
    fn post_merge(&mut self) {
        self.deduplicate();
    }

    /// L1 norm is undefined for symbolic; return 0 to skip coeff-based truncation.
    #[inline]
    fn l1_norm(&self) -> f64 {
        0.0
    }

    /// Monomial count is what actually drives memory/CPU cost for symbolic
    /// coefficients, unlike raw term count. Reads through a shared value.
    #[inline]
    fn size_hint(&self) -> usize {
        self.monomial_count()
    }

    #[inline]
    fn prefetch_read(&self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; the pointers are this
        // coefficient's own live (or shared) buffers.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            let (heads, factors, _mult) = self.view();
            let ptr = factors.as_ptr() as *const i8;
            let bytes = std::mem::size_of_val(factors);
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
            let ptr = heads.as_ptr() as *const i8;
            let bytes = std::mem::size_of_val(heads);
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
        }
    }

    /// A rotation's `param_index` (`Optional[int]`) takes precedence: if
    /// present, the gate is symbolic. Otherwise falls back to `angle`
    /// (`float`), a concrete numeric angle baked in at build time.
    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<GateParam> {
        let param_index: Option<u32> = obj.getattr("param_index")?.extract()?;
        if let Some(param) = param_index {
            return Ok(GateParam::Symbolic { param, prune_freq: None });
        }
        let angle: f64 = obj.getattr("angle")?.extract()?;
        Ok(GateParam::Numeric { angle })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Encode a set of `(param, is_sin)` branches into a canonical factor run:
    /// cos/sin powers per parameter accumulated, sorted ascending by parameter.
    fn enc(branches: &[(u32, bool)]) -> Vec<u32> {
        let mut map: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
        for &(p, is_sin) in branches {
            let e = map.entry(p).or_insert((0, 0));
            if is_sin {
                e.1 += 1;
            } else {
                e.0 += 1;
            }
        }
        map.into_iter().map(|(p, (c, s))| make_factor(p, c, s)).collect()
    }

    /// Build a coefficient from raw `(scalar, [(param, is_sin)])` monomials.
    /// Flagged dirty like a real post-merge coefficient.
    fn coeff(monomials: &[(f64, &[(u32, bool)])]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, branches) in monomials {
            c.push_monomial(scalar, &enc(branches));
        }
        c.dirty = true;
        c
    }

    /// Coefficient of monomials with exactly the given `(scalar, frequency,
    /// tag)` specs: `frequency` distinct cos branches, at parameter positions
    /// made unique per tag so summed frequency equals `freq` directly.
    fn coeff_with_freqs(specs: &[(f64, usize, u32)]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, freq, tag) in specs {
            let branches: Vec<(u32, bool)> = (0..freq as u32).map(|p| (tag * 1000 + p, false)).collect();
            c.push_monomial(scalar, &enc(&branches));
        }
        c
    }

    /// Reference evaluation independent of `evaluate`'s parallel path.
    fn naive_evaluate(c: &SymbolicCoeff, lut: &[f64]) -> f64 {
        c.iter_monomials()
            .map(|(scalar, run)| {
                let mut prod = scalar;
                for &f in run {
                    let p = factor_param(f) as usize;
                    let cos_pow = factor_cos(f) as i32;
                    let sin_pow = factor_sin(f) as i32;
                    prod *= lut[2 * p].powi(cos_pow) * lut[2 * p + 1].powi(sin_pow);
                }
                prod
            })
            .sum()
    }

    fn make_lut(n_params: usize) -> Vec<f64> {
        (0..n_params)
            .flat_map(|i| {
                let t = 0.37 * (i as f64 + 1.0);
                [t.cos(), t.sin()]
            })
            .collect()
    }

    #[test]
    fn factor_pack_round_trips() {
        let f = make_factor(12345, 200, 55);
        assert_eq!(factor_param(f), 12345);
        assert_eq!(factor_cos(f), 200);
        assert_eq!(factor_sin(f), 55);
    }

    #[test]
    fn push_and_iter_round_trip() {
        let c = coeff(&[
            (1.5, &[(1, false), (0, true)]),
            (-2.0, &[(3, false)]),
            (0.5, &[]),
        ]);
        let collected: Vec<(f64, Vec<u32>)> =
            c.iter_monomials().map(|(s, run)| (s, run.to_vec())).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 1.5);
        // param 0 sin, param 1 cos, in ascending-param canonical order.
        assert_eq!(collected[0].1, vec![make_factor(0, 0, 1), make_factor(1, 1, 0)]);
        assert_eq!(collected[1].1, vec![make_factor(3, 1, 0)]);
        assert!(collected[2].1.is_empty());
        assert_eq!(c.monomial_count(), 3);
    }

    #[test]
    fn apply_rotation_matches_trig_identity_and_keeps_runs_canonical() {
        let lut = make_lut(8);
        let mut c = SymbolicCoeff::from_scalar(0.75);
        // Distinct parameter indices.
        for param in [0u32, 1, 2, 5, 7] {
            let before = naive_evaluate(&c, &lut);
            let sin_branch = c.apply_rotation(&GateParam::symbolic(param), Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(param << 1) as usize], lut[((param << 1) | 1) as usize]);
            assert!((naive_evaluate(&c, &lut) - cos_t * before).abs() < 1e-12);
            // branch_phase = (i * -i).re = 1.0
            assert!((naive_evaluate(&sin_branch, &lut) - sin_t * before).abs() < 1e-12);
            for (_, run) in c.iter_monomials().chain(sin_branch.iter_monomials()) {
                assert!(run_is_canonical(run), "factor run must stay canonical (ascending params)");
            }
        }
    }

    #[test]
    fn same_parameter_at_two_gates_multiplies_as_a_power_and_collapses() {
        // Two rotations on the SAME parameter (two gates behind one theta, as in
        // UCJ/LUCJ): the cos branch accumulates into a single cos^2 factor — one
        // monomial, not two — which is the whole point of the parameter-space
        // representation over the gate-indexed scheme.
        let lut = make_lut(1);
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        assert_eq!(c.monomial_count(), 1, "same param at two gates must merge into one monomial");
        let (_, run) = c.iter_monomials().next().unwrap();
        assert_eq!(run, &[make_factor(0, 2, 0)]);
        let expected = lut[0] * lut[0]; // cos(theta_0)^2
        assert!((naive_evaluate(&c, &lut) - expected).abs() < 1e-12);
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-12);
    }

    #[test]
    fn two_paths_through_same_parameter_dedup_to_one_monomial() {
        // The collapse the gate-indexed scheme could not do: cos·sin on one
        // parameter reached two ways (gate A cos / gate B sin, and gate A sin /
        // gate B cos, both on param 0) yields the SAME factor run, so after
        // add_assign + deduplicate the two paths merge into a single monomial.
        let lut = make_lut(1);
        // The two branch orderings, each already accumulated into one factor by
        // enc, land on the same run and sum under deduplicate.
        let mut c = coeff(&[
            (1.0, &[(0, false), (0, true)]), // cos·sin via A-cos, B-sin
            (1.0, &[(0, true), (0, false)]), // cos·sin via A-sin, B-cos
        ]);
        c.deduplicate();
        assert_eq!(c.monomial_count(), 1);
        let (scalar, run) = c.iter_monomials().next().unwrap();
        assert_eq!(run, &[make_factor(0, 1, 1)]);
        assert!((scalar - 2.0).abs() < 1e-12);
        let expected = 2.0 * lut[0] * lut[1];
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_numeric_matches_trig_identity() {
        let c0 = 0.75;
        let angle = 0.4;
        let phase = Complex64::new(0.0, -1.0);

        let mut c = SymbolicCoeff::from_scalar(c0);
        let sin_branch = c.apply_rotation(&GateParam::Numeric { angle }, phase);

        let (cos_scalar, cos_run) = c.iter_monomials().next().unwrap();
        assert!((cos_scalar - c0 * angle.cos()).abs() < 1e-12);
        assert!(cos_run.is_empty());

        let (sin_scalar, sin_run) = sin_branch.iter_monomials().next().unwrap();
        assert!((sin_scalar - c0 * angle.sin()).abs() < 1e-12);
        assert!(sin_run.is_empty());
    }

    #[test]
    fn apply_rotation_numeric_never_touches_the_run() {
        let mut c = SymbolicCoeff::from_scalar(1.0);
        // Seed some pre-existing symbolic branches first, as in a mixed circuit.
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(1), Complex64::new(0.0, -1.0));

        let runs_before: Vec<Vec<u32>> = c.iter_monomials().map(|(_, run)| run.to_vec()).collect();

        let sin_branch = c.apply_rotation(&GateParam::Numeric { angle: 0.3 }, Complex64::new(0.0, -1.0));

        let runs_after: Vec<Vec<u32>> = c.iter_monomials().map(|(_, run)| run.to_vec()).collect();
        let sin_runs: Vec<Vec<u32>> = sin_branch.iter_monomials().map(|(_, run)| run.to_vec()).collect();

        assert_eq!(runs_before, runs_after, "numeric rotation must not touch the cos branch's run");
        assert_eq!(runs_before, sin_runs, "numeric rotation must not touch the sin branch's run");
    }

    #[test]
    fn apply_rotation_mixed_numeric_then_symbolic_composes_correctly() {
        let c0: f64 = 1.0;
        let angle: f64 = 0.6;
        let param = 3u32;
        let phase = Complex64::new(0.0, -1.0);
        let lut = make_lut(8);
        let (cos_t_sym, sin_t_sym) = (lut[(2 * param) as usize], lut[(2 * param + 1) as usize]);
        let (cos_num, sin_num) = (angle.cos(), angle.sin());

        // Numeric first, then symbolic on both resulting branches.
        let mut cos_branch = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch = cos_branch.apply_rotation(&GateParam::Numeric { angle }, phase);
        let cos_cos = cos_branch.apply_rotation(&GateParam::symbolic(param), phase);
        let sin_cos = sin_branch.apply_rotation(&GateParam::symbolic(param), phase);

        assert!((naive_evaluate(&cos_branch, &lut) - c0 * cos_num * cos_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&cos_cos, &lut) - c0 * cos_num * sin_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&sin_branch, &lut) - c0 * sin_num * cos_t_sym).abs() < 1e-12);
        assert!((naive_evaluate(&sin_cos, &lut) - c0 * sin_num * sin_t_sym).abs() < 1e-12);

        // Symbolic first, then numeric on both resulting branches -- same four
        // outcomes, order must not matter.
        let mut cos_branch2 = SymbolicCoeff::from_scalar(c0);
        let mut sin_branch2 = cos_branch2.apply_rotation(&GateParam::symbolic(param), phase);
        let cos_num2 = cos_branch2.apply_rotation(&GateParam::Numeric { angle }, phase);
        let sin_num2 = sin_branch2.apply_rotation(&GateParam::Numeric { angle }, phase);

        assert!((naive_evaluate(&cos_branch2, &lut) - c0 * cos_t_sym * cos_num).abs() < 1e-12);
        assert!((naive_evaluate(&cos_num2, &lut) - c0 * cos_t_sym * sin_num).abs() < 1e-12);
        assert!((naive_evaluate(&sin_branch2, &lut) - c0 * sin_t_sym * cos_num).abs() < 1e-12);
        assert!((naive_evaluate(&sin_num2, &lut) - c0 * sin_t_sym * sin_num).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_numeric_scalar_matches_complex64_apply_rotation() {
        let c0 = 0.42;
        let angle = 1.1;
        let phase = Complex64::new(0.0, -1.0);

        let mut symbolic = SymbolicCoeff::from_scalar(c0);
        let symbolic_sin = symbolic.apply_rotation(&GateParam::Numeric { angle }, phase);

        let mut complex = Complex64::new(c0, 0.0);
        let complex_sin = complex.apply_rotation(&angle, phase);

        let (cos_scalar, _) = symbolic.iter_monomials().next().unwrap();
        let (sin_scalar, _) = symbolic_sin.iter_monomials().next().unwrap();

        assert!(complex_sin.im.abs() < 1e-12);
        assert!((cos_scalar - complex.re).abs() < 1e-12);
        assert!((sin_scalar - complex_sin.re).abs() < 1e-12);
    }

    #[test]
    fn numeric_branch_shares_one_arena() {
        // A numeric branch makes the cos branch (self) and the returned sin
        // branch reference the *same* immutable arena — O(1), no copy.
        let mut c = coeff(&[(1.0, &[(0, false)]), (2.0, &[(1, true)])]);
        let sin = c.apply_rotation(&GateParam::Numeric { angle: 0.7 }, Complex64::new(0.0, -1.0));
        let a = c.shared.as_ref().expect("cos branch is shared");
        let b = sin.shared.as_ref().expect("sin branch is shared");
        assert!(Arc::ptr_eq(&a.0, &b.0), "cos and sin share the same arena");
        // While shared, the owned buffers are empty.
        assert!(c.heads.is_empty() && c.factors.is_empty());
    }

    #[test]
    fn realize_reproduces_shared_value_exactly() {
        let lut = make_lut(8);
        let mut c = coeff(&[(1.5, &[(0, false)]), (2.0, &[(1, true)])]);
        let sin = c.apply_rotation(&GateParam::Numeric { angle: 0.7 }, Complex64::new(0.0, -1.0));

        let before_eval = sin.evaluate(&crate::interning::Generation::new(), &lut);
        let before: Vec<(f64, Vec<u32>)> = sin.iter_monomials().map(|(s, r)| (s, r.to_vec())).collect();

        let mut realized = sin.clone();
        realized.realize();
        assert!(realized.shared.is_none(), "realize materializes to owned");

        assert!((realized.evaluate(&crate::interning::Generation::new(), &lut) - before_eval).abs() < 1e-12);
        let after: Vec<(f64, Vec<u32>)> = realized.iter_monomials().map(|(s, r)| (s, r.to_vec())).collect();
        assert_eq!(before.len(), after.len());
        for ((s1, r1), (s2, r2)) in before.iter().zip(&after) {
            assert!((s1 - s2).abs() < 1e-12);
            assert_eq!(r1, r2);
        }
    }

    #[test]
    fn same_arena_add_assign_sums_multipliers_and_stays_shared() {
        let lut = make_lut(4);
        let mut c = coeff(&[(1.0, &[])]); // single empty-run monomial, value 1
        let sin = c.apply_rotation(&GateParam::Numeric { angle: 0.7 }, Complex64::new(0.0, -1.0));
        let expected = c.evaluate(&crate::interning::Generation::new(), &lut) + sin.evaluate(&crate::interning::Generation::new(), &lut);
        let arc_before = Arc::clone(&c.shared.as_ref().unwrap().0);
        c.add_assign(sin);
        assert!(c.shared.is_some(), "same-arena add stays shared");
        assert!(Arc::ptr_eq(&arc_before, &c.shared.as_ref().unwrap().0), "same arena reused, no realize");
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-12);
    }

    #[test]
    fn scale_real_and_post_merge_keep_shared() {
        let lut = make_lut(4);
        let mut c = coeff(&[(2.0, &[(0, false)])]);
        let _sin = c.apply_rotation(&GateParam::Numeric { angle: 0.5 }, Complex64::new(0.0, -1.0));
        let before = c.evaluate(&crate::interning::Generation::new(), &lut);
        c.scale_real(3.0);
        assert!(c.shared.is_some(), "scale_real is O(1), no realize");
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - 3.0 * before).abs() < 1e-12);
        c.post_merge();
        assert!(c.shared.is_some(), "post_merge (dedup) is a no-op on a shared value");
    }

    #[test]
    fn symbolic_gate_realizes_shared() {
        let lut = make_lut(16);
        let mut c = coeff(&[(1.0, &[(0, false)])]);
        let _num = c.apply_rotation(&GateParam::Numeric { angle: 0.4 }, Complex64::new(0.0, -1.0));
        assert!(c.shared.is_some());
        let c_val = c.evaluate(&crate::interning::Generation::new(), &lut);
        let _sym = c.apply_rotation(&GateParam::symbolic(8), Complex64::new(0.0, -1.0));
        assert!(c.shared.is_none(), "a symbolic gate realizes the shared value");
        // cos branch picks up cos(theta_8): param 8 -> lut[16].
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - c_val * lut[16]).abs() < 1e-12);
    }

    #[test]
    fn apply_rotation_symbolic_prune_matches_unpruned_then_trim() {
        let cap = 2u32;
        let param = 7u32;
        let phase = Complex64::new(0.0, -1.0);
        // Monomials spanning frequencies 0..=3 on params other than `param` (so
        // every branch is at a fresh position); the sin child of each has
        // frequency parent + 1, i.e. 1..=4.
        let build = || {
            coeff(&[
                (1.5, &[]),                                   // freq 0 -> sin freq 1
                (-2.0, &[(0, false)]),                        // freq 1 -> sin freq 2
                (0.7, &[(0, false), (1, true)]),              // freq 2 -> sin freq 3 (pruned)
                (3.1, &[(0, false), (1, true), (2, false)]),  // freq 3 -> sin freq 4 (pruned)
            ])
        };
        let lut = make_lut(8);

        // Pruned rotation: sin children above the cap are never generated.
        let mut c_prune = build();
        let sin_prune =
            c_prune.apply_rotation(&GateParam::Symbolic { param, prune_freq: Some(cap) }, phase);

        // Reference: full rotation, then trim the sin branch to the same cap.
        let mut c_ref = build();
        let mut sin_ref =
            c_ref.apply_rotation(&GateParam::Symbolic { param, prune_freq: None }, phase);
        sin_ref.trim_high_frequency(cap as usize);

        assert!((naive_evaluate(&sin_prune, &lut) - naive_evaluate(&sin_ref, &lut)).abs() < 1e-12);
        assert_eq!(sin_prune.monomial_count(), 2);
        assert_eq!(sin_ref.monomial_count(), 2);
        assert!((naive_evaluate(&c_prune, &lut) - naive_evaluate(&c_ref, &lut)).abs() < 1e-12);
    }

    #[test]
    fn add_assign_into_empty_moves_without_copy_semantics_change() {
        let src = coeff(&[(1.0, &[(0, false)]), (2.0, &[(1, true)])]);
        let mut dst = SymbolicCoeff::default();
        dst.add_assign(src.clone());
        let lut = make_lut(4);
        assert!((naive_evaluate(&dst, &lut) - naive_evaluate(&src, &lut)).abs() < 1e-15);
        assert_eq!(dst.monomial_count(), 2);
    }

    #[test]
    fn add_assign_rebases_offsets_and_marks_dirty() {
        let mut a = coeff(&[(1.0, &[(0, false), (1, false)])]);
        a.dirty = false;
        let b = coeff(&[(2.0, &[(2, true)]), (3.0, &[])]);
        let lut = make_lut(4);
        let expected = naive_evaluate(&a, &lut) + naive_evaluate(&b, &lut);
        a.add_assign(b);
        assert!(a.dirty);
        assert_eq!(a.monomial_count(), 3);
        assert!((naive_evaluate(&a, &lut) - expected).abs() < 1e-12);
        let runs: Vec<Vec<u32>> = a.iter_monomials().map(|(_, r)| r.to_vec()).collect();
        assert_eq!(runs[1], vec![make_factor(2, 0, 1)]); // param 2 sin
        assert!(runs[2].is_empty());
    }

    #[test]
    fn dedup_merges_same_pattern_regardless_of_input_order() {
        let mut c = coeff(&[
            (1.0, &[(1, false), (2, false)]),
            (2.0, &[(2, false), (1, false)]),
        ]);
        c.deduplicate();
        assert_eq!(c.monomial_count(), 1);
        let (scalar, _) = c.iter_monomials().next().unwrap();
        assert!((scalar - 3.0).abs() < 1e-12);
    }

    #[test]
    fn dedup_drops_near_zero_after_merge() {
        let mut c = coeff(&[
            (1.0, &[(0, false)]),
            (-1.0, &[(0, false)]),
        ]);
        c.deduplicate();
        assert!(c.is_empty());
        assert!(c.factors.is_empty());
    }

    #[test]
    fn dedup_skips_clean_coefficients() {
        let mut c = coeff(&[(1.0, &[(0, false)]), (2.0, &[(0, false)])]);
        c.dirty = false;
        c.deduplicate();
        assert_eq!(c.monomial_count(), 2);

        let mut a = coeff(&[(1.0, &[(0, false)])]);
        a.dirty = false;
        let mut b = coeff(&[(2.0, &[(0, false)])]);
        b.dirty = false;
        a.add_assign(b);
        a.deduplicate();
        assert_eq!(a.monomial_count(), 1);
        assert!(!a.dirty);
    }

    /// Reproduces the pattern that motivated `CoeffRepr::post_merge`: a term
    /// receives several outbox entries whose lineage is purely numeric (every
    /// monomial has the same empty run, `dirty == false`, exactly like the
    /// output of `apply_rotation_numeric`). Mirrors `flush_outboxes_to_maps`'s
    /// `entry.add_assign(coeff); entry.post_merge();` sequence one push at a
    /// time. Without `post_merge` these would pile up as separate (but
    /// identical) monomials until the next truncation flush; with it, the
    /// coefficient collapses back to a single monomial after every merge that
    /// actually combined something.
    #[test]
    fn post_merge_collapses_repeated_numeric_history_pushes() {
        let fresh_numeric_push = |scalar: f64| {
            let mut c = SymbolicCoeff::from_scalar(scalar);
            c.dirty = false;
            c
        };

        // First push into an "empty map slot": `*self = other`, nothing to merge yet.
        let mut entry = SymbolicCoeff::default();
        entry.add_assign(fresh_numeric_push(1.0));
        entry.post_merge();
        assert_eq!(entry.monomial_count(), 1);

        // Second push for the same term: a real merge, so post_merge must fire.
        entry.add_assign(fresh_numeric_push(2.0));
        entry.post_merge();
        assert_eq!(entry.monomial_count(), 1, "identical empty-run monomials must collapse immediately, not linger until a truncation flush");
        let (scalar, run) = entry.iter_monomials().next().unwrap();
        assert!((scalar - 3.0).abs() < 1e-12);
        assert!(run.is_empty());

        // A third push keeps it collapsed rather than accumulating further.
        entry.add_assign(fresh_numeric_push(-3.0));
        entry.post_merge();
        assert!(entry.is_empty(), "a merge summing to (near) zero must still be pruned, not left as a live monomial");
    }

    #[test]
    fn hash_path_matches_naive_evaluation() {
        let n_params = 400;
        let lut = make_lut(n_params);

        // > HASH_MERGE_THRESHOLD monomials with many repeated factor patterns,
        // inserted in varying order, to exercise the hash-merge path and its
        // order-independence. i != j always, so each monomial's two branches
        // are at distinct parameter positions.
        let mut c = SymbolicCoeff::default();
        for rep in 0..3usize {
            for i in 0..n_params as u32 {
                for j in 0..n_params as u32 {
                    if i == j {
                        continue;
                    }
                    // Lower param cos, higher param sin (canonical order handled by enc).
                    let (lo, hi) = (i.min(j), i.max(j));
                    c.push_monomial(0.1 * (rep as f64 + 1.0), &enc(&[(lo, false), (hi, true)]));
                }
            }
        }
        c.dirty = true;
        assert!(c.monomial_count() >= HASH_MERGE_THRESHOLD, "test setup should exercise the hash path");

        let expected = naive_evaluate(&c, &lut);
        c.deduplicate();
        let actual = c.evaluate(&crate::interning::Generation::new(), &lut);
        assert!(
            (actual - expected).abs() < 1e-9,
            "hash-merge path changed the evaluated value: {actual} vs {expected}"
        );
    }

    #[test]
    fn small_and_large_paths_agree() {
        let lut = make_lut(8);

        let base: &[(f64, &[(u32, bool)])] = &[
            (1.0, &[(0, false), (1, true)]),
            (2.0, &[(0, false), (1, true)]),
            (-0.5, &[(2, false)]),
        ];
        let mut small = coeff(base);
        let expected = naive_evaluate(&small, &lut);
        small.deduplicate();
        assert!(small.monomial_count() < HASH_MERGE_THRESHOLD);
        assert!((small.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-12);

        let mut large = coeff(base);
        // Fixed param 3 cos plus a varying param drawn from 4..8 (distinct from
        // 3, within lut range), paired as exactly-cancelling entries.
        for k in 0..HASH_MERGE_THRESHOLD as u32 {
            let run = enc(&[(3, false), (4 + (k % 4), true)]);
            large.push_monomial(5.0, &run);
            large.push_monomial(-5.0, &run);
        }
        assert!(large.monomial_count() >= HASH_MERGE_THRESHOLD);
        assert!((naive_evaluate(&large, &lut) - expected).abs() < 1e-9);
        large.deduplicate();
        assert!((large.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-9);
    }

    #[test]
    fn trim_high_frequency_compacts_in_place() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 1, 1), (3.0, 4, 2), (4.0, 2, 3)]);
        c.trim_high_frequency(2);
        assert_eq!(c.monomial_count(), 2);
        let kept: Vec<(f64, usize)> =
            c.iter_monomials().map(|(s, r)| (s, run_frequency(r))).collect();
        assert_eq!(kept, vec![(2.0, 1), (4.0, 2)]);
    }

    #[test]
    fn top_frequency_and_count_finds_top_bucket_only() {
        let c = coeff_with_freqs(&[(1.0, 3, 0), (1.0, 5, 1), (1.0, 5, 2), (1.0, 2, 3)]);
        assert_eq!(c.top_frequency_and_count(), (5, 2));
        assert_eq!(SymbolicCoeff::default().top_frequency_and_count(), (0, 0));
    }

    #[test]
    fn budget_covers_all_hits_removes_all_and_claims_exactly_hits() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 3, 1), (3.0, 1, 2)]);
        let remaining = AtomicUsize::new(10);
        let removed = c.remove_at_frequency_budgeted(3, &remaining);
        assert_eq!(removed, 2);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(run_frequency(c.iter_monomials().next().unwrap().1), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 8);
    }

    #[test]
    fn budget_smaller_than_hits_removes_only_budget_and_exhausts_it() {
        let mut c = coeff_with_freqs(&[(1.0, 5, 0), (2.0, 5, 1), (3.0, 5, 2)]);
        let remaining = AtomicUsize::new(2);
        let removed = c.remove_at_frequency_budgeted(5, &remaining);
        assert_eq!(removed, 2);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn no_matching_frequency_is_a_no_op_and_touches_no_budget() {
        let mut c = coeff_with_freqs(&[(1.0, 2, 0)]);
        let remaining = AtomicUsize::new(5);
        assert_eq!(c.remove_at_frequency_budgeted(9, &remaining), 0);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(remaining.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn zero_remaining_budget_removes_nothing() {
        let mut c = coeff_with_freqs(&[(1.0, 4, 0)]);
        let remaining = AtomicUsize::new(0);
        assert_eq!(c.remove_at_frequency_budgeted(4, &remaining), 0);
        assert_eq!(c.monomial_count(), 1);
    }

    #[test]
    fn shared_budget_across_multiple_coefficients_never_exceeds_total() {
        let remaining = AtomicUsize::new(4);
        let mut a = coeff_with_freqs(&[(1.0, 6, 0), (1.0, 6, 1), (1.0, 6, 2)]);
        let mut b = coeff_with_freqs(&[(1.0, 6, 10), (1.0, 6, 11), (1.0, 6, 12)]);
        let removed_a = a.remove_at_frequency_budgeted(6, &remaining);
        let removed_b = b.remove_at_frequency_budgeted(6, &remaining);
        assert_eq!(removed_a + removed_b, 4);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);
        assert_eq!(a.monomial_count() + b.monomial_count(), 6 - 4);
    }

    #[test]
    fn remove_at_frequency_removes_exactly_that_bucket() {
        let mut c = coeff_with_freqs(&[(1.0, 2, 0), (2.0, 3, 1), (3.0, 2, 2), (4.0, 1, 3)]);
        c.remove_at_frequency(2);
        let lens: Vec<usize> = c.iter_monomials().map(|(_, r)| run_frequency(r)).collect();
        assert_eq!(lens, vec![3, 1]);
    }

    #[test]
    fn add_freq_histogram_and_collect_boundary_scalars() {
        let c = coeff_with_freqs(&[(0.1, 2, 0), (0.5, 2, 1), (0.9, 1, 2)]);
        let mut hist: Vec<u64> = Vec::new();
        c.add_freq_histogram(&mut hist);
        assert_eq!(hist, vec![0, 1, 2]); // freq1: 1, freq2: 2
        let mut scalars: Vec<f64> = Vec::new();
        c.collect_boundary_scalars(2, &mut scalars);
        scalars.sort_by(f64::total_cmp);
        assert_eq!(scalars, vec![0.1, 0.5]);
    }

    #[test]
    fn remove_by_rank_removes_smallest_scalars_in_boundary() {
        // Boundary f*=2, cutoff s*=0.5: drop |scalar|<0.5 (the 0.1) plus one tie
        // at exactly 0.5, keeping the larger 0.9 and the lower-frequency 2.0.
        let mut c = coeff_with_freqs(&[(0.1, 2, 0), (0.5, 2, 1), (0.9, 2, 2), (2.0, 1, 3)]);
        let budget = AtomicUsize::new(1);
        let removed = c.remove_by_rank_budgeted(2, 0.5, &budget);
        assert_eq!(removed, 2);
        let kept: Vec<(f64, usize)> =
            c.iter_monomials().map(|(s, r)| (s, run_frequency(r))).collect();
        assert_eq!(kept, vec![(0.9, 2), (2.0, 1)]);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn remove_by_rank_infinite_threshold_clears_boundary_and_above() {
        // s*=INFINITY: remove everything at freq > f* and all of freq == f*,
        // leaving only the strictly-lower-frequency monomial.
        let mut c = coeff_with_freqs(&[(0.1, 3, 0), (0.5, 2, 1), (0.9, 2, 2), (2.0, 1, 3)]);
        let budget = AtomicUsize::new(0);
        let removed = c.remove_by_rank_budgeted(2, f64::INFINITY, &budget);
        assert_eq!(removed, 3);
        let kept: Vec<(f64, usize)> =
            c.iter_monomials().map(|(s, r)| (s, run_frequency(r))).collect();
        assert_eq!(kept, vec![(2.0, 1)]);
    }

    #[test]
    fn evaluate_parallel_matches_naive_at_scale() {
        let n_params = 64usize;
        let lut = make_lut(n_params);
        let mut c = SymbolicCoeff::default();
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = (state % 6) as usize;
            // Draw `len` distinct parameter positions with random cos/sin codes.
            let mut used: BTreeMap<u32, bool> = BTreeMap::new();
            for k in 0..len {
                let v = (state >> (8 * k)) as u32 % (2 * n_params as u32);
                let (param, is_sin) = (v >> 1, v & 1 == 1);
                used.entry(param).or_insert(is_sin);
            }
            let branches: Vec<(u32, bool)> = used.into_iter().collect();
            c.push_monomial(((state % 1000) as f64 - 500.0) / 250.0, &enc(&branches));
        }
        let expected = naive_evaluate(&c, &lut);
        assert!((c.evaluate(&crate::interning::Generation::new(), &lut) - expected).abs() < 1e-9 * expected.abs().max(1.0));
    }

    #[test]
    fn run_last_param_reports_highest_param() {
        assert_eq!(run_last_param(&[]), 0);
        assert_eq!(run_last_param(&enc(&[(2, false), (9, true), (5, false)])), 9);
    }

    #[test]
    fn reconcile_preserves_value_and_folds_extension_into_base() {
        let lut = make_lut(8);
        let mut c = coeff(&[
            (1.5, &[(0, false), (3, true)]),
            (-2.0, &[(0, false)]),
            (0.75, &[(1, false), (3, false)]),
        ]);
        c.deduplicate();
        // Flat baseline (empty base ⇒ table-free evaluation).
        let before = c.evaluate(&Generation::new(), &lut);

        // First reconcile: empty old generation ⇒ moves each full run into the
        // base of a fresh generation, clearing extensions.
        let mut gen1 = Generation::new();
        c.reconcile_into(&Generation::new(), &mut gen1);
        assert!(
            c.iter_monomials().all(|(_, run)| run.is_empty()),
            "extensions must be empty after reconciliation"
        );
        assert!(c.heads.iter().any(|h| h.base_support != 0 || h.base_exp != 0), "base populated");
        let after1 = c.evaluate(&gen1, &lut);
        assert!((after1 - before).abs() < 1e-12, "reconcile changed the value");

        // Next window: symbolic gates append to the extension, including param 0
        // which already lives in the base (exercises merge_runs recombination).
        let _ = c.apply_rotation(&GateParam::symbolic(0), Complex64::new(0.0, -1.0));
        let _ = c.apply_rotation(&GateParam::symbolic(5), Complex64::new(0.0, -1.0));
        let mid = c.evaluate(&gen1, &lut);

        // Second reconcile: decode against gen1, fold the extension back in.
        let mut gen2 = Generation::new();
        c.reconcile_into(&gen1, &mut gen2);
        assert!(c.iter_monomials().all(|(_, run)| run.is_empty()));
        let after2 = c.evaluate(&gen2, &lut);
        assert!((after2 - mid).abs() < 1e-12, "second reconcile changed the value");
    }

    #[test]
    fn reconcile_merges_cross_lineage_duplicates() {
        let lut = make_lut(4);
        // Two monomials that are equal as full runs but were built to look
        // distinct; after reconcile they share base ids and must merge.
        let mut c = SymbolicCoeff::default();
        c.push_monomial(1.0, &enc(&[(0, false), (2, true)]));
        c.push_monomial(3.0, &enc(&[(0, false), (2, true)]));
        c.dirty = true;
        let before = c.evaluate(&Generation::new(), &lut);

        let mut gen = Generation::new();
        c.reconcile_into(&Generation::new(), &mut gen);
        assert_eq!(c.monomial_count(), 1, "identical full runs collapse to one monomial");
        let (scalar, _) = c.iter_monomials().next().unwrap();
        assert!((scalar - 4.0).abs() < 1e-12);
        assert!((c.evaluate(&gen, &lut) - before).abs() < 1e-12);
    }
}
