use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

use num_complex::Complex64;
use pyo3::prelude::*;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use propaq_core::coeff::CoeffRepr;

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

/// Packed per-parameter exponent pair: one arena slot per *distinct*
/// parameter touched by a monomial, storing how many times `cos`/`sin` of
/// that parameter multiply together -- not one slot per occurrence like the
/// tally-mark design this replaced. `cos^3(theta_0) * sin^2(theta_1)` costs
/// 2 slots (16 bytes) instead of 5 (20 bytes), and a repeat touch of an
/// already-present parameter during propagation becomes an O(1) in-place
/// exponent bump instead of growing the arena.
///
/// bits 63-32: param_index (u32, ~2 billion distinct parameters)
/// bits 31-16: cos_exponent (u16)
/// bits 15-0:  sin_exponent (u16)
///
/// `param_index` occupies the high bits specifically so that `Ord` on the
/// packed `u64` sorts by parameter first "for free" -- exactly the
/// invariant a monomial's sorted factor run needs. `deduplicate`'s sort path
/// and `apply_rotation`'s binary search both just compare/hash the raw
/// `u64`/param, with no field extraction needed on the common path.
///
/// Invariant (checked in `push_monomial`): within one monomial's run, each
/// distinct parameter appears in *at most one* `MonomialUnit`. Repeat
/// touches -- including a parameter that picks up both a `cos` and a `sin`
/// factor across different gates in one monomial's lineage -- accumulate
/// into that one unit's `cos_exp`/`sin_exp` fields, never as duplicate
/// slots.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MonomialUnit(u64);

impl MonomialUnit {
    #[inline]
    pub fn new(param: u32, cos_exp: u16, sin_exp: u16) -> Self {
        MonomialUnit((param as u64) << 32 | (cos_exp as u64) << 16 | sin_exp as u64)
    }

    #[inline]
    pub fn param(self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline]
    pub fn cos_exp(self) -> u16 {
        (self.0 >> 16) as u16
    }

    #[inline]
    pub fn sin_exp(self) -> u16 {
        self.0 as u16
    }

    /// Bump this unit's cos exponent by one, for a repeat touch of an
    /// already-present parameter. A real (not `debug_assert`) check: this
    /// must hold in release builds, where the design-scale runs this
    /// project cares about actually happen, and silently carrying into the
    /// adjacent `param_index` bits on overflow would be exactly the kind of
    /// corruption this project has already been bitten by once in
    /// production (see `cluster_bench`'s doc comment on unbounded blowups).
    /// The branch this guards is cheap and well-predicted relative to the
    /// arithmetic and memory movement already happening per call.
    #[inline]
    pub fn inc_cos(self) -> Self {
        assert!(self.cos_exp() < u16::MAX, "cos exponent overflow for param {}", self.param());
        MonomialUnit(self.0 + (1 << 16))
    }

    /// Bump this unit's sin exponent by one. See `inc_cos` for the overflow
    /// guard rationale.
    #[inline]
    pub fn inc_sin(self) -> Self {
        assert!(self.sin_exp() < u16::MAX, "sin exponent overflow for param {}", self.param());
        MonomialUnit(self.0 + 1)
    }

    /// Total trig-factor degree this single unit represents (`cos_exp +
    /// sin_exp`) -- how many of the old tally-mark tokens it replaces. This,
    /// summed across a monomial's whole run, is that monomial's physical
    /// "frequency": no longer just the run's slot count, since one slot can
    /// now hold `cos^a * sin^b`.
    #[inline]
    pub fn frequency(self) -> usize {
        self.cos_exp() as usize + self.sin_exp() as usize
    }

    /// Raw packed `u64`, for `model.rs`'s binary save/load format.
    #[inline]
    pub(crate) fn raw(self) -> u64 {
        self.0
    }

    /// Reconstruct from a raw packed `u64` (the inverse of `raw`), for
    /// `model.rs`'s load path.
    #[inline]
    pub(crate) fn from_raw(v: u64) -> Self {
        MonomialUnit(v)
    }
}

/// Per-monomial header: the scalar plus the *exclusive* end offset of this
/// monomial's factor run in the owning coefficient's `factors` arena (its
/// start is the previous header's `end`, or 0 for the first monomial).
///
/// `scalar` is real, not complex: `apply_rotation` is only ever invoked on
/// anticommuting (generator, term) pairs, and for Hermitian, involutory
/// operators the commutator phase in that case is always purely imaginary
/// (`±i`); multiplying by the explicit `i` in `apply_rotation` cancels it,
/// leaving a real result at every step. Given a real (Hermitian) seed
/// observable, every monomial's scalar stays real by induction.
///
/// `end` is u64, not u32/usize-of-index: a single coefficient can
/// legitimately hold billions of factors at the design scale (hundreds of
/// millions of monomials, heavily skewed toward a few terms), which would
/// overflow u32; and the fixed layout (8 + 8 bytes, no padding) is what
/// keeps per-monomial overhead at 16 bytes.
#[derive(Clone, Copy, Debug)]
struct MonoHead {
    scalar: f64,
    end: u64,
}

/// Per-thread free-list of previously-live `(heads, factors)` buffer pairs,
/// recycled by `apply_rotation`'s sin branch and `deduplicate`'s rebuild
/// instead of round-tripping through the global allocator — both are the
/// hottest per-gate/per-flush allocation sites for `SymbolicCoeff`. Scoped
/// per OS thread (not passed explicitly) so this needs no change to
/// `CoeffRepr` or any caller: `AbstractPropagator`'s worker threads are
/// long-lived for a whole `build()` run, so buffers recycle across many
/// gates/flushes on the same thread.
///
/// Only fed by `deduplicate`'s old-buffer replacement (the other natural
/// source — a whole coefficient being dropped by truncation's `retain` — has
/// no hook to intercept without a `Drop` impl, which risks the well-known
/// thread-local-during-shutdown footgun; not worth it for a pool that's
/// already fed by the common case). Capped so a burst of large coefficients
/// doesn't pin oversized capacity in idle pooled buffers indefinitely.
const BUFFER_POOL_CAP: usize = 64;

thread_local! {
    static COEFF_BUFFER_POOL: RefCell<Vec<(Vec<MonoHead>, Vec<MonomialUnit>)>> =
        RefCell::new(Vec::new());
}

/// Check out a buffer pair from this thread's pool, or a fresh empty pair if
/// none is available. Callers must `reserve` as needed before use.
fn take_pooled_buffers() -> (Vec<MonoHead>, Vec<MonomialUnit>) {
    COEFF_BUFFER_POOL.with(|pool| pool.borrow_mut().pop()).unwrap_or_default()
}

/// Return a no-longer-needed buffer pair to this thread's pool for reuse, or
/// drop it normally if the pool is already at capacity.
fn return_pooled_buffers(mut heads: Vec<MonoHead>, mut factors: Vec<MonomialUnit>) {
    heads.clear();
    factors.clear();
    COEFF_BUFFER_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < BUFFER_POOL_CAP {
            pool.push((heads, factors));
        }
    });
}

/// A sum of monomials `scalar * product(trig factors)`: a symbolic
/// coefficient accumulated during surrogate propagation.
///
/// Stored in CSR/SoA form — one header per monomial plus a single shared
/// factor arena — instead of one owning object per monomial. At the design
/// scale (hundreds of millions of monomials) the per-monomial representation
/// was the dominant cost: every clone/grow/merge did one allocator
/// round-trip *per monomial*, and every traversal chased a heap pointer per
/// monomial. Here every operation is a streaming pass over two flat buffers
/// with at most one buffer rebuild per call, and a monomial costs 16 bytes
/// of header plus 8 bytes per *distinct parameter it touches* (not per
/// touch): repeat gates on an already-touched parameter accumulate into that
/// parameter's existing slot instead of growing the arena.
///
/// Each monomial's factor run is always kept sorted by parameter index
/// (canonicalized at every construction site), so two monomials with the
/// same factor content compare equal regardless of the order gates touched
/// them in, with no separate canonicalization step before merging. Unlike
/// the tally-mark design this replaced, a parameter may appear in at most
/// one `MonomialUnit` per monomial — see `MonomialUnit`'s own docs.
///
/// `add_assign` simply appends monomials (and flags the coefficient dirty);
/// call `deduplicate` to merge identical factor patterns and drop near-zero
/// terms before evaluation.
#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    heads: Vec<MonoHead>,
    factors: Vec<MonomialUnit>,
    /// Whether identical factor patterns may exist across monomials. Only
    /// `add_assign` can introduce duplicates (rotations preserve pairwise
    /// distinctness, scaling doesn't touch factors), so `deduplicate` skips
    /// clean coefficients entirely — the common case at a flush, where most
    /// live terms received no inbox merges since the last one.
    dirty: bool,
}

impl SymbolicCoeff {
    /// Single scalar monomial with no trig factors (used to seed from observable).
    pub fn from_scalar(c: f64) -> Self {
        SymbolicCoeff {
            heads: vec![MonoHead { scalar: c, end: 0 }],
            factors: Vec::new(),
            dirty: false,
        }
    }

    /// Start offset of monomial `i`'s factor run.
    #[inline]
    fn start(&self, i: usize) -> usize {
        if i == 0 { 0 } else { self.heads[i - 1].end as usize }
    }

    /// Factor run of monomial `i`.
    #[inline]
    fn factor_run(&self, i: usize) -> &[MonomialUnit] {
        &self.factors[self.start(i)..self.heads[i].end as usize]
    }

    pub fn monomial_count(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Iterate `(scalar, sorted factor run)` per monomial, in storage order.
    pub fn iter_monomials(&self) -> impl Iterator<Item = (f64, &[MonomialUnit])> + '_ {
        let mut start = 0usize;
        self.heads.iter().map(move |h| {
            let end = h.end as usize;
            let run = &self.factors[start..end];
            start = end;
            (h.scalar, run)
        })
    }

    /// Append one monomial. `factors` must already be in sorted (canonical)
    /// order, with each distinct parameter appearing at most once — this is
    /// the deserialization/test construction entry point, and save writes
    /// runs in canonical order, so no re-sort/re-merge happens here.
    pub fn push_monomial(&mut self, scalar: f64, factors: &[MonomialUnit]) {
        debug_assert!(
            factors.windows(2).all(|w| w[0].param() < w[1].param()),
            "factor run must be sorted by strictly increasing parameter index, \
             with each parameter appearing in at most one unit"
        );
        self.factors.extend_from_slice(factors);
        self.heads.push(MonoHead { scalar, end: self.factors.len() as u64 });
    }

    /// Reserve for `n_monomials` headers and `n_factors` arena slots.
    pub fn reserve(&mut self, n_monomials: usize, n_factors: usize) {
        self.heads.reserve(n_monomials);
        self.factors.reserve(n_factors);
    }

    /// Drop monomials with frequency (summed trig-factor degree) > max_freq,
    /// compacting the arena in place (no allocation).
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.compact_by_len(|freq| freq <= max_freq);
    }

    /// Drop every monomial whose frequency equals exactly `freq`, in place.
    pub fn remove_at_frequency(&mut self, freq: usize) {
        self.compact_by_len(|freq_i| freq_i != freq);
    }

    /// In-place compaction keeping monomials for which `keep(frequency)`
    /// holds, where frequency is the *summed* `cos_exp + sin_exp` across a
    /// monomial's whole run (not its raw arena slot count — one slot can
    /// represent more than one trig factor, unlike the old tally-mark
    /// design). Writes never overtake reads (removal only shrinks), so both
    /// buffers are rewritten in one forward pass with zero allocation.
    fn compact_by_len(&mut self, mut keep: impl FnMut(usize) -> bool) {
        let mut w_head = 0usize;
        let mut w_fac = 0usize;
        let mut start = 0usize;
        for i in 0..self.heads.len() {
            let end = self.heads[i].end as usize;
            let freq: usize = self.factors[start..end].iter().map(|u| u.frequency()).sum();
            if keep(freq) {
                let len = end - start;
                if w_fac != start {
                    self.factors.copy_within(start..end, w_fac);
                }
                w_fac += len;
                self.heads[w_head] = MonoHead { scalar: self.heads[i].scalar, end: w_fac as u64 };
                w_head += 1;
            }
            start = end;
        }
        self.heads.truncate(w_head);
        self.factors.truncate(w_fac);
    }

    /// Merge monomials with identical factor patterns and drop near-zero
    /// results. Skips clean coefficients outright (see the `dirty` field
    /// docs); a consequence is that near-zero scalars are only pruned on
    /// dirty coefficients — which matches where they can arise, since
    /// destructive cancellation requires a merge in the first place.
    ///
    /// Below `HASH_MERGE_THRESHOLD` monomials, sorts an index permutation
    /// (comparing arena slices) and merges adjacent equal runs (`O(k log k)`,
    /// cheap for small `k`). Above it, accumulates scalars in a hashmap keyed
    /// by *borrowed* arena slices (`O(k)` amortized, no key data moved) —
    /// avoids the comparison sort that dominates a flush when one term's
    /// coefficient has ballooned to a large `k`. Either path rebuilds the two
    /// flat buffers once; there is no per-monomial allocation. Native `Ord`/
    /// `Hash` on `MonomialUnit` (a plain packed `u64`) means slice comparison
    /// and hashing here need no per-element unpacking.
    pub fn deduplicate(&mut self) {
        if !self.dirty || self.heads.len() <= 1 {
            self.dirty = false;
            return;
        }
        self.dirty = false;

        if self.heads.len() < HASH_MERGE_THRESHOLD {
            // Sort path guard: u32 indices are always sufficient below the
            // threshold (100k << u32::MAX).
            let mut order: Vec<u32> = (0..self.heads.len() as u32).collect();
            order.sort_unstable_by(|&a, &b| self.factor_run(a as usize).cmp(self.factor_run(b as usize)));

            let (mut heads, mut factors) = take_pooled_buffers();
            heads.reserve(self.heads.len());
            factors.reserve(self.factors.len());
            let mut i = 0usize;
            while i < order.len() {
                let run = self.factor_run(order[i] as usize);
                let mut scalar = self.heads[order[i] as usize].scalar;
                let mut j = i + 1;
                while j < order.len() && self.factor_run(order[j] as usize) == run {
                    scalar += self.heads[order[j] as usize].scalar;
                    j += 1;
                }
                if scalar.abs() > 1e-15 {
                    factors.extend_from_slice(run);
                    heads.push(MonoHead { scalar, end: factors.len() as u64 });
                }
                i = j;
            }
            let old_heads = std::mem::replace(&mut self.heads, heads);
            let old_factors = std::mem::replace(&mut self.factors, factors);
            return_pooled_buffers(old_heads, old_factors);
            return;
        }

        let mut acc: FxHashMap<&[MonomialUnit], f64> = FxHashMap::default();
        acc.reserve(self.heads.len());
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            *acc.entry(&self.factors[start..end]).or_insert(0.0) += h.scalar;
            start = end;
        }
        let (mut heads, mut factors) = take_pooled_buffers();
        heads.reserve(acc.len());
        factors.reserve(self.factors.len());
        for (run, scalar) in acc {
            if scalar.abs() > 1e-15 {
                factors.extend_from_slice(run);
                heads.push(MonoHead { scalar, end: factors.len() as u64 });
            }
        }
        let old_heads = std::mem::replace(&mut self.heads, heads);
        let old_factors = std::mem::replace(&mut self.factors, factors);
        return_pooled_buffers(old_heads, old_factors);
    }

    /// Evaluate against a flat lookup table indexed by `2 * param_index`
    /// (`cos`) / `2 * param_index + 1` (`sin`): up to two LUT gathers plus a
    /// `powi` per arena slot (one slot can hold `cos^a * sin^b`, unlike the
    /// old tally-mark design's one branch-free gather per factor).
    ///
    /// `SurrogateModel::evaluate` already parallelizes across terms, which
    /// covers the common case; but a handful of terms can carry the
    /// overwhelming majority of monomials (same skew `deduplicate` accounts
    /// for), leaving other threads idle while one thread churns through a
    /// huge single-term monomial list serially. `with_min_len` lets rayon's
    /// splitter fall back to a single sequential chunk for ordinary
    /// (small) terms — avoiding per-call parallel overhead there — while
    /// still splitting (and letting idle threads steal work via the outer
    /// per-term `par_iter`) once a term's monomial count is large enough
    /// to be worth it.
    pub fn evaluate(&self, lut: &[f64]) -> f64 {
        const EVALUATE_PAR_MIN_LEN: usize = 4096;
        let heads = &self.heads;
        let factors = &self.factors;
        (0..heads.len())
            .into_par_iter()
            .with_min_len(EVALUATE_PAR_MIN_LEN)
            .map(|i| {
                let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                let mut prod = heads[i].scalar;
                for u in &factors[start..heads[i].end as usize] {
                    let base = 2 * u.param() as usize;
                    let cos_exp = u.cos_exp();
                    let sin_exp = u.sin_exp();
                    if cos_exp > 0 {
                        prod *= lut[base].powi(cos_exp as i32);
                    }
                    if sin_exp > 0 {
                        prod *= lut[base + 1].powi(sin_exp as i32);
                    }
                }
                prod
            })
            .sum()
    }

    /// Highest frequency present and how many monomials sit at exactly that
    /// frequency; `(0, 0)` if empty. Frequency is the summed trig-factor
    /// degree across a monomial's run (see `compact_by_len`), not raw slot
    /// count. Parallel over monomial chunks (same skew rationale as
    /// `evaluate`) so one giant coefficient doesn't serialize the truncation
    /// pass that calls this per live term.
    pub fn top_frequency_and_count(&self) -> (usize, usize) {
        const PAR_MIN_LEN: usize = 65_536;
        let heads = &self.heads;
        let factors = &self.factors;
        (0..heads.len())
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || (0usize, 0usize),
                |(mut freq, mut count), i| {
                    let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                    let end = heads[i].end as usize;
                    let len: usize = factors[start..end].iter().map(|u| u.frequency()).sum();
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

    /// Remove monomials whose frequency (summed trig-factor degree) equals
    /// exactly `freq`, claiming removals from a `remaining` budget shared
    /// across every coefficient processed in the same pass (see
    /// `apply_truncation_policy`'s monomial-range second stage: only the
    /// single highest observed frequency is ever targeted, clamped to not
    /// remove more than needed to reach `monomial_range`'s floor). Returns
    /// how many were removed.
    ///
    /// Counts this coefficient's own hits first (no synchronization needed
    /// for that — it's a local, read-only scan), then claims
    /// `min(hits, remaining)` in a single compare-exchange loop: one atomic
    /// operation per coefficient that actually has a hit, not per monomial.
    /// Given the known skew (a handful of terms can carry the overwhelming
    /// majority of monomials), the atomic is touched by comparatively few
    /// coefficients even when the total removal count is large.
    pub fn remove_at_frequency_budgeted(&mut self, freq: usize, remaining: &AtomicUsize) -> usize {
        let mut hits = 0usize;
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            let f: usize = self.factors[start..end].iter().map(|u| u.frequency()).sum();
            if f == freq {
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
        self.compact_by_len(|freq_i| {
            if freq_i == freq && removed < claim {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
    }
}

impl CoeffRepr for SymbolicCoeff {
    /// Gate parameter is a parameter index (u32).
    type GateParam = u32;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        // Seed observables are Hermitian, so their Pauli/Majorana-basis
        // coefficients are real; see the `MonoHead::scalar` doc comment.
        debug_assert!(c.im.abs() < 1e-9, "surrogate seed coefficient must be real: {c:?}");
        SymbolicCoeff::from_scalar(c.re)
    }

    fn add_assign(&mut self, mut other: Self) {
        if self.heads.is_empty() {
            // Common case at a flush: a term newly inserted into the map gets
            // `or_default().add_assign(coeff)` — take the buffers instead of
            // copying every monomial into a fresh allocation.
            *self = other;
            return;
        }
        if other.heads.is_empty() {
            return;
        }
        let base = self.factors.len() as u64;
        self.factors.append(&mut other.factors);
        self.heads.reserve(other.heads.len());
        self.heads.extend(other.heads.iter().map(|h| MonoHead { scalar: h.scalar, end: h.end + base }));
        self.dirty = true;
    }

    /// The sin branch is one forward streaming pass over the arena into a
    /// fresh coefficient (two buffer allocations — it's genuinely new data);
    /// the cos branch then mutates `self`'s own arena in place.
    ///
    /// Unlike the old tally-mark design (where every monomial grew by
    /// exactly one slot per gate, letting the backward shift use a fixed
    /// `end += i + 1` closed form), a monomial only grows here if `idx` is
    /// new to it — an already-present parameter just gets its exponent
    /// bumped in its existing slot, with zero arena growth. So growth is now
    /// per-monomial-dependent: a single forward pass classifies every
    /// monomial's touch on `idx` as a "hit" (`Ok(pos)`, in-place bump) or a
    /// "miss" (`Err(pos)`, one new slot) *once*, and that same classification
    /// drives both the sin branch (built immediately below) and the cos
    /// branch's backward shift (via `prefix_misses`, the running count of
    /// misses through each monomial) — one binary search per monomial per
    /// gate, not one per branch.
    ///
    /// The in-place cos shift matters at scale: this runs per anticommuting
    /// (generator, term) pair inside a serial-per-term rayon task, and for a
    /// giant coefficient a rebuild into a fresh arena puts multi-MB
    /// allocation plus first-touch page faults on that serial critical path
    /// every gate. Shifting within the existing buffer keeps its pages warm
    /// and its capacity across gates (growth is `reserve`-amortized), so the
    /// per-gate cost for an already-touched parameter is one memmove of
    /// already-resident data, and for a hit specifically, no growth at all.
    fn apply_rotation(&mut self, idx: &u32, phase: Complex64) -> Self {
        // sin branch scalar: * (i * phase). `phase` is always ±i here (this
        // is only called on anticommuting generator/term pairs), so `i *
        // phase` is always real — see the `MonoHead::scalar` doc comment.
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let n = self.heads.len();

        // Classify every monomial's touch on `idx` once, up front. `Ok(pos)`
        // means this monomial already has a unit for `idx` (a hit — bump in
        // place, no growth); `Err(pos)` means `idx` is new to this monomial
        // (a miss — one new slot at `pos`).
        let mut results: Vec<Result<usize, usize>> = Vec::with_capacity(n);
        // Prefix sum of misses through and including monomial `i`: tells the
        // cos-branch backward pass below exactly how far monomial `i`'s data
        // has shifted right, since growth is no longer uniform across
        // monomials.
        let mut prefix_misses: Vec<usize> = Vec::with_capacity(n);

        // Sin branch first, while the arena is still un-shifted. Buffers
        // come from this thread's pool instead of a fresh allocation — see
        // `take_pooled_buffers`.
        let (mut sin_heads, mut sin_factors) = take_pooled_buffers();
        sin_heads.reserve(n);
        sin_factors.reserve(self.factors.len() + n);

        let mut start = 0usize;
        let mut misses_so_far = 0usize;
        for head in &self.heads {
            let end = head.end as usize;
            let run = &self.factors[start..end];
            let result = run.binary_search_by_key(idx, |u| u.param());
            match result {
                Ok(pos) => {
                    sin_factors.extend_from_slice(&run[..pos]);
                    sin_factors.push(run[pos].inc_sin());
                    sin_factors.extend_from_slice(&run[pos + 1..]);
                }
                Err(pos) => {
                    misses_so_far += 1;
                    sin_factors.extend_from_slice(&run[..pos]);
                    sin_factors.push(MonomialUnit::new(*idx, 0, 1));
                    sin_factors.extend_from_slice(&run[pos..]);
                }
            }
            sin_heads.push(MonoHead { scalar: head.scalar * branch_phase, end: sin_factors.len() as u64 });
            results.push(result);
            prefix_misses.push(misses_so_far);
            start = end;
        }
        let total_misses = misses_so_far;

        // Cos branch: back-to-front in-place shift, sized to exactly how
        // many monomials gained a new parameter — not a uniform +1/monomial.
        // On a miss, suffix is moved before prefix so no source bytes are
        // overwritten before they're read; `copy_within` handles the
        // overlapping ranges either way. The fill value of `resize` is
        // arbitrary — every new slot is written below, either by a copy or a
        // direct insert.
        let old_len = self.factors.len();
        self.factors.resize(old_len + total_misses, MonomialUnit::new(*idx, 0, 0));
        for i in (0..n).rev() {
            let old_start = if i == 0 { 0 } else { self.heads[i - 1].end as usize };
            let old_end = self.heads[i].end as usize;
            let shift_before = if i == 0 { 0 } else { prefix_misses[i - 1] };
            let new_start = old_start + shift_before;
            match results[i] {
                Ok(pos) => {
                    let new_end = old_end + shift_before;
                    self.factors.copy_within(old_start..old_end, new_start);
                    self.factors[new_start + pos] = self.factors[new_start + pos].inc_cos();
                    self.heads[i].end = new_end as u64;
                }
                Err(pos) => {
                    let new_end = old_end + shift_before + 1;
                    self.factors.copy_within(old_start + pos..old_end, new_start + pos + 1);
                    if pos > 0 {
                        self.factors.copy_within(old_start..old_start + pos, new_start);
                    }
                    self.factors[new_start + pos] = MonomialUnit::new(*idx, 1, 0);
                    self.heads[i].end = new_end as u64;
                }
            }
        }

        // Duplicates in self (if any) are duplicated into the branch too.
        SymbolicCoeff { heads: sin_heads, factors: sin_factors, dirty: self.dirty }
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        for h in &mut self.heads {
            h.scalar *= factor;
        }
    }

    /// L1 norm is undefined for symbolic; return 0 to skip coeff-based truncation.
    #[inline]
    fn l1_norm(&self) -> f64 {
        0.0
    }

    /// Monomial count is what actually drives memory/CPU cost for symbolic
    /// coefficients, unlike raw term count — see `flush_and_maybe_truncate`'s
    /// monomial-count flush trigger.
    #[inline]
    fn size_hint(&self) -> usize {
        self.heads.len()
    }

    #[inline]
    fn prefetch_read(&self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch has no memory effects; the pointers are this
        // coefficient's own live buffers.
        unsafe {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            // Cover the whole (typically small) buffers, not just their
            // first line — the merge copies all of both.
            let ptr = self.factors.as_ptr() as *const i8;
            let bytes = self.factors.len() * std::mem::size_of::<MonomialUnit>();
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
            let ptr = self.heads.as_ptr() as *const i8;
            let bytes = self.heads.len() * std::mem::size_of::<MonoHead>();
            let mut off = 0usize;
            while off < bytes {
                _mm_prefetch(ptr.add(off), _MM_HINT_T0);
                off += 64;
            }
        }
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
        obj.getattr("param_index")?.extract()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a coefficient from raw `(scalar, [(param_index, is_sin)])`
    /// monomials — mirroring the old tally-mark occurrence list. Repeated or
    /// mixed occurrences of the same parameter within one monomial are
    /// merged into a single `MonomialUnit`'s cos/sin exponents (exactly what
    /// real propagation does via `apply_rotation`), not left as separate
    /// slots; the result is flagged dirty like a real post-merge coefficient.
    fn coeff(monomials: &[(f64, &[(u32, bool)])]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, occurrences) in monomials {
            let mut exps: std::collections::BTreeMap<u32, (u16, u16)> = std::collections::BTreeMap::new();
            for &(idx, is_sin) in occurrences {
                let entry = exps.entry(idx).or_insert((0, 0));
                if is_sin { entry.1 += 1; } else { entry.0 += 1; }
            }
            let run: Vec<MonomialUnit> = exps
                .into_iter()
                .map(|(idx, (cos_exp, sin_exp))| MonomialUnit::new(idx, cos_exp, sin_exp))
                .collect();
            c.push_monomial(scalar, &run);
        }
        c.dirty = true;
        c
    }

    /// Coefficient of monomials with exactly the given `(scalar, frequency,
    /// tag)` specs, each monomial's factors made unique by its tag — one
    /// distinct parameter per unit of frequency (`cos_exp = 1` each), so
    /// summed frequency equals the requested `freq` directly.
    fn coeff_with_freqs(specs: &[(f64, usize, u32)]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, freq, tag) in specs {
            let run: Vec<MonomialUnit> = (0..freq).map(|p| MonomialUnit::new(tag * 1000 + p as u32, 1, 0)).collect();
            c.push_monomial(scalar, &run);
        }
        c
    }

    /// Reference evaluation independent of `evaluate`'s parallel path.
    fn naive_evaluate(c: &SymbolicCoeff, lut: &[f64]) -> f64 {
        c.iter_monomials()
            .map(|(scalar, run)| {
                scalar
                    * run
                        .iter()
                        .map(|u| {
                            let base = 2 * u.param() as usize;
                            lut[base].powi(u.cos_exp() as i32) * lut[base + 1].powi(u.sin_exp() as i32)
                        })
                        .product::<f64>()
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
    fn push_and_iter_round_trip() {
        let c = coeff(&[
            (1.5, &[(1, false), (0, true)]),
            (-2.0, &[(3, false)]),
            (0.5, &[]),
        ]);
        let collected: Vec<(f64, Vec<MonomialUnit>)> =
            c.iter_monomials().map(|(s, run)| (s, run.to_vec())).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 1.5);
        assert_eq!(collected[0].1, vec![MonomialUnit::new(0, 0, 1), MonomialUnit::new(1, 1, 0)]);
        assert_eq!(collected[1].1, vec![MonomialUnit::new(3, 1, 0)]);
        assert!(collected[2].1.is_empty());
        assert_eq!(c.monomial_count(), 3);
    }

    #[test]
    fn apply_rotation_matches_trig_identity_and_keeps_runs_sorted() {
        let lut = make_lut(8);
        let mut c = SymbolicCoeff::from_scalar(0.75);
        // Descending parameter indices force real insertion (not just
        // append); index 2 repeats, exercising the in-place exponent-bump
        // (hit) path alongside fresh-parameter (miss) inserts.
        for idx in [5u32, 2, 7, 2, 0] {
            let before = naive_evaluate(&c, &lut);
            let sin_branch = c.apply_rotation(&idx, Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(idx << 1) as usize], lut[((idx << 1) | 1) as usize]);
            assert!((naive_evaluate(&c, &lut) - cos_t * before).abs() < 1e-12);
            // branch_phase = (i * -i).re = 1.0
            assert!((naive_evaluate(&sin_branch, &lut) - sin_t * before).abs() < 1e-12);
            for (_, run) in c.iter_monomials().chain(sin_branch.iter_monomials()) {
                assert!(
                    run.windows(2).all(|w| w[0].param() < w[1].param()),
                    "factor run must stay sorted by strictly increasing parameter"
                );
            }
        }
    }

    #[test]
    fn apply_rotation_stacks_exponents_on_repeated_parameter_without_growing_slots() {
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let _ = c.apply_rotation(&3u32, Complex64::new(0.0, -1.0));
        assert_eq!(c.monomial_count(), 1);
        let (_, run) = c.iter_monomials().next().unwrap();
        assert_eq!(run, &[MonomialUnit::new(3, 1, 0)]);

        // Touching parameter 3 again must bump the existing unit's cos
        // exponent in place, not add a second slot.
        let _ = c.apply_rotation(&3u32, Complex64::new(0.0, -1.0));
        let (_, run) = c.iter_monomials().next().unwrap();
        assert_eq!(run, &[MonomialUnit::new(3, 2, 0)]);
    }

    #[test]
    fn apply_rotation_can_produce_mixed_cos_and_sin_on_same_parameter() {
        let mut c = SymbolicCoeff::from_scalar(1.0);
        // First touch on param 3 goes to the sin branch (a brand-new term).
        let mut sin_branch = c.apply_rotation(&3u32, Complex64::new(0.0, -1.0));
        {
            let (_, run) = sin_branch.iter_monomials().next().unwrap();
            assert_eq!(run, &[MonomialUnit::new(3, 0, 1)]);
        }
        // Touching the SAME parameter again on sin_branch's cos branch
        // (in-place mutation) must land in the *same* slot, combining a
        // nonzero sin_exp (from before) with a nonzero cos_exp (from this
        // touch) — structurally impossible under the old tally-mark design,
        // since two different `TrigFactor` tokens (cos(3), sin(3)) would
        // just coexist as separate arena entries instead.
        let _ = sin_branch.apply_rotation(&3u32, Complex64::new(0.0, -1.0));
        let (_, run) = sin_branch.iter_monomials().next().unwrap();
        assert_eq!(run.len(), 1, "same-parameter touches must collapse into one arena slot");
        assert_eq!(run[0].param(), 3);
        assert_eq!(run[0].cos_exp(), 1);
        assert_eq!(run[0].sin_exp(), 1);
    }

    #[test]
    fn apply_rotation_matches_trig_identity_with_heavy_parameter_reuse() {
        // Cycles through a small pool of indices repeatedly, unlike every
        // other test/benchmark in this crate (which only ever uses fresh
        // indices) — this is the regime `MonomialUnit` exists for, and the
        // one the delicate variable-length `apply_rotation` shift must get
        // right.
        let n_params = 6usize;
        let lut = make_lut(n_params);
        let mut c = SymbolicCoeff::from_scalar(1.0);
        let mut state = 0xA5A5_A5A5_A5A5_A5A5u64;
        for _ in 0..200 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let idx = (state % n_params as u64) as u32;
            let before = naive_evaluate(&c, &lut);
            let sin_branch = c.apply_rotation(&idx, Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(2 * idx) as usize], lut[(2 * idx + 1) as usize]);
            let scale = before.abs().max(1.0);
            assert!((naive_evaluate(&c, &lut) - cos_t * before).abs() < 1e-9 * scale);
            assert!((naive_evaluate(&sin_branch, &lut) - sin_t * before).abs() < 1e-9 * scale);
            for (_, run) in c.iter_monomials() {
                assert!(run.windows(2).all(|w| w[0].param() < w[1].param()));
            }
        }
    }

    #[test]
    #[should_panic(expected = "cos exponent overflow")]
    fn inc_cos_panics_on_overflow_instead_of_corrupting_param() {
        let unit = MonomialUnit::new(7, u16::MAX, 0);
        let _ = unit.inc_cos();
    }

    #[test]
    #[should_panic(expected = "sin exponent overflow")]
    fn inc_sin_panics_on_overflow_instead_of_corrupting_param() {
        let unit = MonomialUnit::new(7, 0, u16::MAX);
        let _ = unit.inc_sin();
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
        let runs: Vec<Vec<MonomialUnit>> = a.iter_monomials().map(|(_, r)| r.to_vec()).collect();
        assert_eq!(runs[1], vec![MonomialUnit::new(2, 0, 1)]);
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
        // Same duplicated content, but flagged clean: deduplicate must be a
        // no-op, because only add_assign can introduce duplicates in real use.
        let mut c = coeff(&[(1.0, &[(0, false)]), (2.0, &[(0, false)])]);
        c.dirty = false;
        c.deduplicate();
        assert_eq!(c.monomial_count(), 2);

        // After a real merge it runs.
        let mut a = coeff(&[(1.0, &[(0, false)])]);
        a.dirty = false;
        let mut b = coeff(&[(2.0, &[(0, false)])]);
        b.dirty = false;
        a.add_assign(b);
        a.deduplicate();
        assert_eq!(a.monomial_count(), 1);
        assert!(!a.dirty);
    }

    #[test]
    fn hash_path_matches_naive_evaluation() {
        let n_params = 400;
        let lut = make_lut(n_params);

        // > HASH_MERGE_THRESHOLD monomials with many repeated factor patterns,
        // inserted in varying order, to exercise the hash-merge path and its
        // order-independence. i != j always, so each monomial's two units
        // are on distinct parameters (no same-parameter merge needed here).
        let mut c = SymbolicCoeff::default();
        for rep in 0..3usize {
            for i in 0..n_params {
                for j in 0..n_params {
                    if i == j {
                        continue;
                    }
                    let mut run = [MonomialUnit::new(i as u32, 1, 0), MonomialUnit::new(j as u32, 0, 1)];
                    run.sort_unstable();
                    c.push_monomial(0.1 * (rep as f64 + 1.0), &run);
                }
            }
        }
        c.dirty = true;
        assert!(c.monomial_count() >= HASH_MERGE_THRESHOLD, "test setup should exercise the hash path");

        let expected = naive_evaluate(&c, &lut);
        c.deduplicate();
        let actual = c.evaluate(&lut);
        assert!(
            (actual - expected).abs() < 1e-9,
            "hash-merge path changed the evaluated value: {actual} vs {expected}"
        );
    }

    #[test]
    fn small_and_large_paths_agree() {
        // Same logical multiset, once below and once above HASH_MERGE_THRESHOLD
        // (padded with exactly-cancelling pairs), should evaluate identically.
        let lut = make_lut(8);

        let base: &[(f64, &[(u32, bool)])] = &[
            (1.0, &[(0, false), (1, true)]),
            (2.0, &[(1, true), (0, false)]),
            (-0.5, &[(2, false)]),
        ];
        let mut small = coeff(base);
        let expected = naive_evaluate(&small, &lut);
        small.deduplicate();
        assert!(small.monomial_count() < HASH_MERGE_THRESHOLD);
        assert!((small.evaluate(&lut) - expected).abs() < 1e-12);

        let mut large = coeff(base);
        // Fixed param 3 plus a varying param drawn from a disjoint range
        // (4..8, still within `lut`'s 8-parameter range), so it never
        // collides with 3 and needs no same-parameter merge logic here.
        for k in 0..HASH_MERGE_THRESHOLD as u32 {
            let mut run = [MonomialUnit::new(3, 1, 0), MonomialUnit::new(4 + (k % 4), 0, 1)];
            run.sort_unstable();
            large.push_monomial(5.0, &run);
            large.push_monomial(-5.0, &run);
        }
        assert!(large.monomial_count() >= HASH_MERGE_THRESHOLD);
        assert!((naive_evaluate(&large, &lut) - expected).abs() < 1e-9);
        large.deduplicate();
        assert!((large.evaluate(&lut) - expected).abs() < 1e-9);
    }

    #[test]
    fn trim_high_frequency_compacts_in_place() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 1, 1), (3.0, 4, 2), (4.0, 2, 3)]);
        c.trim_high_frequency(2);
        assert_eq!(c.monomial_count(), 2);
        let kept: Vec<(f64, usize)> = c.iter_monomials().map(|(s, r)| (s, r.len())).collect();
        assert_eq!(kept, vec![(2.0, 1), (4.0, 2)]);
        assert_eq!(c.factors.len(), 3);
    }

    #[test]
    fn top_frequency_and_count_finds_top_bucket_only() {
        let c = coeff_with_freqs(&[(1.0, 3, 0), (1.0, 5, 1), (1.0, 5, 2), (1.0, 2, 3)]);
        assert_eq!(c.top_frequency_and_count(), (5, 2));
        assert_eq!(SymbolicCoeff::default().top_frequency_and_count(), (0, 0));
    }

    #[test]
    fn top_frequency_counts_summed_exponents_not_slot_count() {
        // One slot with cos_exp=4 has frequency 4, same as four distinct
        // single-exponent slots -- this is the case no arena-length-based
        // implementation could get right, since here slot count (1) and
        // frequency (4) diverge.
        let mut c = SymbolicCoeff::default();
        c.push_monomial(1.0, &[MonomialUnit::new(0, 4, 0)]);
        c.push_monomial(2.0, &[MonomialUnit::new(1, 1, 0), MonomialUnit::new(2, 1, 0)]);
        assert_eq!(c.top_frequency_and_count(), (4, 1));
    }

    #[test]
    fn budget_covers_all_hits_removes_all_and_claims_exactly_hits() {
        let mut c = coeff_with_freqs(&[(1.0, 3, 0), (2.0, 3, 1), (3.0, 1, 2)]);
        let remaining = AtomicUsize::new(10);
        let removed = c.remove_at_frequency_budgeted(3, &remaining);
        assert_eq!(removed, 2);
        assert_eq!(c.monomial_count(), 1);
        assert_eq!(c.iter_monomials().next().unwrap().1.len(), 1);
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
        let lens: Vec<usize> = c.iter_monomials().map(|(_, r)| r.len()).collect();
        assert_eq!(lens, vec![3, 1]);
    }

    #[test]
    fn evaluate_parallel_matches_naive_at_scale() {
        let n_params = 64;
        let lut = make_lut(n_params);
        let mut c = SymbolicCoeff::default();
        let mut state = 0x9E3779B97F4A7C15u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let len = (state % 6) as usize;
            // Draw `len` random (param, is_sin) occurrences and merge
            // same-parameter hits into one unit's exponents, exactly like
            // real propagation does -- a random draw can plausibly repeat a
            // param within one monomial, which must collapse into a single
            // slot rather than appear as duplicate entries.
            let mut exps: std::collections::BTreeMap<u32, (u16, u16)> = std::collections::BTreeMap::new();
            for k in 0..len {
                let v = (state >> (8 * k)) as u32 % (2 * n_params as u32);
                let (idx, is_sin) = (v >> 1, v & 1 == 1);
                let entry = exps.entry(idx).or_insert((0, 0));
                if is_sin { entry.1 += 1; } else { entry.0 += 1; }
            }
            let run: Vec<MonomialUnit> = exps
                .into_iter()
                .map(|(idx, (cos_exp, sin_exp))| MonomialUnit::new(idx, cos_exp, sin_exp))
                .collect();
            c.push_monomial(((state % 1000) as f64 - 500.0) / 250.0, &run);
        }
        let expected = naive_evaluate(&c, &lut);
        assert!((c.evaluate(&lut) - expected).abs() < 1e-9 * expected.abs().max(1.0));
    }
}
