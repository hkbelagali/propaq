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

/// Packed trig factor: bit 0 = is_sin, bits 1–31 = param_index.
/// Supports up to 2^31 ≈ 2 billion distinct parameters.
///
/// The packing doubles as the evaluation LUT index: a flat table with
/// `cos(theta_i)` at slot `2i` and `sin(theta_i)` at slot `2i + 1` can be
/// indexed directly by the packed value, with no is_sin branch per factor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TrigFactor(pub u32);

impl TrigFactor {
    #[inline]
    pub fn cos(idx: u32) -> Self {
        TrigFactor(idx << 1)
    }

    #[inline]
    pub fn sin(idx: u32) -> Self {
        TrigFactor((idx << 1) | 1)
    }

    #[inline]
    pub fn param_index(self) -> u32 {
        self.0 >> 1
    }

    #[inline]
    pub fn is_sin(self) -> bool {
        self.0 & 1 == 1
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

/// A sum of monomials `scalar * product(trig factors)`: a symbolic
/// coefficient accumulated during surrogate propagation.
///
/// Stored in CSR/SoA form — one header per monomial plus a single shared
/// factor arena — instead of one owning object per monomial. At the design
/// scale (hundreds of millions of monomials) the per-monomial representation
/// was the dominant cost: every clone/grow/merge did one allocator
/// round-trip *per monomial*, and every traversal chased a heap pointer per
/// monomial. Here every operation is a streaming pass over two flat buffers
/// with at most one buffer rebuild per call, and a monomial costs
/// `16 + 4 * frequency` bytes instead of an 80-byte struct plus (for spilled
/// factor lists) a pooled heap buffer.
///
/// Each monomial's factor run is always kept sorted (canonicalized at every
/// construction site), so two monomials with the same factor content compare
/// equal regardless of the order gates touched them in, with no separate
/// canonicalization step before merging. Duplicate factors (the same
/// parameter touched more than once) are kept, not merged: `cos(idx)`
/// appearing twice represents `cos(theta)^2`, a distinct factor pattern.
///
/// `add_assign` simply appends monomials (and flags the coefficient dirty);
/// call `deduplicate` to merge identical factor patterns and drop near-zero
/// terms before evaluation.
#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    heads: Vec<MonoHead>,
    factors: Vec<TrigFactor>,
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
    fn factor_run(&self, i: usize) -> &[TrigFactor] {
        &self.factors[self.start(i)..self.heads[i].end as usize]
    }

    pub fn monomial_count(&self) -> usize {
        self.heads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// Iterate `(scalar, sorted factor run)` per monomial, in storage order.
    pub fn iter_monomials(&self) -> impl Iterator<Item = (f64, &[TrigFactor])> + '_ {
        let mut start = 0usize;
        self.heads.iter().map(move |h| {
            let end = h.end as usize;
            let run = &self.factors[start..end];
            start = end;
            (h.scalar, run)
        })
    }

    /// Append one monomial. `factors` must already be in sorted (canonical)
    /// order — this is the deserialization/test construction entry point, and
    /// save writes runs in canonical order, so no re-sort happens here.
    pub fn push_monomial(&mut self, scalar: f64, factors: &[TrigFactor]) {
        debug_assert!(factors.windows(2).all(|w| w[0] <= w[1]), "factor run must be sorted");
        self.factors.extend_from_slice(factors);
        self.heads.push(MonoHead { scalar, end: self.factors.len() as u64 });
    }

    /// Reserve for `n_monomials` headers and `n_factors` arena slots.
    pub fn reserve(&mut self, n_monomials: usize, n_factors: usize) {
        self.heads.reserve(n_monomials);
        self.factors.reserve(n_factors);
    }

    /// Drop monomials with frequency (factor count) > max_freq, compacting
    /// the arena in place (no allocation).
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.compact_by_len(|len| len <= max_freq);
    }

    /// Drop every monomial whose factor count equals exactly `freq`, in place.
    pub fn remove_at_frequency(&mut self, freq: usize) {
        self.compact_by_len(|len| len != freq);
    }

    /// In-place compaction keeping monomials for which `keep(factor count)`
    /// holds. Writes never overtake reads (removal only shrinks), so both
    /// buffers are rewritten in one forward pass with zero allocation.
    fn compact_by_len(&mut self, mut keep: impl FnMut(usize) -> bool) {
        let mut w_head = 0usize;
        let mut w_fac = 0usize;
        let mut start = 0usize;
        for i in 0..self.heads.len() {
            let end = self.heads[i].end as usize;
            let len = end - start;
            if keep(len) {
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
    /// flat buffers once; there is no per-monomial allocation.
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

            let mut heads: Vec<MonoHead> = Vec::with_capacity(self.heads.len());
            let mut factors: Vec<TrigFactor> = Vec::with_capacity(self.factors.len());
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
            self.heads = heads;
            self.factors = factors;
            return;
        }

        let mut acc: FxHashMap<&[TrigFactor], f64> = FxHashMap::default();
        acc.reserve(self.heads.len());
        let mut start = 0usize;
        for h in &self.heads {
            let end = h.end as usize;
            *acc.entry(&self.factors[start..end]).or_insert(0.0) += h.scalar;
            start = end;
        }
        let mut heads: Vec<MonoHead> = Vec::with_capacity(acc.len());
        let mut factors: Vec<TrigFactor> = Vec::with_capacity(self.factors.len());
        for (run, scalar) in acc {
            if scalar.abs() > 1e-15 {
                factors.extend_from_slice(run);
                heads.push(MonoHead { scalar, end: factors.len() as u64 });
            }
        }
        self.heads = heads;
        self.factors = factors;
    }

    /// Evaluate against a flat lookup table indexed by the packed factor
    /// value: `lut[2i] = cos(theta_i)`, `lut[2i + 1] = sin(theta_i)`
    /// (`2 * n_params` entries) — one branch-free gather per factor over a
    /// contiguous arena scan.
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
                for f in &factors[start..heads[i].end as usize] {
                    prod *= lut[f.0 as usize];
                }
                prod
            })
            .sum()
    }

    /// Highest frequency present and how many monomials sit at exactly that
    /// frequency; `(0, 0)` if empty. Parallel over monomial chunks (same
    /// skew rationale as `evaluate`) so one giant coefficient doesn't
    /// serialize the truncation pass that calls this per live term.
    pub fn top_frequency_and_count(&self) -> (usize, usize) {
        const PAR_MIN_LEN: usize = 65_536;
        let heads = &self.heads;
        (0..heads.len())
            .into_par_iter()
            .with_min_len(PAR_MIN_LEN)
            .fold(
                || (0usize, 0usize),
                |(mut freq, mut count), i| {
                    let start = if i == 0 { 0 } else { heads[i - 1].end as usize };
                    let len = heads[i].end as usize - start;
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

    /// Remove monomials whose factor count equals exactly `freq`, claiming
    /// removals from a `remaining` budget shared across every coefficient
    /// processed in the same pass (see `apply_truncation_policy`'s
    /// monomial-range second stage: only the single highest observed
    /// frequency is ever targeted, clamped to not remove more than needed
    /// to reach `monomial_range`'s floor). Returns how many were removed.
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
        let mut start = 0u64;
        for h in &self.heads {
            if (h.end - start) as usize == freq {
                hits += 1;
            }
            start = h.end;
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
        self.compact_by_len(|len| {
            if len == freq && removed < claim {
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
    /// the cos branch then inserts `cos(idx)` into every monomial's sorted
    /// run *in place* via a backward shift within `self`'s own arena.
    ///
    /// The in-place cos shift matters at scale: this runs per anticommuting
    /// (generator, term) pair inside a serial-per-term rayon task, and for a
    /// giant coefficient a rebuild into a fresh arena puts multi-MB
    /// allocation plus first-touch page faults on that serial critical path
    /// every gate. Shifting within the existing buffer keeps its pages warm
    /// and its capacity across gates (growth is `reserve`-amortized), so the
    /// per-gate cost is one memmove of already-resident data.
    fn apply_rotation(&mut self, idx: &u32, phase: Complex64) -> Self {
        // sin branch scalar: * (i * phase). `phase` is always ±i here (this
        // is only called on anticommuting generator/term pairs), so `i *
        // phase` is always real — see the `MonoHead::scalar` doc comment.
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let branch_phase = branch_phase.re;

        let cos_factor = TrigFactor::cos(*idx);
        let sin_factor = TrigFactor::sin(*idx);
        let n = self.heads.len();

        // Sin branch first, while the arena is still un-shifted.
        let mut sin_factors: Vec<TrigFactor> = Vec::with_capacity(self.factors.len() + n);
        let mut sin_heads: Vec<MonoHead> = Vec::with_capacity(n);
        let mut start = 0usize;
        for head in &self.heads {
            let end = head.end as usize;
            let run = &self.factors[start..end];
            let pos = run.binary_search(&sin_factor).unwrap_or_else(|e| e);
            sin_factors.extend_from_slice(&run[..pos]);
            sin_factors.push(sin_factor);
            sin_factors.extend_from_slice(&run[pos..]);
            sin_heads.push(MonoHead { scalar: head.scalar * branch_phase, end: sin_factors.len() as u64 });
            start = end;
        }

        // Cos branch: back-to-front, each run shifts right by its index
        // (making room for one inserted factor per preceding monomial plus
        // its own). Suffix is moved before prefix so no source bytes are
        // overwritten before they're read; `copy_within` handles the
        // overlapping ranges. The fill value of `resize` is arbitrary —
        // every slot is overwritten below.
        let old_len = self.factors.len();
        self.factors.resize(old_len + n, cos_factor);
        let mut end = old_len;
        for i in (0..n).rev() {
            let start = if i == 0 { 0 } else { self.heads[i - 1].end as usize };
            let pos = self.factors[start..end].binary_search(&cos_factor).unwrap_or_else(|e| e);
            self.factors.copy_within(start + pos..end, start + i + pos + 1);
            if i > 0 {
                self.factors.copy_within(start..start + pos, start + i);
            }
            self.factors[start + i + pos] = cos_factor;
            self.heads[i].end += i as u64 + 1;
            end = start;
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
            let bytes = self.factors.len() * std::mem::size_of::<TrigFactor>();
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
    /// monomials. Factor runs are canonicalized (sorted) per monomial, same
    /// as real construction sites; monomial-level duplicates are allowed, so
    /// the result is flagged dirty like a real post-merge coefficient.
    fn coeff(monomials: &[(f64, &[(u32, bool)])]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, factors) in monomials {
            let mut run: Vec<TrigFactor> = factors
                .iter()
                .map(|&(idx, is_sin)| if is_sin { TrigFactor::sin(idx) } else { TrigFactor::cos(idx) })
                .collect();
            run.sort_unstable();
            c.push_monomial(scalar, &run);
        }
        c.dirty = true;
        c
    }

    /// Coefficient of monomials with exactly the given `(scalar, frequency,
    /// tag)` specs, each monomial's factors made unique by its tag.
    fn coeff_with_freqs(specs: &[(f64, usize, u32)]) -> SymbolicCoeff {
        let mut c = SymbolicCoeff::default();
        for &(scalar, freq, tag) in specs {
            let run: Vec<TrigFactor> = (0..freq).map(|p| TrigFactor::cos(tag * 1000 + p as u32)).collect();
            c.push_monomial(scalar, &run);
        }
        c
    }

    /// Reference evaluation independent of `evaluate`'s parallel path.
    fn naive_evaluate(c: &SymbolicCoeff, lut: &[f64]) -> f64 {
        c.iter_monomials()
            .map(|(scalar, run)| scalar * run.iter().map(|f| lut[f.0 as usize]).product::<f64>())
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
        let collected: Vec<(f64, Vec<TrigFactor>)> =
            c.iter_monomials().map(|(s, run)| (s, run.to_vec())).collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 1.5);
        assert_eq!(collected[0].1, vec![TrigFactor::sin(0), TrigFactor::cos(1)]);
        assert_eq!(collected[1].1, vec![TrigFactor::cos(3)]);
        assert!(collected[2].1.is_empty());
        assert_eq!(c.monomial_count(), 3);
    }

    #[test]
    fn apply_rotation_matches_trig_identity_and_keeps_runs_sorted() {
        let lut = make_lut(8);
        let mut c = SymbolicCoeff::from_scalar(0.75);
        // Descending parameter indices force real insertion (not just append).
        for idx in [5u32, 2, 7, 2, 0] {
            let before = naive_evaluate(&c, &lut);
            let sin_branch = c.apply_rotation(&idx, Complex64::new(0.0, -1.0));
            let (cos_t, sin_t) = (lut[(idx << 1) as usize], lut[((idx << 1) | 1) as usize]);
            assert!((naive_evaluate(&c, &lut) - cos_t * before).abs() < 1e-12);
            // branch_phase = (i * -i).re = 1.0
            assert!((naive_evaluate(&sin_branch, &lut) - sin_t * before).abs() < 1e-12);
            for (_, run) in c.iter_monomials().chain(sin_branch.iter_monomials()) {
                assert!(run.windows(2).all(|w| w[0] <= w[1]), "factor run must stay sorted");
            }
        }
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
        let runs: Vec<Vec<TrigFactor>> = a.iter_monomials().map(|(_, r)| r.to_vec()).collect();
        assert_eq!(runs[1], vec![TrigFactor::sin(2)]);
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
        // order-independence.
        let mut c = SymbolicCoeff::default();
        for rep in 0..3usize {
            for i in 0..n_params {
                for j in 0..n_params {
                    if i == j {
                        continue;
                    }
                    let mut run = [TrigFactor::cos(i as u32), TrigFactor::sin(j as u32)];
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
        for k in 0..HASH_MERGE_THRESHOLD as u32 {
            let mut run = [TrigFactor::cos(3), TrigFactor::sin(k % 4)];
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
            let mut run: Vec<TrigFactor> = (0..len)
                .map(|k| {
                    let v = (state >> (8 * k)) as u32 % (2 * n_params as u32);
                    TrigFactor(v)
                })
                .collect();
            run.sort_unstable();
            c.push_monomial(((state % 1000) as f64 - 500.0) / 250.0, &run);
        }
        let expected = naive_evaluate(&c, &lut);
        assert!((c.evaluate(&lut) - expected).abs() < 1e-9 * expected.abs().max(1.0));
    }
}
