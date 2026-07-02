use num_complex::Complex64;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use propaq_core::coeff::CoeffRepr;

/// Inline capacity for a monomial's factor list. Chosen comfortably above
/// typical `max_frequency` settings (e.g. 11) so factor lists stay inline —
/// spilling to the heap here means a separate allocation per monomial, which
/// at scale (hundreds of millions of monomials) turns into heavy concurrent
/// small-allocation traffic across worker threads.
type Factors = SmallVec<[TrigFactor; 16]>;

/// Insert `factor` into `factors`, keeping it sorted. `factors` is
/// canonicalized this way at every construction site (never appended to
/// directly), so it's always already sorted by the time `deduplicate` runs —
/// no separate sort pass is needed there. Duplicate factors (the same
/// parameter touched more than once) are kept, not merged: `cos(idx)` and
/// `cos(idx)` appearing twice represents `cos(theta)^2`, a distinct term.
#[inline]
fn insert_sorted_factor(factors: &mut Factors, factor: TrigFactor) {
    let pos = factors.binary_search(&factor).unwrap_or_else(|e| e);
    factors.insert(pos, factor);
}

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
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

/// A single term in a symbolic polynomial: `scalar * product(factors)`.
///
/// `scalar` is real, not complex: `apply_rotation` is only ever invoked on
/// anticommuting (generator, term) pairs, and for Hermitian, involutory
/// operators the commutator phase in that case is always purely imaginary
/// (`±i`); multiplying by the explicit `i` in `apply_rotation` cancels it,
/// leaving a real result at every step. Given a real (Hermitian) seed
/// observable, every monomial's scalar stays real by induction.
///
/// `factors` is always kept sorted (see `insert_sorted_factor`) — two
/// monomials with the same factor content compare/hash equal regardless of
/// the order gates touched them in, with no separate canonicalization step
/// needed before merging.
#[derive(Clone)]
pub struct Monomial {
    pub scalar: f64,
    pub factors: Factors,
}

impl Monomial {
    fn new(scalar: f64) -> Self {
        Monomial { scalar, factors: SmallVec::new() }
    }
}

/// A sum of monomials: represents a symbolic coefficient accumulated
/// during surrogate propagation.
///
/// `add_assign` simply appends monomials; call `deduplicate` to merge
/// identical factor patterns and drop near-zero terms before evaluation.
#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    pub monomials: Vec<Monomial>,
}

impl SymbolicCoeff {
    /// Single scalar monomial with no trig factors (used to seed from observable).
    pub fn from_scalar(c: f64) -> Self {
        SymbolicCoeff { monomials: vec![Monomial::new(c)] }
    }

    /// Insert a cos(param_idx) factor into every existing monomial, preserving order.
    pub fn multiply_cos(&mut self, idx: u32) {
        let factor = TrigFactor::cos(idx);
        for m in &mut self.monomials {
            insert_sorted_factor(&mut m.factors, factor);
        }
    }

    /// Clone self, multiply each scalar by `phase`, insert sin(param_idx) preserving order.
    pub fn branch_sin(&self, idx: u32, phase: f64) -> Self {
        let factor = TrigFactor::sin(idx);
        let monomials = self.monomials.iter().map(|m| {
            let mut factors = m.factors.clone();
            insert_sorted_factor(&mut factors, factor);
            Monomial { scalar: m.scalar * phase, factors }
        }).collect();
        SymbolicCoeff { monomials }
    }

    /// Multiply all scalars by a real factor (for noise damping).
    pub fn scale(&mut self, factor: f64) {
        for m in &mut self.monomials {
            m.scalar *= factor;
        }
    }

    /// Drop monomials with frequency (factor count) > max_freq.
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.monomials.retain(|m| m.factors.len() <= max_freq);
    }

    /// Merge monomials with identical factor patterns and drop near-zero
    /// results. Each monomial's own `factors` is already canonically sorted
    /// (an invariant maintained by `insert_sorted_factor` at every
    /// construction site), so no per-monomial sort is needed here.
    ///
    /// Below `HASH_MERGE_THRESHOLD` monomials, sorts the whole list and merges
    /// adjacent equal entries (`O(k log k)`, cheap for small `k`). Above it,
    /// groups monomials by factor pattern in a hashmap instead (`O(k)`
    /// amortized) — avoids the comparison sort that dominates a flush when one
    /// term's coefficient has ballooned to a large `k`.
    pub fn deduplicate(&mut self) {
        if self.monomials.len() <= 1 {
            return;
        }

        if self.monomials.len() < HASH_MERGE_THRESHOLD {
            self.monomials.sort_unstable_by(|a, b| a.factors.cmp(&b.factors));

            let mut out: Vec<Monomial> = Vec::with_capacity(self.monomials.len());
            for m in self.monomials.drain(..) {
                if let Some(last) = out.last_mut() {
                    if last.factors == m.factors {
                        last.scalar += m.scalar;
                        continue;
                    }
                }
                out.push(m);
            }
            out.retain(|m| m.scalar.abs() > 1e-15);
            self.monomials = out;
            return;
        }

        let mut acc: FxHashMap<Factors, f64> = FxHashMap::default();
        acc.reserve(self.monomials.len());
        for m in self.monomials.drain(..) {
            *acc.entry(m.factors).or_insert(0.0) += m.scalar;
        }
        self.monomials = acc
            .into_iter()
            .filter(|(_, scalar)| scalar.abs() > 1e-15)
            .map(|(factors, scalar)| Monomial { scalar, factors })
            .collect();
    }

    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Evaluate at the given (cos, sin) lookup table indexed by param_index.
    pub fn evaluate(&self, cos_sin: &[(f64, f64)]) -> f64 {
        self.monomials.iter().map(|m| {
            let prod: f64 = m.factors.iter().map(|&f| {
                let (c, s) = cos_sin[f.param_index() as usize];
                if f.is_sin() { s } else { c }
            }).product();
            m.scalar * prod
        }).sum()
    }

    /// Maximum factor count across all monomials; 0 if empty.
    pub fn max_frequency(&self) -> usize {
        self.monomials.iter().map(|m| m.factors.len()).max().unwrap_or(0)
    }
}

impl CoeffRepr for SymbolicCoeff {
    /// Gate parameter is a parameter index (u32).
    type GateParam = u32;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        // Seed observables are Hermitian, so their Pauli/Majorana-basis
        // coefficients are real; see the `Monomial::scalar` doc comment.
        debug_assert!(c.im.abs() < 1e-9, "surrogate seed coefficient must be real: {c:?}");
        SymbolicCoeff::from_scalar(c.re)
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.monomials.extend(other.monomials);
    }

    #[inline]
    fn apply_rotation(&mut self, idx: &u32, phase: Complex64) -> Self {
        // sin branch: clone * (i * phase), push sin factor. `phase` is always
        // ±i here (this is only called on anticommuting generator/term pairs),
        // so `i * phase` is always real — see the `Monomial::scalar` doc comment.
        let branch_phase = Complex64::new(0.0, 1.0) * phase;
        debug_assert!(branch_phase.im.abs() < 1e-9, "expected real branch phase: {branch_phase:?}");
        let sin_branch = self.branch_sin(*idx, branch_phase.re);
        // cos branch (self): push cos factor in-place
        self.multiply_cos(*idx);
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        self.scale(factor);
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
        self.monomials.len()
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
        obj.getattr("param_index")?.extract()
    }
}

#[cfg(test)]
mod dedup_tests {
    use super::*;

    /// Builds via `insert_sorted_factor`, same as real construction sites, so
    /// two calls with the same (idx, is_sin) content in different input order
    /// produce the same canonical `Monomial` — exercising the actual
    /// order-independence invariant instead of relying on `deduplicate` to
    /// paper over an out-of-invariant input.
    fn mono(scalar: f64, factors: &[(u32, bool)]) -> Monomial {
        let mut fs: Factors = SmallVec::new();
        for &(idx, is_sin) in factors {
            let factor = if is_sin { TrigFactor::sin(idx) } else { TrigFactor::cos(idx) };
            insert_sorted_factor(&mut fs, factor);
        }
        Monomial { scalar, factors: fs }
    }

    /// Reference evaluation over a raw (possibly duplicated, unsorted) monomial
    /// list, computed independently of `deduplicate`/`evaluate` — ground truth
    /// for checking that merging never changes the represented value.
    fn naive_evaluate(monomials: &[Monomial], cos_sin: &[(f64, f64)]) -> f64 {
        monomials.iter().map(|m| {
            let prod: f64 = m.factors.iter().map(|&f| {
                let (c, s) = cos_sin[f.param_index() as usize];
                if f.is_sin() { s } else { c }
            }).product();
            m.scalar * prod
        }).sum()
    }

    #[test]
    fn merges_same_pattern_different_insertion_order() {
        let mut c = SymbolicCoeff {
            monomials: vec![
                mono(1.0, &[(1, false), (2, false)]),
                mono(2.0, &[(2, false), (1, false)]),
            ],
        };
        c.deduplicate();
        assert_eq!(c.monomials.len(), 1);
        assert!((c.monomials[0].scalar - 3.0).abs() < 1e-12);
    }

    #[test]
    fn drops_near_zero_after_merge() {
        let mut c = SymbolicCoeff {
            monomials: vec![
                mono(1.0, &[(0, false)]),
                mono(-1.0, &[(0, false)]),
            ],
        };
        c.deduplicate();
        assert!(c.monomials.is_empty());
    }

    #[test]
    fn hash_path_matches_naive_evaluation() {
        let n_params = 400;
        let cos_sin: Vec<(f64, f64)> = (0..n_params)
            .map(|i| {
                let t = 0.37 * (i as f64 + 1.0);
                (t.cos(), t.sin())
            })
            .collect();

        // > HASH_MERGE_THRESHOLD monomials with many repeated factor patterns,
        // inserted in varying order, to exercise the hash-merge path and its
        // order-independence.
        let mut raw: Vec<Monomial> = Vec::new();
        for rep in 0..3usize {
            for i in 0..n_params {
                for j in 0..n_params {
                    if i == j {
                        continue;
                    }
                    let factors = if rep % 2 == 0 {
                        vec![(i as u32, false), (j as u32, true)]
                    } else {
                        vec![(j as u32, true), (i as u32, false)]
                    };
                    raw.push(mono(0.1 * (rep as f64 + 1.0), &factors));
                }
            }
        }
        assert!(raw.len() >= HASH_MERGE_THRESHOLD, "test setup should exercise the hash path");

        let expected = naive_evaluate(&raw, &cos_sin);

        let mut c = SymbolicCoeff { monomials: raw };
        c.deduplicate();
        let actual = c.evaluate(&cos_sin);

        assert!(
            (actual - expected).abs() < 1e-9,
            "hash-merge path changed the evaluated value: {actual} vs {expected}"
        );
    }

    #[test]
    fn small_and_large_paths_agree() {
        // Same logical multiset, once below and once above HASH_MERGE_THRESHOLD
        // (padded with exactly-cancelling pairs), should evaluate identically.
        let cos_sin: Vec<(f64, f64)> = (0..4).map(|i| {
            let t = 0.2 * (i as f64 + 1.0);
            (t.cos(), t.sin())
        }).collect();

        let small_raw = vec![
            mono(1.0, &[(0, false), (1, true)]),
            mono(2.0, &[(1, true), (0, false)]),
            mono(-0.5, &[(2, false)]),
        ];
        let small_expected = naive_evaluate(&small_raw, &cos_sin);
        let mut small = SymbolicCoeff { monomials: small_raw.clone() };
        small.deduplicate();
        assert!(small.monomials.len() < HASH_MERGE_THRESHOLD);
        assert!((small.evaluate(&cos_sin) - small_expected).abs() < 1e-12);

        let mut large_raw = small_raw.clone();
        for k in 0..HASH_MERGE_THRESHOLD {
            large_raw.push(mono(5.0, &[(3, false), (k as u32 % 4, true)]));
            large_raw.push(mono(-5.0, &[(k as u32 % 4, true), (3, false)]));
        }
        assert!(large_raw.len() >= HASH_MERGE_THRESHOLD);
        let large_expected = naive_evaluate(&large_raw, &cos_sin);
        assert!((large_expected - small_expected).abs() < 1e-9);

        let mut large = SymbolicCoeff { monomials: large_raw };
        large.deduplicate();
        assert!((large.evaluate(&cos_sin) - small_expected).abs() < 1e-9);
    }
}
