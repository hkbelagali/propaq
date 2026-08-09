///
/// The operator container and its rotation kernel, on monoprop's lifecycle.
///
/// Keys live in an [`OperatorIndex`], coefficients in a row-aligned `Vec`. This
/// is still structure-of-arrays, deliberately: monoprop separates its store from
/// its `op_coeffs` the same way. What changes against `soa::SoaTermSum` is the
/// lifecycle. Duplicates are folded when a term is emitted, against the index
/// the store already carries, so there is no merge pass, no flags column, no
/// prefix sum, and no compaction. Rows are never removed, so a row index is
/// stable for the store's life.
///
use num_complex::Complex64;

use crate::algebra::Algebra;
use crate::coeff::CoeffRepr;
use crate::monomial::Monomial;
use crate::inverted_index::{for_each_set_bit, InvertedIndex};
use crate::operator_index::{OperatorIndex, Pos, TermIndexCeilingReached};

/// Tolerance for treating `sin(theta)` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// Truncation applied when a child term is emitted, rather than after the fact.
///
/// This is where the engine's semantics differ from the SoA engine, so the two
/// bounds are kept separate and independently switchable.
#[derive(Clone, Copy, Default, Debug)]
pub struct EmitCutoff {
    /// Structural weight bound. A child above it is never created.
    ///
    /// Converts exactly: weight is a property of the key alone, so deciding at
    /// emit time and deciding after accumulation agree.
    pub max_weight: Option<u32>,
    /// Magnitude bound on the emitted branch, monoprop's `atol`.
    ///
    /// This does *not* convert exactly. It bounds the child from its parent
    /// before the child exists, so a term whose coefficient only becomes small
    /// once several contributions cancel is kept here and dropped by a
    /// post-accumulation cutoff.
    pub min_coeff: Option<f64>,
}

impl EmitCutoff {
    /// No truncation at all, which is the setting the equivalence tests use.
    pub fn none() -> Self {
        Self::default()
    }

    /// True if a child with this key and coefficient is worth creating.
    #[inline]
    pub(crate) fn admits<A: Algebra<W>, C: CoeffRepr, const W: usize>(
        &self,
        key: &Monomial<W>,
        coeff: &C,
    ) -> bool {
        if let Some(w) = self.max_weight {
            if A::weight(key) > w {
                return false;
            }
        }
        if let Some(c) = self.min_coeff {
            if coeff.magnitude() < c {
                return false;
            }
        }
        true
    }
}

/// A sum of terms: keys in a fixed-stride indexed store, coefficients alongside.
pub struct Operator<C: CoeffRepr, P: Pos, const W: usize> {
    store: OperatorIndex<P, W>,
    coeffs: Vec<C>,
    /// Transposed view of `store`, synced lazily before each indexed scan.
    inverted: InvertedIndex,
    /// Reused scan buffers, so a gate allocates nothing.
    scan_bitmap: Vec<u64>,
    scan_rows: Vec<usize>,
    n_units: usize,
}

impl<C: CoeffRepr, P: Pos, const W: usize> Operator<C, P, W> {
    /// Creates an empty operator over `n_units` qubits or modes.
    pub fn new(n_units: usize) -> Self {
        Operator {
            store: OperatorIndex::with_default_width(),
            coeffs: Vec::new(),
            inverted: InvertedIndex::new(Monomial::<W>::num_bits()),
            scan_bitmap: Vec::new(),
            scan_rows: Vec::new(),
            n_units,
        }
    }

    /// Creates an empty operator whose rows are sized for a structural cutoff.
    pub fn with_weight_cutoff(n_units: usize, max_weight: usize) -> Self {
        let width = OperatorIndex::<P, W>::inline_width_for_support_cutoff(max_weight);
        Operator {
            store: OperatorIndex::new(width),
            coeffs: Vec::new(),
            inverted: InvertedIndex::new(Monomial::<W>::num_bits()),
            scan_bitmap: Vec::new(),
            scan_rows: Vec::new(),
            n_units,
        }
    }

    /// Number of live terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// True if the operator has no terms.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Number of qubits or modes this operator is sized for.
    #[inline]
    pub fn n_units(&self) -> usize {
        self.n_units
    }

    /// Row `i`'s key.
    #[inline]
    pub fn key(&self, i: usize) -> Monomial<W> {
        self.store.row(i)
    }

    /// Row `i`'s coefficient.
    #[inline]
    pub fn coeff(&self, i: usize) -> &C {
        &self.coeffs[i]
    }

    /// The underlying store, for memory accounting.
    #[inline]
    pub fn store(&self) -> &OperatorIndex<P, W> {
        &self.store
    }

    /// Adds `coeff` to `key`'s term, creating the term if it is absent.
    pub fn add(&mut self, key: &Monomial<W>, coeff: C) -> Result<(), TermIndexCeilingReached> {
        if let Some(row) = self.store.find(key) {
            self.coeffs[row].add_assign(coeff);
            return Ok(());
        }
        let row = self.store.push(key)?;
        self.store.insert_absent(key, row)?;
        debug_assert_eq!(self.coeffs.len(), row);
        self.coeffs.push(coeff);
        Ok(())
    }

    /// Bytes of resident key storage, rows plus index.
    pub fn key_bytes(&self) -> usize {
        self.store.memory_bytes() + self.store.index_memory_bytes()
    }

    /// Bytes held by the transposed index, which is scan acceleration rather
    /// than key storage and so is reported separately.
    pub fn inverted_index_bytes(&self) -> usize {
        self.inverted.memory_bytes()
    }

    /// Dense bytes, sparse bytes, and dense column count of the transposed index.
    pub fn inverted_index_tiers(&self) -> (usize, usize, usize) {
        self.inverted.tier_stats()
    }

    /// Scales every anticommuting term by `factor`, emitting nothing.
    ///
    /// The rotation path a vanishing sine collapses to.
    pub fn scale_anticommuting<A: Algebra<W>>(&mut self, ctx: &A::GenContext, factor: f64) {
        for i in 0..self.store.len() {
            if A::anticommutes(ctx, &self.store.row(i)) {
                self.coeffs[i].scale_real(factor);
            }
        }
    }

    /// Rotates every source into its cosine branch and routes the sine branches
    /// into `outbox`, bucketed by the partition that owns each child.
    ///
    /// This is the half of a rotation that touches only this partition's own
    /// rows, so it is safe to run concurrently across partitions. Nothing is
    /// accumulated here: an emitted child can land on a row that is itself a
    /// source this same gate, so every sine branch must be taken against a
    /// pre-rotation coefficient before any absorption starts.
    pub fn scan_into<A: Algebra<W>>(
        &mut self,
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        n_partitions: usize,
        outbox: &mut [Vec<(Monomial<W>, u64, C)>],
    ) {
        // A basis whose column fold is not exact has no usable index yet, so it
        // keeps the per-term scan.
        if A::fold_needs_odd_correction(ctx) {
            for i in 0..self.store.len() {
                if A::anticommutes(ctx, &self.store.row(i)) {
                    self.emit_from_row::<A>(i, ctx, param, cutoff, n_partitions, outbox);
                }
            }
            return;
        }

        // Otherwise fold the generator's columns into one bitmap and visit only
        // the terms that branch. The index is synced lazily: the store only
        // grows, so rows already indexed never need revisiting.
        self.inverted.sync_to(&self.store);
        let mut bitmap = std::mem::take(&mut self.scan_bitmap);
        self.inverted.combine(A::fold_generator(ctx).positions(), &mut bitmap);

        let mut rows = std::mem::take(&mut self.scan_rows);
        rows.clear();
        for_each_set_bit(&bitmap, |r| rows.push(r));
        for &i in rows.iter() {
            self.emit_from_row::<A>(i, ctx, param, cutoff, n_partitions, outbox);
        }

        self.scan_bitmap = bitmap;
        self.scan_rows = rows;
    }

    /// Rotates row `i` into its cosine branch and routes the sine branch.
    ///
    /// Split out so the indexed and per-term scans cannot drift apart.
    #[inline]
    fn emit_from_row<A: Algebra<W>>(
        &mut self,
        i: usize,
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        n_partitions: usize,
        outbox: &mut [Vec<(Monomial<W>, u64, C)>],
    ) {
        let mono = self.store.row(i);
        debug_assert!(A::anticommutes(ctx, &mono), "a selected row must anticommute");
        let (child, phase) = A::product(ctx, &mono);
        debug_assert!(phase.re.abs() < 1e-9, "an anticommuting product must be purely imaginary");
        let sin_branch = self.coeffs[i].apply_rotation(param, phase);
        if !cutoff.admits::<A, C, W>(&child, &sin_branch) {
            return;
        }
        // The hash is needed to route the child anyway, so it travels with it
        // rather than being recomputed by the absorbing partition.
        let hash = OperatorIndex::<P, W>::hash_of(&child);
        outbox[partition_from_hash(hash, n_partitions)].push((child, hash, sin_branch));
    }

    /// Folds one routed child into this partition, appending it if absent.
    ///
    /// Returns true when a new row was created. `insert_absent` is sound on the
    /// append path because a gate's children are pairwise distinct: `M -> M ^ G`
    /// is injective for fixed `G`, so no two absorbed keys in one gate collide,
    /// and this one was just confirmed absent.
    pub fn absorb(&mut self, key: &Monomial<W>, coeff: C) -> Result<bool, TermIndexCeilingReached> {
        let hash = OperatorIndex::<P, W>::hash_of(key);
        self.absorb_with_hash(key, hash, coeff)
    }

    /// Issues a prefetch for the table slot `hash` will probe.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.store.prefetch_for_hash(hash);
    }

    /// [`Operator::absorb`] with the key's hash already computed.
    pub fn absorb_with_hash(
        &mut self,
        key: &Monomial<W>,
        hash: u64,
        coeff: C,
    ) -> Result<bool, TermIndexCeilingReached> {
        if let Some(row) = self.store.find_with_hash(key, hash) {
            self.coeffs[row].add_assign(coeff);
            self.coeffs[row].post_merge();
            return Ok(false);
        }
        let row = self.store.push(key)?;
        self.store.insert_absent_with_hash(row, hash)?;
        debug_assert_eq!(self.coeffs.len(), row);
        self.coeffs.push(coeff);
        Ok(true)
    }

    /// Applies the rotation generated by `gen`, returning the number of new
    /// terms created.
    ///
    /// The single-partition case of the partitioned engine: scan into one
    /// bucket, then drain it.
    pub fn apply_rotation<A: Algebra<W>>(
        &mut self,
        gen: &Monomial<W>,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if self.store.is_empty() {
            return Ok(0);
        }
        let ctx = A::make_gen_context(gen);

        // A rotation whose sine vanishes only rescales the anticommuting terms.
        // Short-circuiting keeps it from appending a zero-coefficient row that
        // an append-only store could never reclaim.
        if let Some(cos_t) = C::phase_only_scale(param, PHASE_ONLY_EPS) {
            self.scale_anticommuting::<A>(&ctx, cos_t);
            return Ok(0);
        }

        let mut outbox = [Vec::new()];
        self.scan_into::<A>(&ctx, param, cutoff, 1, &mut outbox);
        let [pending] = outbox;
        let mut added = 0usize;
        for (child, hash, coeff) in pending {
            if self.absorb_with_hash(&child, hash, coeff)? {
                added += 1;
            }
        }
        Ok(added)
    }

    /// Expectation value against a computational basis state.
    pub fn expectation<A: Algebra<W>>(&self, fock: &[u64]) -> f64 {
        (0..self.store.len())
            .map(|i| self.coeffs[i].to_f64() * A::trace(&self.store.row(i), fock))
            .sum()
    }

    /// Every live term as a key and coefficient pair.
    pub fn iter(&self) -> impl Iterator<Item = (Monomial<W>, &C)> + '_ {
        (0..self.store.len()).map(move |i| (self.store.row(i), &self.coeffs[i]))
    }

    /// Number of terms whose coefficient is exactly zero.
    ///
    /// An append-only store never reclaims these, so this is the running cost
    /// of not compacting and is worth reporting alongside `len`.
    pub fn zero_terms(&self) -> usize {
        self.coeffs[..self.store.len()].iter().filter(|c| c.magnitude() == 0.0).count()
    }
}

/// The partition that owns `key`.
///
/// Ownership is by key alone, so a term lives in exactly one partition for its
/// whole life and a rotation's child is routed to whichever partition owns it.
/// Uses the high half of the hash: the low bits already drive the store's own
/// index table, and reusing them here would correlate the two.
#[inline]
pub fn partition_of<const W: usize>(key: &Monomial<W>, n_partitions: usize) -> usize {
    partition_from_hash(key.hash_value(), n_partitions)
}

/// [`partition_of`] with the key's hash already computed.
#[inline]
pub fn partition_from_hash(hash: u64, n_partitions: usize) -> usize {
    if n_partitions <= 1 {
        return 0;
    }
    ((hash >> 32) % n_partitions as u64) as usize
}

/// Phase of the product `G * M` as a complex factor, for callers that need to
/// build one outside an `Algebra`.
pub fn imaginary_phase(sign: f64) -> Complex64 {
    Complex64::new(0.0, sign)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monomial::Monomial;

    const W: usize = 1;

    /// A minimal test algebra: units are single bits, terms anticommute when
    /// their overlap with the generator is odd, and the product is an XOR.
    struct TestAlgebra;

    impl Algebra<W> for TestAlgebra {
        type GenContext = Monomial<W>;

        fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext {
            assert_eq!(sign, 1.0, "the test algebra carries no generator sign");
            *gen
        }
        fn generator(ctx: &Self::GenContext) -> &Monomial<W> {
            ctx
        }
        fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool {
            mono.parity_and(ctx)
        }
        // The test algebra's fold is the generator itself: anticommutation is
        // the plain overlap parity, with no pair swap.
        fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W> {
            ctx
        }
        fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64) {
            (*mono ^ *ctx, Complex64::new(0.0, 1.0))
        }
        fn weight(mono: &Monomial<W>) -> u32 {
            mono.count() as u32
        }
        fn trace(mono: &Monomial<W>, fock: &[u64]) -> f64 {
            let f = fock.first().copied().unwrap_or(0);
            if mono.words()[0] & f == 0 {
                1.0
            } else {
                -1.0
            }
        }
    }

    type Op = Operator<f64, u16, W>;

    fn mono(bits: &[usize]) -> Monomial<W> {
        Monomial::from_positions(bits.iter().copied())
    }

    fn values(op: &Op) -> std::collections::HashMap<u64, f64> {
        op.iter().map(|(k, c)| (k.words()[0], *c)).collect()
    }

    #[test]
    fn add_folds_a_duplicate_key_instead_of_appending() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1.0).unwrap();
        op.add(&mono(&[1]), 2.0).unwrap();
        op.add(&mono(&[0]), 3.0).unwrap();
        assert_eq!(op.len(), 2, "the duplicate must fold, not append");
        let v = values(&op);
        assert_eq!(v[&0b01], 4.0);
        assert_eq!(v[&0b10], 2.0);
    }

    #[test]
    fn a_commuting_term_is_untouched_by_a_rotation() {
        let mut op = Op::new(8);
        op.add(&mono(&[1]), 5.0).unwrap();
        // Overlap with generator {0} is zero, so it commutes.
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.7, &EmitCutoff::none()).unwrap();
        assert_eq!(added, 0);
        assert_eq!(op.len(), 1);
        assert_eq!(*op.coeff(0), 5.0);
    }

    #[test]
    fn an_anticommuting_term_splits_into_cos_and_sin_branches() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 2.0).unwrap();
        let angle = 0.3f64;
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[0]), &angle, &EmitCutoff::none()).unwrap();
        assert_eq!(added, 1);
        assert_eq!(op.len(), 2);
        let v = values(&op);
        assert!((v[&0b01] - 2.0 * angle.cos()).abs() < 1e-12, "cos branch");
        assert!((v[&0b00] - (2.0 * angle.sin() * -1.0)).abs() < 1e-12, "sin branch");
    }

    /// A generator of even popcount, so that a key and its image under the
    /// generator both anticommute with it.
    ///
    /// With `M2 = M1 ^ G`, `parity(M2 & G) = parity(M1 & G) ^ parity(G)`, so an
    /// odd-popcount generator always sends an anticommuting key to a commuting
    /// one and the two can never be sources in the same gate.
    fn paired_generator() -> Monomial<W> {
        mono(&[0, 1])
    }

    #[test]
    fn a_child_landing_on_an_existing_row_accumulates_rather_than_appending() {
        let mut op = Op::new(8);
        // Both keys anticommute with {0,1}, and each maps onto the other.
        op.add(&mono(&[0]), 2.0).unwrap();
        op.add(&mono(&[1]), 3.0).unwrap();
        assert_eq!(op.len(), 2);
        let added = op
            .apply_rotation::<TestAlgebra>(&paired_generator(), &0.3, &EmitCutoff::none())
            .unwrap();
        assert_eq!(added, 0, "both children already exist, so nothing is appended");
        assert_eq!(op.len(), 2);
    }

    #[test]
    fn the_sine_branch_is_taken_against_the_pre_rotation_coefficient() {
        // Rows 0 and 1 map onto each other under the generator and are both
        // sources this gate. If phase 1 did not snapshot before phase 2
        // accumulated, one row's sine branch would be taken against a
        // coefficient the other had already rotated.
        let angle = 0.4f64;
        let (c0, c1) = (2.0f64, 5.0f64);
        let mut op = Op::new(8);
        op.add(&mono(&[0]), c0).unwrap();
        op.add(&mono(&[1]), c1).unwrap();
        op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &EmitCutoff::none()).unwrap();

        let v = values(&op);
        let (sin_t, cos_t) = angle.sin_cos();
        // Each row keeps its own cos branch and receives the other's sin branch.
        let want_0 = c0 * cos_t + c1 * sin_t * -1.0;
        let want_1 = c1 * cos_t + c0 * sin_t * -1.0;
        assert!((v[&0b01] - want_0).abs() < 1e-12, "got {}, want {want_0}", v[&0b01]);
        assert!((v[&0b10] - want_1).abs() < 1e-12, "got {}, want {want_1}", v[&0b10]);
    }

    #[test]
    fn a_phase_only_rotation_scales_without_appending() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 2.0).unwrap();
        op.add(&mono(&[1]), 3.0).unwrap();
        let added = op
            .apply_rotation::<TestAlgebra>(&mono(&[0]), &std::f64::consts::PI, &EmitCutoff::none())
            .unwrap();
        assert_eq!(added, 0);
        assert_eq!(op.len(), 2, "a phase-only rotation must not grow the store");
        let v = values(&op);
        assert!((v[&0b01] + 2.0).abs() < 1e-12, "anticommuting term scaled by cos(pi)");
        assert_eq!(v[&0b10], 3.0, "commuting term untouched");
    }

    #[test]
    fn a_weight_cutoff_suppresses_the_child_at_emit_time() {
        let mut op = Op::new(8);
        op.add(&mono(&[0, 1]), 1.0).unwrap();
        // The child is {0,1,2}, weight 3, above the cutoff.
        let cutoff = EmitCutoff { max_weight: Some(2), min_coeff: None };
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[2]), &0.3, &cutoff).unwrap();
        assert_eq!(added, 0, "the over-weight child must never be created");
        assert_eq!(op.len(), 1);
    }

    #[test]
    fn a_coefficient_cutoff_suppresses_a_small_child_at_emit_time() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1e-12).unwrap();
        let cutoff = EmitCutoff { max_weight: None, min_coeff: Some(1e-6) };
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &cutoff).unwrap();
        assert_eq!(added, 0, "the tiny child must never be created");
        assert_eq!(op.len(), 1);
    }

    #[test]
    fn repeated_rotations_keep_the_store_deduplicated() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1.0).unwrap();
        for _ in 0..12 {
            op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none()).unwrap();
        }
        // The generator toggles bit 0, so the orbit is exactly two keys no
        // matter how many times it is applied.
        assert_eq!(op.len(), 2, "dedup on insert must bound the orbit");
    }

    #[test]
    fn expectation_sums_coefficient_times_trace() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 2.0).unwrap();
        op.add(&mono(&[1]), 3.0).unwrap();
        let got = op.expectation::<TestAlgebra>(&[0b01]);
        assert!((got - (2.0 * -1.0 + 3.0 * 1.0)).abs() < 1e-12);
    }

    #[test]
    fn rows_are_never_removed_so_indices_stay_stable() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1.0).unwrap();
        let key0 = op.key(0);
        for _ in 0..8 {
            op.apply_rotation::<TestAlgebra>(&mono(&[1]), &0.3, &EmitCutoff::none()).unwrap();
        }
        assert_eq!(op.key(0), key0, "row 0 must still hold its original key");
    }

    #[test]
    fn u16_positions_are_wide_enough_for_this_width() {
        assert!(Monomial::<W>::num_bits() <= u16::MAX as usize);
    }
}
