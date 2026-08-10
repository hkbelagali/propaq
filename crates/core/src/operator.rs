///
/// The operator container and its rotation kernel, on monoprop's lifecycle.
///
/// Keys live in an [`OperatorIndex`], coefficients in a row-aligned `Vec`. This
/// is still structure-of-arrays, deliberately: monoprop separates its store from
/// its `op_coeffs` the same way. What changes against `store::TermSum` is the
/// lifecycle. Duplicates are folded when a term is emitted, against the index
/// the store already carries, so there is no merge pass, no flags column, no
/// prefix sum, and no compaction. Rows are never removed, so a row index is
/// stable for the store's life.
///
use num_complex::Complex64;

use crate::algebra::Algebra;
use crate::coeff::CoeffRepr;
use crate::monomial::Monomial;
use crate::native_truncator::NativeTruncator;
use crate::inverted_index::{for_each_set_bit, InvertedIndex};
use crate::operator_index::{OperatorIndex, Pos, TermIndexCeilingReached};

/// Tolerance for treating `sin(theta)` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// Truncation applied when a child term is emitted, rather than after the fact.
///
/// This is where the engine's semantics differ from the merge-then-sweep engine
/// propaq used before, so the two
/// bounds are kept separate and independently switchable.
/// The truncation applied when a child term is emitted, rather than after the
/// fact.
///
/// This is where the engine's semantics differ from the previous engine's, which
/// created every child and swept afterwards.
///
/// Not `Copy`: `NativeTruncator` holds an `Arc`-shared plugin handle.
#[derive(Clone, Default)]
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
    /// A loaded plugin deciding per term, from `(weight, |coeff|, active_modes)`.
    ///
    /// When set it *replaces* `max_weight` and `min_coeff` rather than adding to
    /// them, which is what `ResolvedConfig::native` has always meant. It
    /// also disables the emit precheck: the precheck predicts a branch's fate
    /// from its source alone, and an opaque predicate over the child's weight
    /// cannot be predicted that way.
    pub native: Option<NativeTruncator>,
    /// Live-term floor, from `TermBudget::min_terms`.
    ///
    /// While the store holds fewer terms than this, the lossy predicates above
    /// are suppressed and only lossless work happens. Evaluated once per gate
    /// against the whole operator, not per partition, since the budget counts
    /// the operator.
    pub min_terms: Option<usize>,
}

impl EmitCutoff {
    /// The same cutoff with every lossy predicate removed.
    ///
    /// What `TermBudget::min_terms` asks for: below the floor, only lossless
    /// work happens and nothing is refused.
    pub fn lossless(&self) -> Self {
        EmitCutoff { min_terms: self.min_terms, ..Default::default() }
    }

    /// This cutoff as it applies to an operator currently holding `n_live` terms.
    ///
    /// Evaluated once per gate against the whole operator rather than per
    /// partition: the budget counts the operator, and a partition holding a
    /// fraction of the terms would otherwise suppress the cutoff far too long.
    pub fn at_size(&self, n_live: usize) -> std::borrow::Cow<'_, Self> {
        match self.min_terms {
            Some(floor) if n_live < floor => std::borrow::Cow::Owned(self.lossless()),
            _ => std::borrow::Cow::Borrowed(self),
        }
    }
}

impl std::fmt::Debug for EmitCutoff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmitCutoff")
            .field("max_weight", &self.max_weight)
            .field("min_coeff", &self.min_coeff)
            .field("native", &self.native.is_some())
            .field("min_terms", &self.min_terms)
            .finish()
    }
}

impl EmitCutoff {
    /// No truncation at all, which is the setting the equivalence tests use.
    pub fn none() -> Self {
        Self::default()
    }

    /// True if a child with this key may be created at all.
    ///
    /// Weight is a property of the key, so this is the half of the decision
    /// that no partner can overturn.
    #[inline]
    pub(crate) fn admits_key<A: Algebra<W>, const W: usize>(
        &self,
        key: &Monomial<W>,
        n_units: usize,
    ) -> bool {
        if self.native.is_some() {
            // The plugin needs the coefficient too, so the whole decision is
            // deferred to `admits_child`.
            return true;
        }
        match self.max_weight {
            Some(w) => A::weight(key, n_units) <= w,
            None => true,
        }
    }

    /// The full emit decision for a formed child: key and coefficient together.
    ///
    /// Split from [`EmitCutoff::admits_key`] because a native plugin decides on
    /// both at once, where the built-in bounds decide independently and so can
    /// reject on the key before a coefficient is even formed.
    #[inline]
    pub(crate) fn admits_child<A: Algebra<W>, C: CoeffRepr, const W: usize>(
        &self,
        key: &Monomial<W>,
        coeff: &C,
        n_units: usize,
    ) -> bool {
        match &self.native {
            // `active_modes` is passed as zero here exactly as the previous kernels
            // pass it at every call site; nothing has ever supplied a value.
            Some(nt) => nt.keep(A::weight(key, n_units), coeff.magnitude(), 0),
            None => self.admits_coeff(coeff),
        }
    }

    /// True if a term belongs in the store at all, key and coefficient both.
    ///
    /// Used for the terms an observable starts with, which reach the store
    /// through `add` rather than through a rotation and so never face the emit
    /// gate. The previous engine applied its cutoff to every live row on its first
    /// flush, initial terms included, so an observable carrying a term the
    /// cutoff excludes has to be filtered here or the two engines disagree
    /// before a single gate has run.
    pub fn admits_initial<A: Algebra<W>, C: CoeffRepr, const W: usize>(
        &self,
        key: &Monomial<W>,
        coeff: &C,
        n_units: usize,
    ) -> bool {
        self.admits_key::<A, W>(key, n_units) && self.admits_child::<A, C, W>(key, coeff, n_units)
    }

    /// True if a branch of this magnitude is worth carrying.
    #[inline]
    pub(crate) fn admits_coeff<C: CoeffRepr>(&self, coeff: &C) -> bool {
        match self.min_coeff {
            Some(c) => coeff.magnitude() >= c,
            None => true,
        }
    }

}


/// The sine-branch magnitude test, hoisted out of the per-term loop.
///
/// A rotation's sine branch has magnitude `|source| * |sin(theta)|`, which
/// depends on neither the source's key nor the child's. So a branch that cannot
/// clear the cutoff can be declined *before* its row is read and its product
/// formed, which is monoprop's predictive `abs_sin_val * abs_coeff >= atol` and
/// the reason its scan does not carry the work propaq's used to.
///
/// On 6x6 Ising-Trotter step 21 the scan reaches 4.8 billion rows and keeps 24%
/// of them, so three quarters of the products it formed were discarded.
#[derive(Clone, Copy)]
pub(crate) struct EmitPrecheck {
    /// Signed `sin(theta)`. A held-back branch is scaled by this and takes its
    /// `+-i` phase later, from the product the claim path already forms.
    sin: f64,
    cos: f64,
    min_coeff: f64,
}

impl EmitPrecheck {
    /// The test for this gate, or `None` when it does not apply.
    ///
    /// It does not apply without a coefficient cutoff, without a coefficient
    /// representation whose branches are a plain scaling, or under a native
    /// truncator, whose predicate over the child's weight cannot be predicted
    /// from the source.
    ///
    /// The pair rule does not disqualify it. A held-back branch's value is
    /// `source * sin(theta) * (-phase.im)`, and only the phase needs the
    /// product; the rest is known here. So the decline records
    /// `source * sin(theta)` and the claim path multiplies in the phase from the
    /// product it re-forms anyway, which is bit-identical to forming the branch
    /// up front and costs nothing on the branches that are never claimed.
    pub(crate) fn for_gate(factors: Option<(f64, f64)>, cutoff: &EmitCutoff) -> Option<Self> {
        if cutoff.native.is_some() {
            return None;
        }
        let min_coeff = cutoff.min_coeff?;
        let (sin, cos) = factors?;
        Some(EmitPrecheck { sin, cos, min_coeff })
    }

    /// True when this source cannot produce a branch that survives the cutoff.
    ///
    /// Bit-identical to testing the formed branch: an anticommuting product's
    /// phase is exactly `+-i` and so cannot change a magnitude, and
    /// [`CoeffRepr::sin_branch_magnitude`] multiplies in the same precision and
    /// the same order the branch itself will.
    #[inline]
    fn declines<C: CoeffRepr>(&self, coeff: &C) -> bool {
        coeff.sin_branch_magnitude(self.sin) < self.min_coeff
    }
}


/// One routed child: the key it lands on, the hash that routes it, and the
/// contribution it carries.
///
/// Deliberately no source address. Carrying one would let the partition that
/// owns the child pay a partner's half straight back without a second probe,
/// which is how monoprop settles a rotation pair from one end. Measured here it
/// loses: see `set_pair_scan` in `partitioned.rs`.
#[derive(Clone)]
pub struct Routed<C, const W: usize> {
    pub key: Monomial<W>,
    pub hash: u64,
    pub coeff: C,
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
    /// Branches this gate held back rather than routed: a follower's half, which
    /// its leader claims on arrival, or one the cutoff rejected.
    ///
    /// `pending_slot` maps a row to its entry, or `u32::MAX` for none. It costs
    /// four bytes per row and is reset through `pending_rows`, so the per-gate
    /// cost is the number held back rather than the store size. A search would
    /// be cheaper in memory and far worse in time: the absorb path looks one up
    /// per claim, millions of times a gate.
    pending_slot: Vec<u32>,
    pending_rows: Vec<u32>,
    pending_vals: Vec<C>,
    /// Rows whose held-back branch a partner has paid for this gate.
    claimed_rows: Vec<u32>,
    /// True when this gate's held-back values are `source * sin(theta)` with the
    /// product phase still to come, which is what the precheck records. False
    /// when they are whole branches taken from the full path. Uniform within a
    /// gate: the precheck and the full path's cutoff test are complementary, so
    /// only one of them can hold anything back.
    pending_needs_phase: bool,
    /// Children routed, and how many of those landed on a row that already
    /// existed. A hit means both halves of the pair are in the store, which is
    /// the case a pair-visiting scan could serve with one probe instead of two.
    emitted: u64,
    hits: u64,
    /// Anticommuting rows the scan reached, against `emitted`, which counts only
    /// the branches that then cleared the cutoff. The difference is work the
    /// scan did and threw away.
    visited: u64,
    /// Branches the cutoff declined. `declined` reached through the precheck
    /// cost a compare and a scaling; the rest cost a row read and a product.
    declined: u64,
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
            pending_slot: Vec::new(),
            pending_rows: Vec::new(),
            pending_vals: Vec::new(),
            claimed_rows: Vec::new(),
            pending_needs_phase: false,
            emitted: 0,
            hits: 0,
            visited: 0,
            declined: 0,
            n_units,
        }
    }

    /// Creates an empty operator whose rows are sized for a structural cutoff.
    pub fn with_weight_cutoff(n_units: usize, max_weight: usize) -> Self {
        Self::with_inline_positions(
            n_units,
            OperatorIndex::<P, W>::inline_width_for_support_cutoff(max_weight),
        )
    }

    /// Creates an empty operator holding `width` positions inline per row.
    ///
    /// Any width is correct: a row that outgrows it spills losslessly. It trades
    /// bytes per row against overflow-map lookups, and both sides of that trade
    /// are on the hot path, so callers that know the term support should say so.
    pub fn with_inline_positions(n_units: usize, width: usize) -> Self {
        Operator {
            store: OperatorIndex::new(width),
            coeffs: Vec::new(),
            inverted: InvertedIndex::new(Monomial::<W>::num_bits()),
            scan_bitmap: Vec::new(),
            scan_rows: Vec::new(),
            pending_slot: Vec::new(),
            pending_rows: Vec::new(),
            pending_vals: Vec::new(),
            claimed_rows: Vec::new(),
            pending_needs_phase: false,
            emitted: 0,
            hits: 0,
            visited: 0,
            declined: 0,
            n_units,
        }
    }

    /// Number of live terms.
    #[inline]
    /// Rebuilds the store at a wider inline row, keeping every term.
    ///
    /// A row holds `inline_width` positions and spills the rest into a hash map,
    /// so every read of an overflowed row costs a lookup. The right width grows
    /// with circuit depth, and the store cannot restride in place, so this
    /// reuses `reclaim`'s rebuild to do it: keep everything, change the stride.
    pub fn repack(&mut self, inline_width: usize) -> Result<(), TermIndexCeilingReached> {
        let mut store = OperatorIndex::<P, W>::new(inline_width);
        store.reserve(self.store.len());
        for i in 0..self.store.len() {
            let key = self.store.row(i);
            let row = store.push(&key)?;
            store.insert_absent_with_hash(row, OperatorIndex::<P, W>::hash_of(&key))?;
        }
        self.store = store;
        // Rows keep their indices here, unlike `reclaim`, but the transposed
        // view is rebuilt anyway since its column tiers were judged on the old
        // store.
        self.inverted.reset();
        self.pending_slot.clear();
        self.pending_rows.clear();
        self.pending_vals.clear();
        self.claimed_rows.clear();
        Ok(())
    }

    /// Positions held inline per row.
    pub fn inline_width(&self) -> usize {
        self.store.inline_width()
    }

    /// Rows whose positions spilled past the inline width.
    pub fn overflow_len(&self) -> usize {
        self.store.overflow_len()
    }

    /// Scales every coefficient by `factor(weight)`.
    ///
    /// The shape a damping channel needs: a term of Jordan-Wigner weight `w` is
    /// multiplied by some `f(w)`. Weight is a property of the key, so this reads
    /// rows and writes coefficients and touches no index.
    ///
    /// This is the operation the emit gate cannot compensate for. It only ever
    /// shrinks coefficients, so terms drift under the cutoff without any of them
    /// being emitted again; pair it with [`Operator::reclaim`].
    pub fn scale_by_weight<A: Algebra<W>>(&mut self, factor: impl Fn(u32) -> f64) {
        for i in 0..self.store.len() {
            let w = A::weight(&self.store.row(i), self.n_units);
            self.coeffs[i].scale_real(factor(w));
        }
    }

    /// [`Operator::scale_by_weight`] for a factor that can fail.
    ///
    /// Serial and short-circuiting, because the only caller that needs it is a
    /// Python noise model: the factor is a callback into the interpreter, so it
    /// holds the GIL, cannot be run in parallel, and can raise. Row order is the
    /// visit order, matching what the previous engine did.
    pub fn try_scale_by_weight<A: Algebra<W>, E>(
        &mut self,
        mut factor: impl FnMut(u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for i in 0..self.store.len() {
            let w = A::weight(&self.store.row(i), self.n_units);
            self.coeffs[i].scale_real(factor(w)?);
        }
        Ok(())
    }

    /// Visits every live term as `(key, &mut coefficient)`, in row order.
    ///
    /// The key is materialized from the store's packed row, so this is the seam
    /// for work needing both halves at once: the surrogate's compile pass takes
    /// each coefficient out and needs the key to compute the term's overlap
    /// with the reference state first.
    pub fn for_each_term_mut(&mut self, mut f: impl FnMut(Monomial<W>, &mut C)) {
        for i in 0..self.store.len() {
            let key = self.store.row(i);
            f(key, &mut self.coeffs[i]);
        }
    }

    /// Hands this partition's whole coefficient column to `f`.
    ///
    /// For a coefficient representation with internal structure to collapse (the
    /// surrogate's symbolic DAGs), where the work is per coefficient and wants a
    /// contiguous slice rather than a per-term callback.
    pub fn with_coeffs_mut(&mut self, f: impl FnOnce(&mut [C])) {
        let n = self.store.len();
        f(&mut self.coeffs[..n]);
    }

    /// Sums `measure` over every live coefficient.
    ///
    /// `u128` because the surrogate counts monomials, which a symbolic
    /// coefficient can carry exponentially many of.
    pub fn sum_coeffs(&self, measure: impl Fn(&C) -> u128) -> u128 {
        (0..self.store.len()).map(|i| measure(&self.coeffs[i])).sum()
    }

    /// Rebuilds the store from the terms `keep` admits, returning how many went.
    ///
    /// The store is otherwise append-only: a term is gated when it is emitted
    /// and nothing ever removes one, which is what lets the hash index store
    /// bare row indices and the inverted index stay incremental. That holds
    /// until a coefficient shrinks after the fact. Noise multiplies every
    /// coefficient down each layer, so without this the store fills with terms
    /// far below the cutoff that are still scanned and still held in memory.
    ///
    /// Everything keyed by row index therefore has to be rebuilt, not patched:
    /// a fresh `OperatorIndex` at the same inline width, a fresh coefficient
    /// column, and a reset inverted index that `sync_to` refills on the next
    /// scan. Callers should run this between gates; the pending and claimed
    /// state is per gate and is cleared here rather than remapped.
    pub fn reclaim(
        &mut self,
        keep: impl Fn(&Monomial<W>, &C) -> bool,
    ) -> Result<usize, TermIndexCeilingReached> {
        let before = self.store.len();
        let mut store = OperatorIndex::<P, W>::new(self.store.inline_width());
        let mut coeffs = Vec::with_capacity(self.coeffs.len());
        store.reserve(before);
        for i in 0..before {
            let key = self.store.row(i);
            if !keep(&key, &self.coeffs[i]) {
                continue;
            }
            let row = store.push(&key)?;
            store.insert_absent_with_hash(row, OperatorIndex::<P, W>::hash_of(&key))?;
            coeffs.push(self.coeffs[i].clone());
        }
        self.store = store;
        self.coeffs = coeffs;
        self.inverted.reset();
        // Row indices have moved, so anything addressed by one is now wrong.
        self.pending_slot.clear();
        self.pending_rows.clear();
        self.pending_vals.clear();
        self.claimed_rows.clear();
        Ok(before - self.store.len())
    }

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
        outbox: &mut [Vec<Routed<C, W>>],
    ) {
        self.reset_pending();
        // Once per gate, not once per term: `sin_cos` is a transcendental and
        // the angle is fixed for the gate.
        let factors = C::rotation_factors(param);
        let precheck = EmitPrecheck::for_gate(factors, cutoff);
        self.pending_needs_phase = precheck.is_some();

        // Fold the generator's columns into one bitmap and visit only the terms
        // that branch. The index is synced lazily: the store only grows, so rows
        // already indexed never need revisiting.
        self.inverted.sync_to(&self.store);
        let mut bitmap = std::mem::take(&mut self.scan_bitmap);
        self.inverted.combine(A::fold_generator(ctx).positions(), &mut bitmap);
        // The columns give `|M and G| mod 2`. A basis whose test also carries a
        // `|M|` term (Majorana under an odd-length generator) needs the row's own
        // key parity XORed in, which the index keeps as one more column. Before
        // this existed, such a gate fell back to a parity computation per live
        // term.
        if A::fold_needs_odd_correction(ctx) {
            self.inverted.apply_row_parity(&mut bitmap);
        }

        let mut rows = std::mem::take(&mut self.scan_rows);
        rows.clear();
        for_each_set_bit(&bitmap, |r| rows.push(r));
        for &i in rows.iter() {
            self.emit_from_row::<A>(i, ctx, param, cutoff, n_partitions, outbox, precheck, factors);
        }

        self.scan_bitmap = bitmap;
        self.scan_rows = rows;
    }

    /// Routes the held-back branches a partner has since paid for.
    ///
    /// A claim means the destination row is both a source of this gate and the
    /// child of a branch that cleared the cutoff, so the pair is wholly in the
    /// store and the rotation applies to it as a unit.
    pub fn drain_claims<A: Algebra<W>>(
        &mut self,
        ctx: &A::GenContext,
        n_partitions: usize,
        outbox: &mut [Vec<Routed<C, W>>],
    ) {
        let claims = std::mem::take(&mut self.claimed_rows);
        for &row in claims.iter() {
            let slot = self.pending_slot_of(row as usize).expect("a claimed row holds a branch");
            let (child, phase) = A::product(ctx, &self.store.row(row as usize));
            let hash = OperatorIndex::<P, W>::hash_of(&child);
            let mut coeff = self.pending_vals[slot].clone();
            if self.pending_needs_phase {
                // `-phase.im` is exactly +-1, so this rounds the same way
                // forming the branch in one multiplication would have.
                coeff.scale_real(-phase.im);
            }
            self.emitted += 1;
            outbox[partition_from_hash(hash, n_partitions)]
                .push(Routed { key: child, hash, coeff });
        }
        self.claimed_rows = claims;
        self.claimed_rows.clear();
    }

    /// True if a partner has paid for a held-back branch this gate.
    #[inline]
    pub fn has_claims(&self) -> bool {
        !self.claimed_rows.is_empty()
    }

    /// The entry a row's held-back branch sits in, if it still has one.
    #[inline]
    fn pending_slot_of(&self, row: usize) -> Option<usize> {
        match self.pending_slot.get(row) {
            Some(&slot) if slot != u32::MAX => Some(slot as usize),
            _ => None,
        }
    }

    /// Clears the previous gate's held-back branches, sizing the bitmap for
    /// this one. Rows added later in the gate fall past its end, which keeps a
    /// freshly created row from being mistaken for a held-back source.
    fn reset_pending(&mut self) {
        for &row in &self.pending_rows {
            self.pending_slot[row as usize] = u32::MAX;
        }
        self.pending_rows.clear();
        self.pending_vals.clear();
        self.claimed_rows.clear();
        if self.pending_slot.len() < self.store.len() {
            self.pending_slot.resize(self.store.len(), u32::MAX);
        }
    }

    /// Holds row `row`'s branch back for its partner to claim.
    #[inline]
    fn hold_back(&mut self, row: usize, coeff: C) {
        let Some(slot) = self.pending_slot.get_mut(row) else {
            return;
        };
        *slot = self.pending_rows.len() as u32;
        self.pending_rows.push(row as u32);
        self.pending_vals.push(coeff);
    }


    /// Notes that a partner has paid for row `row`'s held-back branch.
    ///
    /// A rejected branch only travels back when the pair rule is on; without it
    /// the cutoff's word on that branch stands.
    #[inline]
    fn claim(&mut self, row: usize) {
        if self.pending_slot_of(row).is_some() {
            self.claimed_rows.push(row as u32);
        }
    }

    /// Rotates row `i` into its cosine branch and either routes the sine branch
    /// or holds it back for the partner to claim.
    ///
    /// Split out so the indexed and per-term scans cannot drift apart. The
    /// cosine lands on every anticommuting row whatever happens to its sine
    /// half, so holding a branch back never changes the row it came from.
    #[inline]
    fn emit_from_row<A: Algebra<W>>(
        &mut self,
        i: usize,
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        n_partitions: usize,
        outbox: &mut [Vec<Routed<C, W>>],
        precheck: Option<EmitPrecheck>,
        factors: Option<(f64, f64)>,
    ) {
        self.visited += 1;
        // Decided from the source coefficient alone. The cosine branch still
        // lands, but it needs neither the row nor the product.
        if let Some(pre) = precheck {
            if pre.declines(&self.coeffs[i]) {
                self.declined += 1;
                // The source's own contribution, phase deferred to the claim.
                let mut held = self.coeffs[i].clone();
                held.scale_real(pre.sin);
                self.hold_back(i, held);
                self.coeffs[i].scale_real(pre.cos);
                return;
            }
        }
        let mono = self.store.row(i);
        debug_assert!(A::anticommutes(ctx, &mono), "a selected row must anticommute");
        let (child, phase) = A::product(ctx, &mono);
        debug_assert!(phase.re.abs() < 1e-9, "an anticommuting product must be purely imaginary");
        let sin_branch = self.coeffs[i].apply_rotation_with(param, factors, phase);
        // A child over the weight bound cannot be in the store, so no partner
        // can ever claim this branch and there is nothing to hold back.
        if !cutoff.admits_key::<A, W>(&child, self.n_units) {
            return;
        }
        if !cutoff.admits_child::<A, C, W>(&child, &sin_branch, self.n_units) {
            // Too small on its own, but a partner may yet pay for it: a pair
            // wholly in the store rotates as one unit, so this waits to be
            // claimed rather than being dropped.
            self.hold_back(i, sin_branch);
            self.declined += 1;
            return;
        }
        // The hash is needed to route the child anyway, so it travels with it
        // rather than being recomputed by the absorbing partition.
        let hash = OperatorIndex::<P, W>::hash_of(&child);
        self.emitted += 1;
        outbox[partition_from_hash(hash, n_partitions)].push(Routed { key: child, hash, coeff: sin_branch });
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
            self.hits += 1;
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

    /// Folds a routed child in, noting a claim if this row was holding a branch
    /// that the arriving child has now paid for.
    ///
    /// Returns true when a new row was created.
    pub fn absorb_routed(
        &mut self,
        msg: &Routed<C, W>,
    ) -> Result<bool, TermIndexCeilingReached> {
        if let Some(row) = self.store.find_with_hash(&msg.key, msg.hash) {
            self.hits += 1;
            self.coeffs[row].add_assign(msg.coeff.clone());
            self.coeffs[row].post_merge();
            self.claim(row);
            return Ok(false);
        }
        let row = self.store.push(&msg.key)?;
        self.store.insert_absent_with_hash(row, msg.hash)?;
        debug_assert_eq!(self.coeffs.len(), row);
        self.coeffs.push(msg.coeff.clone());
        Ok(true)
    }

    /// Children routed and children that landed on an existing row, cumulative.
    ///
    /// A hit means both halves of a pair were in the store, which is the case
    /// the pair-visiting scan settles with one probe instead of two.
    pub fn exchange_counts(&self) -> (u64, u64) {
        (self.emitted, self.hits)
    }

    /// Anticommuting rows the scan reached, and how many of those were declined.
    pub fn scan_counts(&self) -> (u64, u64) {
        (self.visited, self.declined)
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
        let mut added = 0usize;
        for msg in std::mem::take(&mut outbox[0]) {
            if self.absorb_routed(&msg)? {
                added += 1;
            }
        }
        if self.has_claims() {
            self.drain_claims::<A>(&ctx, 1, &mut outbox);
            for msg in std::mem::take(&mut outbox[0]) {
                let created = self.absorb_routed(&msg)?;
                debug_assert!(!created, "a claimed branch lands on a row of this gate");
            }
        }
        Ok(added)
    }

    /// Expectation value against a computational basis state.
    pub fn expectation<A: Algebra<W>>(&self, fock: &[u64]) -> f64 {
        (0..self.store.len())
            .map(|i| self.coeffs[i].to_f64() * A::trace(&self.store.row(i), self.n_units, fock))
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
        fn weight(mono: &Monomial<W>, _n_units: usize) -> u32 {
            mono.count() as u32
        }
        fn trace(mono: &Monomial<W>, _n_units: usize, fock: &[u64]) -> f64 {
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
        let cutoff = EmitCutoff { max_weight: Some(2), ..Default::default() };
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[2]), &0.3, &cutoff).unwrap();
        assert_eq!(added, 0, "the over-weight child must never be created");
        assert_eq!(op.len(), 1);
    }

    #[test]
    fn a_coefficient_cutoff_suppresses_a_small_child_at_emit_time() {
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1e-12).unwrap();
        let cutoff = EmitCutoff { min_coeff: Some(1e-6), ..Default::default() };
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &cutoff).unwrap();
        assert_eq!(added, 0, "the tiny child must never be created");
        assert_eq!(op.len(), 1);
    }

    #[test]
    // There is no "rule off" counterpart to these any more: the pair rule is the
    // engine's truncation semantics rather than a switch, so what used to be an
    // A/B is now just the specification. The three tests below pin it: a pair
    // whose partner paid rotates as a unit, one neither half earned does not,
    // and a partner that is absent cannot pay.
    fn a_pair_rescue_keeps_the_branch_its_partner_paid_for() {
        // Rows 0 and 1 are each other's image under the generator, so the gate
        // rotates them as a pair. The small one's branch is far below the
        // cutoff and the large one's is far above it.
        let angle = 0.3f64;
        let (big, small) = (1.0f64, 1e-9f64);
        let cutoff = EmitCutoff { min_coeff: Some(1e-6), ..Default::default() };
        let mut op = Op::new(8);
        op.add(&mono(&[0]), big).unwrap();
        op.add(&mono(&[1]), small).unwrap();
        op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &cutoff).unwrap();

        let v = values(&op);
        let (sin_t, cos_t) = angle.sin_cos();
        assert_eq!(op.len(), 2, "a rescue must not create a row");
        assert!(
            (v[&0b01] - (big * cos_t + small * sin_t * -1.0)).abs() < 1e-18,
            "the rejected branch is owed back to its partner, got {}",
            v[&0b01]
        );
        assert!((v[&0b10] - (small * cos_t + big * sin_t * -1.0)).abs() < 1e-15, "the cleared branch");
    }

    #[test]
    fn a_term_floor_suppresses_the_lossy_predicates_while_the_store_is_small() {
        // `TermBudget::min_terms`. Below the floor nothing may be refused, so a
        // branch the coefficient bound would drop has to survive.
        let angle = 0.3f64;
        let biting = EmitCutoff { min_coeff: Some(1e-6), ..Default::default() };
        assert!(biting.at_size(1).min_coeff.is_some(), "no floor set, so nothing is suppressed");

        let floored = EmitCutoff { min_coeff: Some(1e-6), min_terms: Some(1000), ..Default::default() };
        assert!(floored.at_size(10).min_coeff.is_none(), "below the floor the bound must be off");
        assert!(floored.at_size(1000).min_coeff.is_some(), "at the floor the bound comes back");

        // And end to end. The generator must overlap the term oddly or nothing
        // branches at all and the test would pass for the wrong reason.
        let tiny = 1e-9f64;
        let gen = mono(&[0]);
        let mut with_floor = Op::new(8);
        with_floor.add(&mono(&[0]), tiny).unwrap();
        let effective = floored.at_size(with_floor.len());
        with_floor.apply_rotation::<TestAlgebra>(&gen, &angle, &effective).unwrap();
        assert_eq!(with_floor.len(), 2, "below the floor the tiny branch must survive");

        let mut no_floor = Op::new(8);
        no_floor.add(&mono(&[0]), tiny).unwrap();
        let effective = biting.at_size(no_floor.len());
        no_floor.apply_rotation::<TestAlgebra>(&gen, &angle, &effective).unwrap();
        assert_eq!(no_floor.len(), 1, "without a floor the same branch must be refused");
    }

    #[test]
    fn a_term_floor_carries_through_a_lossless_copy() {
        // `lossless` must keep the floor itself, or the suppression would
        // un-suppress on the next gate and oscillate.
        let c = EmitCutoff {
            max_weight: Some(3),
            min_coeff: Some(1e-6),
            min_terms: Some(50),
            ..Default::default()
        };
        let l = c.lossless();
        assert_eq!(l.min_terms, Some(50));
        assert!(l.max_weight.is_none() && l.min_coeff.is_none() && l.native.is_none());
    }

    #[test]
    fn a_pair_rescue_does_not_revive_a_branch_neither_half_earned() {
        // Both branches are below the cutoff, so there is nothing to rescue:
        // the pair is only scaled by its cosines.
        let angle = 0.3f64;
        let cutoff = EmitCutoff { min_coeff: Some(1e-6), ..Default::default() };
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1e-9).unwrap();
        op.add(&mono(&[1]), 2e-9).unwrap();
        op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &cutoff).unwrap();

        let v = values(&op);
        let cos_t = angle.cos();
        assert_eq!(op.len(), 2);
        assert!((v[&0b01] - 1e-9 * cos_t).abs() < 1e-24);
        assert!((v[&0b10] - 2e-9 * cos_t).abs() < 1e-24);
    }

    #[test]
    fn a_pair_rescue_needs_both_halves_in_the_store() {
        // The partner is absent, so the small branch has no pair to belong to
        // and the cutoff stands.
        let cutoff = EmitCutoff { min_coeff: Some(1e-6), ..Default::default() };
        let mut op = Op::new(8);
        op.add(&mono(&[0]), 1e-9).unwrap();
        let added = op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &cutoff).unwrap();
        assert_eq!(added, 0, "a lone tiny term still may not create a child");
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
