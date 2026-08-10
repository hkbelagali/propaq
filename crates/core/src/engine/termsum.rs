//!
//! Sparse representation of $\sum_i c_i B_i$ where $B_i$ is
//! a basis element represented by a `BasisString<W>`, and
//! $c_i$ is a coefficient represented by a `CoeffRepr`.
//!

use std::sync::Arc;

use num_complex::Complex64;

use crate::basis::Basis;
use crate::coeff::CoeffRepr;
use crate::inverted_index::{for_each_set_bit, InvertedIndex};
use crate::native_truncator::NativeTruncator;
use crate::operator_index::{OperatorIndex, Pos, TermIndexCeilingReached};
use crate::strings::BasisString;
use crate::term_kernel::{NoiseKernel, TermView, TruncationKernel, KERNEL_BATCH};
use crate::truncators::ResolvedConfig;

/// Tolerance for treating `$\sin(\theta)$` as zero when classifying a rotation.
const PHASE_ONLY_EPS: f64 = 1e-9;

/// Truncation predicates for the emit gate.
#[derive(Clone, Default)]
pub struct EmitCutoff {
    /// Structural weight bound. A child above it is never created.
    pub max_weight: Option<u32>,
    /// Magnitude bound on the emitted branch. A child below it is never created.
    pub min_coeff: Option<f64>,
    /// A loaded plugin f(w) for scalar parameters (weight, cutoff, active modes).
    pub native: Option<NativeTruncator>,
    /// A term-aware plugin, using the term's key as a parameter.
    pub term: Option<Arc<dyn TruncationKernel>>,
    /// Live-term floor, from `TermBudget::min_terms`. Lossy truncation is
    /// inhibited until the operator has at least this many terms.
    pub min_terms: Option<usize>,
}

impl EmitCutoff {
    pub fn lossless(&self) -> Self {
        EmitCutoff {
            min_terms: self.min_terms,
            ..Default::default()
        }
    }

    /// Apply cutoff to a term sum with `n_live` terms.
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
            .field("term", &self.term.is_some())
            .field("min_terms", &self.min_terms)
            .finish()
    }
}

/// The cutoff a resolved truncation pipeline asks the emit gate for.
impl From<&ResolvedConfig> for EmitCutoff {
    fn from(cfg: &ResolvedConfig) -> Self {
        let (native, term) = match cfg.native.as_ref() {
            Some(nt) => match nt.as_term_kernel() {
                Some(kernel) => (None, Some(kernel)),
                None => (Some(nt.clone()), None),
            },
            None => (None, None),
        };
        EmitCutoff {
            max_weight: cfg.weight,
            min_coeff: cfg.coefficient,
            native,
            term,
            min_terms: cfg.min_terms,
        }
    }
}

impl EmitCutoff {
    /// No truncation at all.
    pub fn none() -> Self {
        Self::default()
    }

    /// True if a child with this key may be created at all.
    #[inline]
    pub(crate) fn admits_key<A: Basis<W>, const W: usize>(
        &self,
        key: &BasisString<W>,
        n_units: usize,
    ) -> bool {
        if self.native.is_some() || self.term.is_some() {
            return true;
        }
        match self.max_weight {
            Some(w) => A::weight(key, n_units) <= w,
            None => true,
        }
    }

    #[inline]
    pub(crate) fn admits_child<A: Basis<W>, C: CoeffRepr, const W: usize>(
        &self,
        key: &BasisString<W>,
        coeff: &C,
        n_units: usize,
    ) -> bool {
        if let Some(kernel) = &self.term {
            return kernel.keep(
                TermView {
                    basis_kind: A::KIND,
                    words: key.words(),
                    n_units,
                    weight: A::weight(key, n_units),
                },
                coeff.magnitude(),
            );
        }
        match &self.native {
            Some(nt) => nt.keep(A::weight(key, n_units), coeff.magnitude(), 0),
            None => self.admits_coeff(coeff),
        }
    }

    /// True if a term belongs in the term sum.
    pub fn admits_initial<A: Basis<W>, C: CoeffRepr, const W: usize>(
        &self,
        key: &BasisString<W>,
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

#[derive(Clone, Copy)]
pub(crate) struct EmitPrecheck {
    sin: f64,
    cos: f64,
    min_coeff: f64,
}

impl EmitPrecheck {
    pub(crate) fn for_gate(factors: Option<(f64, f64)>, cutoff: &EmitCutoff) -> Option<Self> {
        if cutoff.native.is_some() || cutoff.term.is_some() {
            return None;
        }
        let min_coeff = cutoff.min_coeff?;
        let (sin, cos) = factors?;
        Some(EmitPrecheck {
            sin,
            cos,
            min_coeff,
        })
    }

    /// True when this source cannot produce a branch that survives the cutoff.
    #[inline]
    fn declines<C: CoeffRepr>(&self, coeff: &C) -> bool {
        coeff.sin_branch_magnitude(self.sin) < self.min_coeff
    }
}

/// A routed child term, with its key, hash, and coefficient.
#[derive(Clone)]
pub struct Routed<C, const W: usize> {
    pub key: BasisString<W>,
    pub hash: u64,
    pub coeff: C,
}

/// Represent a sum of terms.
pub struct TermSum<C: CoeffRepr, P: Pos, const W: usize> {
    store: OperatorIndex<P, W>,
    coeffs: Vec<C>,
    /// Transposed view of the term sum.
    inverted: InvertedIndex,
    scan_bitmap: Vec<u64>,
    scan_rows: Vec<usize>,
    pending_slot: Vec<u32>,
    pending_rows: Vec<u32>,
    pending_vals: Vec<C>,
    claimed_rows: Vec<u32>,
    /// True when the pending rows need a phase factor applied, relevant for the sin branch.
    pending_needs_phase: bool,
    emitted: u64,
    hits: u64,
    visited: u64,
    declined: u64,
    n_units: usize,
}

impl<C: CoeffRepr, P: Pos, const W: usize> TermSum<C, P, W> {
    /// Creates an empty operator over `n_units` qubits or modes.
    pub fn new(n_units: usize) -> Self {
        TermSum {
            store: OperatorIndex::with_default_width(),
            coeffs: Vec::new(),
            inverted: InvertedIndex::new(BasisString::<W>::num_bits()),
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
    pub fn with_inline_positions(n_units: usize, width: usize) -> Self {
        TermSum {
            store: OperatorIndex::new(width),
            coeffs: Vec::new(),
            inverted: InvertedIndex::new(BasisString::<W>::num_bits()),
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
    pub fn repack(&mut self, inline_width: usize) -> Result<(), TermIndexCeilingReached> {
        let mut store = OperatorIndex::<P, W>::new(inline_width);
        store.reserve(self.store.len());
        for i in 0..self.store.len() {
            let key = self.store.row(i);
            let row = store.push(&key)?;
            store.insert_absent_with_hash(row, OperatorIndex::<P, W>::hash_of(&key))?;
        }
        self.store = store;
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
    pub fn scale_by_weight<A: Basis<W>>(&mut self, factor: impl Fn(u32) -> f64) {
        for i in 0..self.store.len() {
            let w = A::weight(&self.store.row(i), self.n_units);
            self.coeffs[i].scale_real(factor(w));
        }
    }

    pub fn try_scale_by_weight<A: Basis<W>, E>(
        &mut self,
        mut factor: impl FnMut(u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for i in 0..self.store.len() {
            let w = A::weight(&self.store.row(i), self.n_units);
            self.coeffs[i].scale_real(factor(w)?);
        }
        Ok(())
    }

    /// Scales every coefficient by a kernel's factor for that term's key.
    pub fn scale_by_key<A: Basis<W>>(&mut self, kernel: &dyn NoiseKernel) {
        let n = self.store.len();
        if n == 0 {
            return;
        }
        let chunk = KERNEL_BATCH.min(n);
        let mut words = vec![0u64; chunk * W];
        let mut weights = vec![0u32; chunk];
        let mut factors = vec![0f64; chunk];
        let mut start = 0usize;
        while start < n {
            let len = chunk.min(n - start);
            for j in 0..len {
                let key = self.store.row(start + j);
                words[j * W..(j + 1) * W].copy_from_slice(key.words());
                weights[j] = A::weight(&key, self.n_units);
            }
            kernel.factor_batch(
                A::KIND,
                &words[..len * W],
                W,
                &weights[..len],
                self.n_units,
                &mut factors[..len],
            );
            for (j, &factor) in factors[..len].iter().enumerate() {
                self.coeffs[start + j].scale_real(factor);
            }
            start += len;
        }
    }

    /// [`TermSum::scale_by_key`] for a factor that can fail.
    pub fn try_scale_by_key<A: Basis<W>, E>(
        &mut self,
        mut factor: impl FnMut(&BasisString<W>, u32) -> Result<f64, E>,
    ) -> Result<(), E> {
        for i in 0..self.store.len() {
            let key = self.store.row(i);
            let w = A::weight(&key, self.n_units);
            self.coeffs[i].scale_real(factor(&key, w)?);
        }
        Ok(())
    }

    /// Batched reclaim of terms by a kernel.
    pub fn reclaim_by_kernel<A: Basis<W>>(
        &mut self,
        kernel: &dyn TruncationKernel,
    ) -> Result<usize, TermIndexCeilingReached> {
        let n = self.store.len();
        if n == 0 {
            return Ok(0);
        }
        let chunk = KERNEL_BATCH.min(n);
        let mut words = vec![0u64; chunk * W];
        let mut weights = vec![0u32; chunk];
        let mut magnitudes = vec![0f64; chunk];
        let mut decided = vec![0u8; chunk];
        let mut flags = Vec::with_capacity(n);
        let n_units = self.n_units;
        let mut start = 0usize;
        while start < n {
            let len = chunk.min(n - start);
            for j in 0..len {
                let key = self.store.row(start + j);
                words[j * W..(j + 1) * W].copy_from_slice(key.words());
                weights[j] = A::weight(&key, n_units);
                magnitudes[j] = self.coeffs[start + j].magnitude();
            }
            let batched = kernel.keep_batch(
                A::KIND,
                &words[..len * W],
                W,
                &weights[..len],
                n_units,
                &magnitudes[..len],
                &mut decided[..len],
            );
            if batched {
                flags.extend(decided[..len].iter().map(|&k| k != 0));
            } else {
                for j in 0..len {
                    flags.push(kernel.keep(
                        TermView {
                            basis_kind: A::KIND,
                            words: &words[j * W..(j + 1) * W],
                            n_units,
                            weight: weights[j],
                        },
                        magnitudes[j],
                    ));
                }
            }
            start += len;
        }

        let mut i = 0usize;
        self.reclaim(|_, _| {
            let keep = flags[i];
            i += 1;
            keep
        })
    }

    /// Visits every live term as `(key, &mut coefficient)`, in row order.
    pub fn for_each_term_mut(&mut self, mut f: impl FnMut(BasisString<W>, &mut C)) {
        for i in 0..self.store.len() {
            let key = self.store.row(i);
            f(key, &mut self.coeffs[i]);
        }
    }

    /// Hands this partition's whole coefficient column to `f`.
    pub fn with_coeffs_mut(&mut self, f: impl FnOnce(&mut [C])) {
        let n = self.store.len();
        f(&mut self.coeffs[..n]);
    }

    /// Sums `measure` over every live coefficient.
    pub fn sum_coeffs(&self, measure: impl Fn(&C) -> u128) -> u128 {
        (0..self.store.len())
            .map(|i| measure(&self.coeffs[i]))
            .sum()
    }

    /// Rebuilds the term sum from the admitted terms.
    /// Returns the number of terms that were dropped.
    pub fn reclaim(
        &mut self,
        mut keep: impl FnMut(&BasisString<W>, &C) -> bool,
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
    pub fn key(&self, i: usize) -> BasisString<W> {
        self.store.row(i)
    }

    /// Row `i`'s coefficient.
    #[inline]
    pub fn coeff(&self, i: usize) -> &C {
        &self.coeffs[i]
    }

    /// The underlying store
    #[inline]
    pub fn store(&self) -> &OperatorIndex<P, W> {
        &self.store
    }

    /// Adds `coeff` to `key`'s term, creating the term if it is absent.
    pub fn add(&mut self, key: &BasisString<W>, coeff: C) -> Result<(), TermIndexCeilingReached> {
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

    pub fn key_bytes(&self) -> usize {
        self.store.memory_bytes() + self.store.index_memory_bytes()
    }

    /// Scales every anticommuting term by `factor`
    pub fn scale_anticommuting<A: Basis<W>>(&mut self, ctx: &A::GenContext, factor: f64) {
        for i in 0..self.store.len() {
            if A::anticommutes(ctx, &self.store.row(i)) {
                self.coeffs[i].scale_real(factor);
            }
        }
    }

    /// Rotates every source into its cosine branch and routes the sine branches
    /// into `outbox`, bucketed by the owner partition
    pub fn scan_into<A: Basis<W>>(
        &mut self,
        ctx: &A::GenContext,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
        n_partitions: usize,
        outbox: &mut [Vec<Routed<C, W>>],
    ) {
        self.reset_pending();
        let factors = C::rotation_factors(param);
        let precheck = EmitPrecheck::for_gate(factors, cutoff);
        self.pending_needs_phase = precheck.is_some();

        self.inverted.sync_to(&self.store);
        let mut bitmap = std::mem::take(&mut self.scan_bitmap);
        self.inverted
            .combine(A::fold_generator(ctx).positions(), &mut bitmap);

        if A::fold_needs_odd_correction(ctx) {
            self.inverted.apply_row_parity(&mut bitmap);
        }

        let mut rows = std::mem::take(&mut self.scan_rows);
        rows.clear();
        for_each_set_bit(&bitmap, |r| rows.push(r));
        for &i in rows.iter() {
            self.emit_from_row::<A>(
                i,
                ctx,
                param,
                cutoff,
                n_partitions,
                outbox,
                precheck,
                factors,
            );
        }

        self.scan_bitmap = bitmap;
        self.scan_rows = rows;
    }

    /// Drain the claimed branches into `outbox`.
    pub fn drain_claims<A: Basis<W>>(
        &mut self,
        ctx: &A::GenContext,
        n_partitions: usize,
        outbox: &mut [Vec<Routed<C, W>>],
    ) {
        let claims = std::mem::take(&mut self.claimed_rows);
        for &row in claims.iter() {
            let slot = self
                .pending_slot_of(row as usize)
                .expect("a claimed row holds a branch");
            let (child, phase) = A::product(ctx, &self.store.row(row as usize));
            let hash = OperatorIndex::<P, W>::hash_of(&child);
            let mut coeff = self.pending_vals[slot].clone();
            if self.pending_needs_phase {
                coeff.scale_real(-phase.im);
            }
            self.emitted += 1;
            outbox[partition_from_hash(hash, n_partitions)].push(Routed {
                key: child,
                hash,
                coeff,
            });
        }
        self.claimed_rows = claims;
        self.claimed_rows.clear();
    }

    /// True if a partner has paid for a held-back branch this gate.
    #[inline]
    pub fn has_claims(&self) -> bool {
        !self.claimed_rows.is_empty()
    }

    /// The entry a row's held-back branch sits in
    #[inline]
    fn pending_slot_of(&self, row: usize) -> Option<usize> {
        match self.pending_slot.get(row) {
            Some(&slot) if slot != u32::MAX => Some(slot as usize),
            _ => None,
        }
    }

    /// Clears the previous gate's held-back branches and
    /// resize the pending slot.
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
    #[inline]
    fn claim(&mut self, row: usize) {
        if self.pending_slot_of(row).is_some() {
            self.claimed_rows.push(row as u32);
        }
    }

    /// Rotates row `i` into its cosine branch and either routes the sine branch
    /// or holds it back for the partner to claim.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn emit_from_row<A: Basis<W>>(
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
        if let Some(pre) = precheck {
            if pre.declines(&self.coeffs[i]) {
                self.declined += 1;
                let mut held = self.coeffs[i].clone();
                held.scale_real(pre.sin);
                self.hold_back(i, held);
                self.coeffs[i].scale_real(pre.cos);
                return;
            }
        }
        let mono = self.store.row(i);
        debug_assert!(
            A::anticommutes(ctx, &mono),
            "a selected row must anticommute"
        );
        let (child, phase) = A::product(ctx, &mono);
        debug_assert!(
            phase.re.abs() < 1e-9,
            "an anticommuting product must be purely imaginary"
        );
        let sin_branch = self.coeffs[i].apply_rotation_with(param, factors, phase);

        if !cutoff.admits_key::<A, W>(&child, self.n_units) {
            return;
        }
        if !cutoff.admits_child::<A, C, W>(&child, &sin_branch, self.n_units) {
            self.hold_back(i, sin_branch);
            self.declined += 1;
            return;
        }

        let hash = OperatorIndex::<P, W>::hash_of(&child);
        self.emitted += 1;
        outbox[partition_from_hash(hash, n_partitions)].push(Routed {
            key: child,
            hash,
            coeff: sin_branch,
        });
    }

    /// Folds one routed child into this partition, appending it if absent.
    pub fn absorb(
        &mut self,
        key: &BasisString<W>,
        coeff: C,
    ) -> Result<bool, TermIndexCeilingReached> {
        let hash = OperatorIndex::<P, W>::hash_of(key);
        self.absorb_with_hash(key, hash, coeff)
    }

    /// Issues a prefetch for the table slot `hash` will probe.
    #[inline]
    pub fn prefetch(&self, hash: u64) {
        self.store.prefetch_for_hash(hash);
    }

    /// [`TermSum::absorb`] with the key's hash already computed.
    pub fn absorb_with_hash(
        &mut self,
        key: &BasisString<W>,
        hash: u64,
        coeff: C,
    ) -> Result<bool, TermIndexCeilingReached> {
        if let Some(row) = self.store.find_with_hash(key, hash) {
            self.hits += 1;
            self.coeffs[row].add_assign(coeff);
            return Ok(false);
        }
        let row = self.store.push(key)?;
        self.store.insert_absent_with_hash(row, hash)?;
        debug_assert_eq!(self.coeffs.len(), row);
        self.coeffs.push(coeff);
        Ok(true)
    }

    pub fn absorb_routed(&mut self, msg: &Routed<C, W>) -> Result<bool, TermIndexCeilingReached> {
        if let Some(row) = self.store.find_with_hash(&msg.key, msg.hash) {
            self.hits += 1;
            self.coeffs[row].add_assign(msg.coeff.clone());
            self.claim(row);
            return Ok(false);
        }
        let row = self.store.push(&msg.key)?;
        self.store.insert_absent_with_hash(row, msg.hash)?;
        debug_assert_eq!(self.coeffs.len(), row);
        self.coeffs.push(msg.coeff.clone());
        Ok(true)
    }

    pub fn exchange_counts(&self) -> (u64, u64) {
        (self.emitted, self.hits)
    }

    pub fn scan_counts(&self) -> (u64, u64) {
        (self.visited, self.declined)
    }

    pub fn apply_rotation<A: Basis<W>>(
        &mut self,
        gen: &BasisString<W>,
        param: &C::GateParam,
        cutoff: &EmitCutoff,
    ) -> Result<usize, TermIndexCeilingReached> {
        if self.store.is_empty() {
            return Ok(0);
        }
        let ctx = A::make_gen_context(gen);

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
    pub fn expectation<A: Basis<W>>(&self, fock: &[u64]) -> f64 {
        (0..self.store.len())
            .map(|i| self.coeffs[i].to_f64() * A::trace(&self.store.row(i), self.n_units, fock))
            .sum()
    }

    /// Every live term as a key and coefficient pair.
    pub fn iter(&self) -> impl Iterator<Item = (BasisString<W>, &C)> + '_ {
        (0..self.store.len()).map(move |i| (self.store.row(i), &self.coeffs[i]))
    }
}

/// The partition that owns `key`.
#[inline]
pub fn partition_of<const W: usize>(key: &BasisString<W>, n_partitions: usize) -> usize {
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

pub fn imaginary_phase(sign: f64) -> Complex64 {
    Complex64::new(0.0, sign)
}

#[cfg(test)]
#[path = "../../tests/unit/engine/termsum.rs"]
mod tests;
