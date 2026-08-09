///
/// Structure-of-Arrays term storage shared by the numerical and surrogate
/// propagators.
///
/// Term keys are stored sparsely: each row is the sorted list of positions of
/// its set bits (see [`sparse::SparseRows`]). Dense word planes exist only as
/// short-lived, explicitly-owned workspaces borrowed by a kernel for the
/// duration of one call.
///
pub mod kernels;
pub mod propagator;
pub mod sparse;

use std::sync::atomic::{AtomicU8, Ordering};

use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

use crate::coeff::CoeffRepr;

pub use sparse::{
    reset_workspace_peak, workspace_peak_bytes, DenseWorkspace, Position, SparseRows,
};

/// Chunk size floor below which a pass runs serially.
pub const PAR_MIN_LEN: usize = 512;

/// A raw pointer that rayon tasks may carry into disjoint-index scatters.
pub(crate) struct SendPtr<T>(pub(crate) *mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    pub(crate) unsafe fn add(&self, idx: usize) -> *mut T {
        unsafe { self.0.add(idx) }
    }
}

/// Which per-row strategy the kernels use to reach a basis operation.
///
/// Storage is sparse either way; this only selects how a kernel gets at a row's
/// algebraic content. It exists so an A/B run can attribute a timing difference
/// to the kernels rather than to the storage change.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelLayout {
    /// Operate on position lists directly.
    Sparse,
    /// Decode each row into a per-worker dense workspace and use the word-plane
    /// basis methods.
    Dense,
}

const LAYOUT_UNSET: u8 = 0;
const LAYOUT_SPARSE: u8 = 1;
const LAYOUT_DENSE: u8 = 2;

static KERNEL_LAYOUT: AtomicU8 = AtomicU8::new(LAYOUT_UNSET);

/// The active kernel layout, resolved once from `PROPAQ_SOA_LAYOUT`.
///
/// Anything other than `dense` selects the sparse kernels.
pub fn kernel_layout() -> KernelLayout {
    match KERNEL_LAYOUT.load(Ordering::Relaxed) {
        LAYOUT_SPARSE => KernelLayout::Sparse,
        LAYOUT_DENSE => KernelLayout::Dense,
        _ => {
            let layout = match std::env::var("PROPAQ_SOA_LAYOUT").as_deref() {
                Ok("dense") => KernelLayout::Dense,
                _ => KernelLayout::Sparse,
            };
            set_kernel_layout(layout);
            layout
        }
    }
}

/// Overrides the layout `PROPAQ_SOA_LAYOUT` would have selected.
///
/// Intended for A/B tests that need to drive both kernel paths in one process.
pub fn set_kernel_layout(layout: KernelLayout) {
    let encoded = match layout {
        KernelLayout::Sparse => LAYOUT_SPARSE,
        KernelLayout::Dense => LAYOUT_DENSE,
    };
    KERNEL_LAYOUT.store(encoded, Ordering::Relaxed);
}

/// Splits a row into its plane-0 and plane-1 position slices.
#[inline]
pub fn split_planes(row: &[Position], plane_span: usize) -> (&[Position], &[Position]) {
    row.split_at(row.partition_point(|&p| (p as usize) < plane_span))
}

/// Decodes one row into a stack-resident buffer and hands its word planes to `f`.
///
/// Only ever one row wide, so it is bounded regardless of term count; the
/// chunked [`DenseWorkspace`] is what the `Dense` kernel layout uses.
fn with_decoded<R>(row: &[Position], plane_span: usize, f: impl FnOnce([&[u64]; 2]) -> R) -> R {
    let stride = plane_span / 64;
    let mut buf: SmallVec<[u64; 8]> = smallvec![0u64; 2 * stride];
    let (a, b) = buf.split_at_mut(stride);
    sparse::decode_row_into(row, plane_span, [a, b]);
    f([&*a, &*b])
}

/// Decodes two rows into one stack-resident buffer and hands both to `f`.
fn with_decoded2<R>(
    a_row: &[Position],
    b_row: &[Position],
    plane_span: usize,
    f: impl FnOnce([&[u64]; 2], [&[u64]; 2]) -> R,
) -> R {
    let stride = plane_span / 64;
    let mut buf: SmallVec<[u64; 16]> = smallvec![0u64; 4 * stride];
    let (first, second) = buf.split_at_mut(2 * stride);
    let (a0, a1) = first.split_at_mut(stride);
    let (b0, b1) = second.split_at_mut(stride);
    sparse::decode_row_into(a_row, plane_span, [a0, a1]);
    sparse::decode_row_into(b_row, plane_span, [b0, b1]);
    f([&*a0, &*a1], [&*b0, &*b1])
}

/// Hash of a position list, used as the default sparse key hash.
pub fn hash_positions(row: &[Position]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    row.hash(&mut h);
    h.finish()
}

/// The algebra a `SoaTermSum` needs from its term representation (Pauli, Majorana, etc.) to
/// run the shared kernels in `soa::kernels`.
///
/// The word-plane methods define the algebra. The `*_sparse` methods are what
/// the kernels actually call on the hot path; their defaults decode a single
/// row and delegate, so a basis only overrides the ones worth doing directly on
/// positions.
pub trait SoaBasis: Send + Sync + 'static {
    /// The owned, per-term representation used at the Python/FFI boundary (e.g. `PauliString`).
    type Term: Clone + Send + Sync;

    /// Number of `u64` words needed to store one term's plane for `n_units` qubits/modes.
    fn stride_words(n_units: usize) -> usize {
        let width = n_units.next_power_of_two().max(1);
        width.div_ceil(64)
    }

    /// True if `term` commutes with generator `gen`.
    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool;

    /// Computes `gen * term`, writing the result into `out` and returning its phase factor.
    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64;

    /// The term's weight (number of non-identity single-qubit/mode factors).
    fn weight(term: [&[u64]; 2], n_units: usize) -> u32;

    /// The term's expectation value trace against a computational basis state `fock`.
    fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64;

    /// Hash of `term`'s key (its algebraic content, ignoring any coefficient), for the merge
    /// hash table. Must agree with `key_eq`.
    fn key_hash(term: [&[u64]; 2]) -> u64;

    /// True if `a` and `b` have identical key content. Must agree with `key_hash`.
    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool;

    /// Reconstructs the owned `Self::Term` from its word planes.
    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> Self::Term;

    /// Writes `term`'s word planes into `out`.
    fn term_into_planes(term: &Self::Term, n_units: usize, out: [&mut [u64]; 2]);

    fn local_word(_gen: [&[u64]; 2]) -> Option<usize> {
        None
    }

    fn commutes_at_word(_term_word: [u64; 2], _gen_word: [u64; 2]) -> bool {
        unimplemented!("commutes_at_word must be implemented whenever local_word can return Some")
    }

    fn product_at_word(_term_word: [u64; 2], _gen_word: [u64; 2]) -> ([u64; 2], Complex64) {
        unimplemented!("product_at_word must be implemented whenever local_word can return Some")
    }

    /// The weight of a term given as a sparse row.
    ///
    /// The default is the number of distinct units the two planes touch, which
    /// is the usual definition; a basis whose weight is not a per-unit popcount
    /// (Majorana's Jordan-Wigner weight, for instance) overrides this.
    fn weight_sparse(row: &[Position], plane_span: usize, _n_units: usize) -> u32 {
        let (p0, p1) = split_planes(row, plane_span);
        sparse::shifted_union_count(p0, p1, plane_span)
    }

    /// The trace of a term given as a sparse row.
    fn trace_sparse(row: &[Position], plane_span: usize, n_units: usize, fock: &[u64]) -> f64 {
        with_decoded(row, plane_span, |t| Self::trace(t, n_units, fock))
    }

    /// Hash of a sparse row's key. Must agree with `key_eq_sparse`.
    fn key_hash_sparse(row: &[Position], _plane_span: usize) -> u64 {
        hash_positions(row)
    }

    /// True if two sparse rows have identical key content.
    fn key_eq_sparse(a: &[Position], b: &[Position], _plane_span: usize) -> bool {
        a == b
    }

    /// True if the sparse row `term` commutes with the sparse generator `gen`.
    fn commutes_sparse(term: &[Position], gen: &[Position], plane_span: usize) -> bool {
        with_decoded2(term, gen, plane_span, |t, g| Self::commutes(t, g))
    }

    /// Appends the sparse row of `gen * term` to `out` and returns its phase factor.
    fn product_sparse(
        term: &[Position],
        gen: &[Position],
        plane_span: usize,
        out: &mut Vec<Position>,
    ) -> Complex64 {
        let stride = plane_span / 64;
        let mut result: SmallVec<[u64; 8]> = smallvec![0u64; 2 * stride];
        let phase = with_decoded2(term, gen, plane_span, |t, g| {
            let (r0, r1) = result.split_at_mut(stride);
            Self::product(t, g, [r0, r1])
        });
        let (r0, r1) = result.split_at(stride);
        sparse::encode_planes_into([r0, r1], plane_span, out);
        phase
    }
}

/// Structure-of-Arrays storage for a sum of terms.
///
/// Keys live in `rows` as sparse position lists; coefficients and merge
/// metadata stay row-aligned with them. There is no persistent dense plane
/// storage.
pub struct SoaTermSum<C: CoeffRepr> {
    /// Sparse, row-major term keys. Authoritative: nothing else stores a key.
    rows: SparseRows,

    /// Per-row coefficients, parallel to `rows`.
    pub coeffs: Vec<C>,

    aux_coeffs: Vec<C>,

    flags: Vec<u32>,
    index: Vec<usize>,

    hashes: Vec<u64>,
    // Double-buffer for `hashes`
    aux_hashes: Vec<u64>,

    // One reusable hash table per merge batch
    merge_tables: Vec<hashbrown::HashTable<usize>>,

    merge_synced_len: usize,
    /// Number of `u64` words one decoded row occupies per plane.
    pub stride: usize,
    /// Number of qubits (Pauli) or modes (Majorana) this term sum's rows are sized for.
    pub n_units: usize,
}

impl<C: CoeffRepr> SoaTermSum<C> {
    /// Creates an empty term sum sized for `n_units` qubits/modes at the given `stride`
    pub fn new(n_units: usize, stride: usize) -> Self {
        SoaTermSum {
            rows: SparseRows::new(stride),
            coeffs: Vec::new(),
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            stride,
            n_units,
        }
    }

    /// Number of live rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    fn cap(&self) -> usize {
        self.coeffs.len()
    }

    /// True if there are no live rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The sparse key rows.
    #[inline]
    pub fn rows(&self) -> &SparseRows {
        &self.rows
    }

    /// The sparse key rows alongside the live coefficient column, for passes
    /// that read every key while rewriting its coefficient.
    pub fn rows_and_coeffs_mut(&mut self) -> (&SparseRows, &mut [C]) {
        let n = self.rows.len();
        (&self.rows, &mut self.coeffs[..n])
    }

    /// Row `i`'s ascending set-bit positions.
    #[inline]
    pub fn row_positions(&self, i: usize) -> &[Position] {
        self.rows.row(i)
    }

    /// Position offset between the two algebra planes (`stride * 64`).
    #[inline]
    pub fn plane_span(&self) -> usize {
        self.rows.plane_span()
    }

    /// Row `i`'s coefficient.
    #[inline]
    pub fn coeff(&self, i: usize) -> &C {
        &self.coeffs[i]
    }

    /// Decodes row `i` into `buf` (which must be `2 * stride` words long) and
    /// returns its two word planes.
    ///
    /// For reconstructing an owned `Self::Term` at an export boundary; the hot
    /// kernels work on `row_positions` instead.
    pub fn decode_row<'a>(&self, i: usize, buf: &'a mut [u64]) -> [&'a [u64]; 2] {
        let stride = self.stride;
        assert!(buf.len() >= 2 * stride, "decode buffer must hold both planes");
        let (a, rest) = buf.split_at_mut(stride);
        let (b, _) = rest.split_at_mut(stride);
        self.rows.decode_into(i, [&mut *a, &mut *b]);
        [&*a, &*b]
    }

    /// Grows `coeffs` so at least `needed_len` rows fit
    pub fn ensure_capacity(&mut self, needed_len: usize) {
        if needed_len <= self.cap() {
            return;
        }
        let new_cap = (2 * needed_len).max(16);
        self.coeffs.resize(new_cap, C::default());
    }

    /// Appends one new row with the given word planes and coefficient, growing capacity if
    /// needed.
    pub fn push(&mut self, term_planes: [&[u64]; 2], coeff: C) {
        let row = self.rows.len();
        self.ensure_capacity(row + 1);
        self.coeffs[row] = coeff;
        self.rows.push_planes(term_planes);
    }

    /// Appends one new row from an ascending position list.
    pub fn push_positions(&mut self, positions: &[Position], coeff: C) {
        let row = self.rows.len();
        self.ensure_capacity(row + 1);
        self.coeffs[row] = coeff;
        self.rows.push_row(positions);
    }

    /// Truncates to zero live rows. Does not shrink or clear coefficient capacity.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.invalidate_merge_index();
    }

    pub(crate) fn ensure_aux_capacity(&mut self, needed_len: usize) {
        if self.aux_coeffs.len() < needed_len {
            self.aux_coeffs.resize(needed_len, C::default());
        }
        if self.aux_hashes.len() < needed_len {
            self.aux_hashes.resize(needed_len, 0);
        }
    }

    /// Swaps the scattered `aux_*` coefficient/hash buffers in as the live ones.
    pub(crate) fn swap_in_aux(&mut self) {
        std::mem::swap(&mut self.coeffs, &mut self.aux_coeffs);
        std::mem::swap(&mut self.hashes, &mut self.aux_hashes);
    }

    /// Marks the persisted merge index as untrustworthy, forcing the next `merge()` call to do
    /// a full rebuild instead of an incremental one.
    pub(crate) fn invalidate_merge_index(&mut self) {
        self.merge_synced_len = 0;
    }

    /// Bytes occupied by the resident sparse term keys.
    ///
    /// Excludes coefficients, merge metadata, and every temporary workspace;
    /// see [`workspace_peak_bytes`] for the latter.
    pub fn sparse_key_bytes(&self) -> usize {
        self.rows.memory_bytes()
    }

    /// Bytes occupied by the resident coefficient column.
    pub fn coeff_bytes(&self) -> usize {
        self.coeffs.capacity() * std::mem::size_of::<C>()
    }

    /// Bytes held by key-rebuild scratch, which is not resident key storage.
    pub fn key_scratch_bytes(&self) -> usize {
        self.rows.scratch_bytes()
    }

    pub(crate) fn ensure_scratch_capacity(&mut self, needed_len: usize) {
        if self.flags.len() < needed_len {
            self.flags.resize(needed_len, 0);
        }
        if self.index.len() < needed_len {
            self.index.resize(needed_len, 0);
        }
    }

    pub(crate) fn ensure_hashes_capacity(&mut self, needed_len: usize) {
        if self.hashes.len() < needed_len {
            self.hashes.resize(needed_len, 0);
        }
    }

    /// Grow the reusable merge-table pool to at least `n_batches` tables.
    pub(crate) fn ensure_merge_tables_capacity(&mut self, n_batches: usize) {
        if self.merge_tables.len() < n_batches {
            self.merge_tables.resize_with(n_batches, hashbrown::HashTable::new);
        }
    }

    /// Clears the merge-table pool
    pub(crate) fn clear_merge_tables(&mut self) {
        self.merge_tables.iter_mut().for_each(|t| t.clear());
    }

    /// Deep-copies the live rows into a fresh term sum
    pub fn copy(&self) -> Self
    where
        C: Clone,
    {
        SoaTermSum {
            rows: self.rows.clone(),
            coeffs: self.coeffs[..self.len()].to_vec(),
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            stride: self.stride,
            n_units: self.n_units,
        }
    }

    /// Maps every live row's coefficient through `f`
    pub fn map_coeffs<C2: CoeffRepr>(&self, f: impl Fn(&C) -> C2) -> SoaTermSum<C2> {
        SoaTermSum {
            rows: self.rows.clone(),
            coeffs: self.coeffs[..self.len()].iter().map(f).collect(),
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            stride: self.stride,
            n_units: self.n_units,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the storage objective directly: two persistent dense planes would
    /// cost `2 * stride * rows` words no matter how few bits each key sets, so a
    /// wide, low-weight term sum whose key bytes stay near the set-bit count can
    /// have no persistent dense plane fields.
    #[test]
    fn production_term_sum_has_no_persistent_dense_planes() {
        let stride = 64; // 4096 qubits per plane
        let n_rows = 1000;
        let mut terms = SoaTermSum::<f64>::new(4096, stride);
        let mut x = vec![0u64; stride];
        let z = vec![0u64; stride];
        for i in 0..n_rows {
            x[i % stride] = 1;
            terms.push([&x, &z], 1.0);
            x[i % stride] = 0;
        }
        assert_eq!(terms.len(), n_rows);

        let dense_bytes = 2 * stride * n_rows * std::mem::size_of::<u64>();
        assert!(
            terms.sparse_key_bytes() * 8 < dense_bytes,
            "key storage ({} bytes) is not sparse against the dense equivalent ({dense_bytes} bytes)",
            terms.sparse_key_bytes()
        );
    }

    #[test]
    fn decode_row_reproduces_the_pushed_planes() {
        let stride = 3;
        let mut terms = SoaTermSum::<f64>::new(160, stride);
        let x = [0b101u64, 0, 1 << 40];
        let z = [0u64, 1 << 7, 0];
        terms.push([&x, &z], 2.0);
        let mut buf = vec![0u64; 2 * stride];
        let planes = terms.decode_row(0, &mut buf);
        assert_eq!(planes[0], &x[..]);
        assert_eq!(planes[1], &z[..]);
    }

    #[test]
    fn clear_drops_rows_and_invalidates_the_merge_index() {
        let mut terms = SoaTermSum::<f64>::new(8, 1);
        terms.push([&[1], &[0]], 1.0);
        terms.merge_synced_len = 1;
        terms.clear();
        assert!(terms.is_empty());
        assert_eq!(terms.merge_synced_len, 0);
    }

    #[test]
    fn kernel_layout_can_be_overridden_for_a_b_runs() {
        let original = kernel_layout();
        set_kernel_layout(KernelLayout::Dense);
        assert_eq!(kernel_layout(), KernelLayout::Dense);
        set_kernel_layout(KernelLayout::Sparse);
        assert_eq!(kernel_layout(), KernelLayout::Sparse);
        set_kernel_layout(original);
    }
}
