///
/// Term storage: sparse keys in one arena, coefficients in a parallel column.
///
/// This is the interchange representation the Python datatypes are built on
/// (`PauliTermSum`, `MajoranaTermSum`) and what the hybrid contraction reads.
/// It is deliberately *only* storage: propagation lives in
/// `crate::partitioned`, which keeps its own store, and a term sum is what a run
/// is seeded from and what it hands back.
///
/// It was the working state of an earlier structure-of-arrays engine, which is
/// why the layout is columnar. That engine is gone; the scratch columns it
/// needed (merge flags, hashes, per-batch tables) went with it, leaving the four
/// fields that actually hold a term sum.
///

use num_complex::Complex64;
use smallvec::{smallvec, SmallVec};

use crate::coeff::CoeffRepr;

/// Below this row count a pass runs serially: the rayon fork costs more than
/// the work saves.
pub const PAR_MIN_LEN: usize = 512;

pub mod sparse;

pub use sparse::{
    reset_workspace_peak, workspace_peak_bytes, DenseWorkspace, Position, SparseRows,
};

/// Chunk size floor below which a pass runs serially.

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






/// Splits a row into its plane-0 and plane-1 position slices.
#[inline]
pub fn split_planes(row: &[Position], plane_span: usize) -> (&[Position], &[Position]) {
    row.split_at(row.partition_point(|&p| (p as usize) < plane_span))
}

/// Decodes one row into a stack-resident buffer and hands its word planes to `f`.
///
/// Only ever one row wide, so it is bounded regardless of term count; the
/// chunked [`DenseWorkspace`] is for callers that need a run of rows at once.
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

/// The algebra a `TermSum` needs from its term representation (Pauli, Majorana, etc.) to
/// run the shared kernels in `store::kernels`.
///
/// The word-plane methods define the algebra. The `*_sparse` methods are what
/// the kernels actually call on the hot path; their defaults decode a single
/// row and delegate, so a basis only overrides the ones worth doing directly on
/// positions.
pub trait TermBasis: Send + Sync + 'static {
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
pub struct TermSum<C: CoeffRepr> {
    /// Sparse, row-major term keys. Authoritative: nothing else stores a key.
    rows: SparseRows,

    /// Per-row coefficients, parallel to `rows`.
    pub coeffs: Vec<C>,





    /// Number of `u64` words one decoded row occupies per plane.
    pub stride: usize,
    /// Number of qubits (Pauli) or modes (Majorana) this term sum's rows are sized for.
    pub n_units: usize,
}

impl<C: CoeffRepr> TermSum<C> {
    /// Creates an empty term sum sized for `n_units` qubits/modes at the given `stride`
    pub fn new(n_units: usize, stride: usize) -> Self {
        TermSum {
            rows: SparseRows::new(stride),
            coeffs: Vec::new(),
            stride,
            n_units,
        }
    }

    /// Number of live rows.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
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


    /// Appends one new row with the given word planes and coefficient, growing capacity if
    /// needed.
    pub fn push(&mut self, term_planes: [&[u64]; 2], coeff: C) {
        self.coeffs.push(coeff);
        self.rows.push_planes(term_planes);
    }

    /// Appends one new row from an ascending position list.
    pub fn push_positions(&mut self, positions: &[Position], coeff: C) {
        self.coeffs.push(coeff);
        self.rows.push_row(positions);
    }

    /// Truncates to zero live rows. Does not shrink or clear coefficient capacity.
    pub fn clear(&mut self) {
        self.rows.clear();
        self.coeffs.clear();
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


    /// Deep-copies the live rows into a fresh term sum
    pub fn copy(&self) -> Self
    where
        C: Clone,
    {
        TermSum {
            rows: self.rows.clone(),
            coeffs: self.coeffs[..self.len()].to_vec(),
            stride: self.stride,
            n_units: self.n_units,
        }
    }

    /// Maps every live row's coefficient through `f`
    pub fn map_coeffs<C2: CoeffRepr>(&self, f: impl Fn(&C) -> C2) -> TermSum<C2> {
        TermSum {
            rows: self.rows.clone(),
            coeffs: self.coeffs[..self.len()].iter().map(f).collect(),
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
        let mut terms = TermSum::<f64>::new(4096, stride);
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
        let mut terms = TermSum::<f64>::new(160, stride);
        let x = [0b101u64, 0, 1 << 40];
        let z = [0u64, 1 << 7, 0];
        terms.push([&x, &z], 2.0);
        let mut buf = vec![0u64; 2 * stride];
        let planes = terms.decode_row(0, &mut buf);
        assert_eq!(planes[0], &x[..]);
        assert_eq!(planes[1], &z[..]);
    }

}
