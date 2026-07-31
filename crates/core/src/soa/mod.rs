///
/// Structure-of-Arrays term storage shared by the numerical and surrogate
/// propagators.
///
pub mod kernels;
pub mod propagator;

use num_complex::Complex64;

use crate::coeff::CoeffRepr;

/// The algebra a `SoaTermSum` needs from its term representation (Pauli, Majorana, etc.) to
/// run the shared kernels in `soa::kernels`.
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
}

/// Structure-of-Arrays storage for a sum of terms
pub struct SoaTermSum<C: CoeffRepr> {
    /// The two word planes (`[x_words, z_words]` for Pauli), each `stride * cap()` words long.
    pub planes: [Vec<u64>; 2],

    /// Per-row coefficients, parallel to `planes`.
    pub coeffs: Vec<C>,

    aux_planes: [Vec<u64>; 2],
    aux_coeffs: Vec<C>,

    flags: Vec<u32>,
    index: Vec<usize>,

    hashes: Vec<u64>,
    // Double-buffer for `hashes`
    aux_hashes: Vec<u64>,

    // One reusable hash table per merge batch
    merge_tables: Vec<hashbrown::HashTable<usize>>,

    merge_synced_len: usize,
    len: usize,
    /// Number of `u64` words per row in each plane.
    pub stride: usize,
    /// Number of qubits (Pauli) or modes (Majorana) this term sum's rows are sized for.
    pub n_units: usize,
}

impl<C: CoeffRepr> SoaTermSum<C> {
    /// Creates an empty term sum sized for `n_units` qubits/modes at the given `stride`
    pub fn new(n_units: usize, stride: usize) -> Self {
        SoaTermSum {
            planes: [Vec::new(), Vec::new()],
            coeffs: Vec::new(),
            aux_planes: [Vec::new(), Vec::new()],
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            len: 0,
            stride,
            n_units,
        }
    }

    /// Number of live rows.
    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    fn cap(&self) -> usize { self.coeffs.len() }

    /// True if there are no live rows.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// The live portion (`[0, len * stride)`) of word plane `p` (0 or 1), across every row.
    #[inline]
    pub fn plane(&self, p: usize) -> &[u64] {
        &self.planes[p][..self.len * self.stride]
    }

    /// Row `i`'s slice of word plane `p`.
    #[inline]
    pub fn term_plane(&self, i: usize, p: usize) -> &[u64] {
        let s = self.stride;
        &self.planes[p][i * s..(i + 1) * s]
    }

    /// Both of row `i`'s word plane slices, as `SoaBasis` methods expect them.
    #[inline]
    pub fn term_planes(&self, i: usize) -> [&[u64]; 2] {
        [self.term_plane(i, 0), self.term_plane(i, 1)]
    }

    /// Row `i`'s coefficient.
    #[inline]
    pub fn coeff(&self, i: usize) -> &C { &self.coeffs[i] }

    /// Grows `planes`/`coeffs` so at least `needed_len` rows fit
    pub fn ensure_capacity(&mut self, needed_len: usize) {
        if needed_len <= self.cap() {
            return;
        }
        let new_cap = (2 * needed_len).max(16);
        for plane in &mut self.planes {
            plane.resize(new_cap * self.stride, 0);
        }
        self.coeffs.resize(new_cap, C::default());
    }

    /// Sets the live row count directly, without touching capacity. Callers must have already
    /// written valid data into `[0, new_len)`.
    pub fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
    }

    /// Appends one new row with the given word planes and coefficient, growing capacity if
    /// needed.
    pub fn push(&mut self, term_planes: [&[u64]; 2], coeff: C) {
        self.ensure_capacity(self.len + 1);
        let s = self.stride;
        for p in 0..2 {
            self.planes[p][self.len * s..(self.len + 1) * s].copy_from_slice(term_planes[p]);
        }
        self.coeffs[self.len] = coeff;
        self.len += 1;
    }

    /// Truncates to zero live rows. Does not shrink or clear any underlying capacity.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn ensure_aux_capacity(&mut self, needed_len: usize) {
        let stride = self.stride;
        for plane in &mut self.aux_planes {
            if plane.len() < needed_len * stride {
                plane.resize(needed_len * stride, 0);
            }
        }
        if self.aux_coeffs.len() < needed_len {
            self.aux_coeffs.resize(needed_len, C::default());
        }
        if self.aux_hashes.len() < needed_len {
            self.aux_hashes.resize(needed_len, 0);
        }
    }

    pub(crate) fn swap_in_aux(&mut self, new_len: usize) {
        std::mem::swap(&mut self.planes, &mut self.aux_planes);
        std::mem::swap(&mut self.coeffs, &mut self.aux_coeffs);
        std::mem::swap(&mut self.hashes, &mut self.aux_hashes);
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
    }

    /// Marks the persisted merge index as untrustworthy, forcing the next `merge()` call to do
    /// a full rebuild instead of an incremental one.
    pub(crate) fn invalidate_merge_index(&mut self) {
        self.merge_synced_len = 0;
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
    pub fn copy(&self) -> Self where C: Clone {
        let s = self.stride;
        SoaTermSum {
            planes: [self.planes[0][..self.len * s].to_vec(), self.planes[1][..self.len * s].to_vec()],
            coeffs: self.coeffs[..self.len].to_vec(),
            aux_planes: [Vec::new(), Vec::new()],
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }

    /// Maps every live row's coefficient through `f`
    pub fn map_coeffs<C2: CoeffRepr>(&self, f: impl Fn(&C) -> C2) -> SoaTermSum<C2> {
        let s = self.stride;
        let live = self.len * s;
        SoaTermSum {
            planes: [self.planes[0][..live].to_vec(), self.planes[1][..live].to_vec()],
            coeffs: self.coeffs[..self.len].iter().map(f).collect(),
            aux_planes: [Vec::new(), Vec::new()],
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }
}
