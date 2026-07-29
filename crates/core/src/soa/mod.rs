///
/// Structure-of-Arrays term storage shared by the numerical and surrogate
/// propagators.
///
pub mod kernels;
pub mod propagator;

use num_complex::Complex64;

use crate::coeff::CoeffRepr;

pub trait SoaBasis: Send + Sync + 'static {
    type Term: Clone + Send + Sync;

    fn stride_words(n_units: usize) -> usize {
        let width = n_units.next_power_of_two().max(1);
        width.div_ceil(64)
    }

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool;

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64;

    fn weight(term: [&[u64]; 2], n_units: usize) -> u32;

    fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64;

    fn key_hash(term: [&[u64]; 2]) -> u64;

    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool;

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> Self::Term;

    fn term_into_planes(term: &Self::Term, n_units: usize, out: [&mut [u64]; 2]);
}

pub struct SoaTermSum<C: CoeffRepr> {
    pub planes: [Vec<u64>; 2],

    pub coeffs: Vec<C>,

    aux_planes: [Vec<u64>; 2],
    aux_coeffs: Vec<C>,

    flags: Vec<u32>,
    index: Vec<usize>,

    hashes: Vec<u64>,

    // One reusable hash table per merge batch -- cleared (not reallocated) and reused across
    // every `merge()` call instead of being built from scratch each time. Merge runs on nearly
    // every gate, so allocating fresh multi-million-entry tables every call was a large,
    // avoidable cost (confirmed by profiling: this was the single largest hotspot in the whole
    // propagator, and pyrauli's equivalent `DirtySet` already reuses its backing storage the
    // same way via `clear()` + conditional `reserve()`).
    merge_tables: Vec<hashbrown::HashTable<usize>>,
    len: usize,
    pub stride: usize,
    pub n_units: usize,
}

impl<C: CoeffRepr> SoaTermSum<C> {
    pub fn new(n_units: usize, stride: usize) -> Self {
        SoaTermSum {
            planes: [Vec::new(), Vec::new()],
            coeffs: Vec::new(),
            aux_planes: [Vec::new(), Vec::new()],
            aux_coeffs: Vec::new(),
            flags: Vec::new(),
            index: Vec::new(),
            hashes: Vec::new(),
            merge_tables: Vec::new(),
            len: 0,
            stride,
            n_units,
        }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    fn cap(&self) -> usize { self.coeffs.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    #[inline]
    pub fn plane(&self, p: usize) -> &[u64] {
        &self.planes[p][..self.len * self.stride]
    }

    #[inline]
    pub fn term_plane(&self, i: usize, p: usize) -> &[u64] {
        let s = self.stride;
        &self.planes[p][i * s..(i + 1) * s]
    }

    #[inline]
    pub fn term_planes(&self, i: usize) -> [&[u64]; 2] {
        [self.term_plane(i, 0), self.term_plane(i, 1)]
    }

    #[inline]
    pub fn coeff(&self, i: usize) -> &C { &self.coeffs[i] }

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

    pub fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
    }

    pub fn push(&mut self, term_planes: [&[u64]; 2], coeff: C) {
        self.ensure_capacity(self.len + 1);
        let s = self.stride;
        for p in 0..2 {
            self.planes[p][self.len * s..(self.len + 1) * s].copy_from_slice(term_planes[p]);
        }
        self.coeffs[self.len] = coeff;
        self.len += 1;
    }

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
    }

    pub(crate) fn swap_in_aux(&mut self, new_len: usize) {
        std::mem::swap(&mut self.planes, &mut self.aux_planes);
        std::mem::swap(&mut self.coeffs, &mut self.aux_coeffs);
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
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

    /// Grow the reusable merge-table pool to at least `n_batches` tables. Tables already
    /// present are left as-is (still holding their allocated capacity from prior merges);
    /// callers must `.clear()` each table before reuse, same as `flags`/`index`/`hashes`
    /// are logically "reset" (via overwrite) rather than reallocated between calls.
    pub(crate) fn ensure_merge_tables_capacity(&mut self, n_batches: usize) {
        if self.merge_tables.len() < n_batches {
            self.merge_tables.resize_with(n_batches, hashbrown::HashTable::new);
        }
    }

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
            merge_tables: Vec::new(),
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }

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
            merge_tables: Vec::new(),
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }
}
