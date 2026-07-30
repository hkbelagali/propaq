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

    /// If `gen`'s nonzero bits all live in a single stride-word, returns that word's index --
    /// letting callers use `commutes_at_word`/`product_at_word` (O(1) in the number of words,
    /// touching only that one word of any term) instead of the fully generic `commutes`/
    /// `product` (O(stride), scanning every word of every term). This is exactly the common
    /// case for single-qubit gates (Rz/Rx/Ry) in circuits with more than 64 qubits.
    ///
    /// Default: always `None`, so bases with a different product/phase algebra (e.g. Majorana,
    /// whose product isn't the same X/Z symplectic formula) safely fall back to the generic
    /// path without needing to implement anything.
    fn local_word(_gen: [&[u64]; 2]) -> Option<usize> {
        None
    }

    /// Commutation check restricted to the single word `local_word` identified. `term_word`/
    /// `gen_word` are that one word's (x, z) bits; the rest of `term`/`gen` outside that word
    /// never affects the result, since `gen` is confined to it. Must agree exactly with
    /// `Self::commutes` for that `gen`, for any term. Only ever called when `local_word`
    /// returned `Some` for the same `gen`; the default is unreachable in that case.
    fn commutes_at_word(_term_word: [u64; 2], _gen_word: [u64; 2]) -> bool {
        unimplemented!("commutes_at_word must be implemented whenever local_word can return Some")
    }

    /// Product restricted to the single word `local_word` identified: writes the new word's
    /// (x, z) bits and returns the phase. Must agree exactly with `Self::product` for that
    /// `gen`/`term` pair (only word `local_word` of `term` actually changes; the phase's
    /// dependence on the rest of the term telescopes away identically between `term` and the
    /// full product, since they're equal everywhere outside that one word). Only ever called
    /// when `local_word` returned `Some` for the same `gen`.
    fn product_at_word(_term_word: [u64; 2], _gen_word: [u64; 2]) -> ([u64; 2], Complex64) {
        unimplemented!("product_at_word must be implemented whenever local_word can return Some")
    }
}

pub struct SoaTermSum<C: CoeffRepr> {
    pub planes: [Vec<u64>; 2],

    pub coeffs: Vec<C>,

    aux_planes: [Vec<u64>; 2],
    aux_coeffs: Vec<C>,

    flags: Vec<u32>,
    index: Vec<usize>,

    hashes: Vec<u64>,
    // Double-buffer for `hashes`, mirroring `aux_planes`/`aux_coeffs`. Needed because
    // `hashbrown::HashTable::entry`'s hasher closure (`|&cand| hashes[cand]`) gets called for
    // *existing* entries whenever a table grows -- if `hashes[]` for old rows weren't kept
    // correct across compactions, a grow event would redistribute a stale-hashed entry into the
    // wrong bucket, permanently orphaning it (a real duplicate that silently never merges again
    // for the rest of the run). So `hashes` must be relocated by `compact()` in lockstep with
    // `planes`/`coeffs`, not just recomputed wholesale every `merge()` call.
    aux_hashes: Vec<u64>,

    // One reusable hash table per merge batch -- cleared (not reallocated) and reused across
    // every `merge()` call instead of being built from scratch each time. Merge runs on nearly
    // every gate, so allocating fresh multi-million-entry tables every call was a large,
    // avoidable cost (confirmed by profiling: this was the single largest hotspot in the whole
    // propagator, and pyrauli's equivalent `DirtySet` already reuses its backing storage the
    // same way via `clear()` + conditional `reserve()`).
    merge_tables: Vec<hashbrown::HashTable<usize>>,
    // Rows `[0, merge_synced_len)` are exactly the rows already inserted into `merge_tables`,
    // keyed by their *current* physical index (kept valid across compaction by
    // `compact()`/`remap_merge_index`). Rows `[merge_synced_len, len())` are new since the last
    // `merge()` call and not yet tracked. `0` means "not trustworthy, do a full rebuild" --
    // covers first-ever merge, post-`copy()`/`map_coeffs()`, and post-invalidation (see
    // `invalidate_merge_index`) as one code path.
    merge_synced_len: usize,
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
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
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
    /// a full rebuild instead of an incremental one. Required whenever a row's *key* (its
    /// Pauli/Majorana content) changes without going through `compact()` -- e.g. `apply_rotation`'s
    /// Clifford in-place rewrite, which overwrites an existing row at a fixed physical index.
    /// Without this, a persisted table entry for that row would silently point at stale content:
    /// not a wrong coefficient (lookups always dereference live content), but a ghost duplicate
    /// entry that accumulates over the run, since the row gets re-inserted under its new key the
    /// next time the table is rebuilt while the stale entry is still sitting there.
    ///
    /// Deliberately does not clear `merge_tables` here -- that happens lazily, once, inside
    /// `merge()`'s own `merge_synced_len == 0` gate, so several Clifford gates between two merges
    /// don't pay for clearing more than once.
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

    /// Grow the reusable merge-table pool to at least `n_batches` tables. Tables already
    /// present are left as-is (still holding their allocated capacity from prior merges);
    /// callers must `.clear()` each table before reuse, same as `flags`/`index`/`hashes`
    /// are logically "reset" (via overwrite) rather than reallocated between calls.
    pub(crate) fn ensure_merge_tables_capacity(&mut self, n_batches: usize) {
        if self.merge_tables.len() < n_batches {
            self.merge_tables.resize_with(n_batches, hashbrown::HashTable::new);
        }
    }

    /// Clears the merge-table pool, reusing each table's allocated capacity rather than
    /// reallocating -- same rationale as `ensure_merge_tables_capacity`. Replaces the
    /// `merge_tables.iter_mut().for_each(|t| t.clear())` block that used to be duplicated
    /// inline in `merge()`/`merge_and_truncate()`.
    pub(crate) fn clear_merge_tables(&mut self) {
        self.merge_tables.iter_mut().for_each(|t| t.clear());
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
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
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
            aux_hashes: Vec::new(),
            merge_tables: Vec::new(),
            merge_synced_len: 0,
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }
}
