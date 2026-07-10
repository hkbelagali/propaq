///
/// Structure-of-Arrays term storage shared by the numerical and surrogate
/// propagators.
///
/// Instead of one term per heap-allocated struct (`PauliString { x, z, ... }`,
/// `MajoranaMonomial { modes, ... }`) keyed into a hashmap, every term's bit
/// planes live in a handful of contiguous `Vec<u64>` columns, and every
/// coefficient in one `Vec<C>` column. All operations (truncate, merge,
/// rotation-gate application, noise, expectation value) are then flag ->
/// prefix-sum -> scatter kernels over these columns (see `soa::kernels`),
/// which are trivially thread-safe: each kernel only ever writes to indices
/// derived from a bijective prefix sum, so parallel workers never alias.
///
/// This replaces the previous hash-partition/outbox design (see
/// `propagator::AbstractPropagator`), which paid a rayon fork/join and a
/// hashmap insert on every gate application.
///
pub mod kernels;
pub mod propagator;

use num_complex::Complex64;

use crate::coeff::CoeffRepr;

/// Per-basis seam for the SoA engine: the vectorized counterparts of
/// `AbstractTerm`'s per-term methods, operating directly on word slices so
/// the container's contiguous planes never need to materialize a per-term
/// struct on the hot path.
///
/// Every basis currently needs exactly two `u64` bit-planes per term: Pauli
/// uses both for identity (`x`, `z`); Majorana uses the first (`modes`) for
/// identity and the second (`p`, the cached Jordan-Wigner prefix-XOR-scan) as
/// a derived value that must travel with a term through sort/compaction but
/// plays no part in equality.
pub trait SoaBasis: Send + Sync + 'static {
    /// Per-basis Python term type (`PauliString` / `MajoranaMonomial`) used
    /// only at construction/reconstruction boundaries, not on the hot path.
    type Term: Clone + Send + Sync;

    /// Words per plane needed to store `n_units` qubits/modes.
    fn stride_words(n_units: usize) -> usize {
        let width = n_units.next_power_of_two().max(1);
        width.div_ceil(64)
    }

    /// Whether `term` commutes with `gen`.
    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool;

    /// `gen @ term -> (phase, product)`, written word-for-word into `out`.
    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64;

    /// Operator weight of `term` (used by truncation and noise damping).
    /// Takes `n_units` because Majorana's weight reads the cached
    /// Jordan-Wigner prefix-scan plane (`p`) rather than rescanning `modes`
    /// from scratch, and that reconstruction needs the qubit count; Pauli's
    /// weight (`popcount(x|z)`) ignores it.
    fn weight(term: [&[u64]; 2], n_units: usize) -> u32;

    /// $\langle \psi | \text{term} | \psi \rangle$ for a computational basis
    /// state. Takes `n_units` for the same reason as `weight` (Majorana
    /// iterates per-qubit pairs; Pauli ignores it).
    fn trace(term: [&[u64]; 2], n_units: usize, fock: u64) -> f64;

    /// Hash of the key-relevant plane words only (ignoring derived caches
    /// like Majorana's `p`), for `soa::kernels::merge`'s hash-based
    /// duplicate detection. **Must agree with `key_eq`: if `key_eq(a, b)` is
    /// `true`, `key_hash(a) == key_hash(b)` must also hold.** `merge`'s
    /// parallel-batch correctness depends on this — it assigns rows to
    /// worker batches by (bits of) this hash specifically so that every
    /// instance of a duplicate group is guaranteed to land in the same
    /// batch, needing no cross-batch synchronization to accumulate them.
    fn key_hash(term: [&[u64]; 2]) -> u64;

    /// Equality over the key-relevant planes only. Cheaper than checking
    /// `key_cmp(a, b) == Ordering::Equal` when only equality (not a full
    /// order) is needed, and what `merge` uses to resolve hash collisions
    /// within a batch.
    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool;

    /// Build the per-basis Python term object from its plane words.
    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> Self::Term;

    /// Decompose a Python term object into its plane words.
    fn term_into_planes(term: &Self::Term, n_units: usize, out: [&mut [u64]; 2]);
}

/// Columnar term storage: identity (+ derived) contiguous bit-planes,
/// one coefficient column, and reusable auxiliary buffers for the
/// flag/prefix-sum/scatter kernels (the `P2`/`C2`/`F`/`I` of the design).
pub struct SoaTermSum<C: CoeffRepr> {
    /// Primary bit-planes, each of length `cap * stride`; live data occupies
    /// `[0, len * stride)`.
    pub planes: [Vec<u64>; 2],
    /// Primary coefficient column, length `cap`; live data occupies `[0, len)`.
    pub coeffs: Vec<C>,
    /// Scratch destination planes/coeffs reused across truncate/merge/gate
    /// passes to avoid a fresh allocation on every call.
    aux_planes: [Vec<u64>; 2],
    aux_coeffs: Vec<C>,
    /// Flag array `F` and index array `I`, reused across passes.
    flags: Vec<u32>,
    index: Vec<usize>,
    /// Per-row `SoaBasis::key_hash` scratch for `merge`'s hash-based
    /// duplicate detection, reused across calls like `flags`/`index`.
    hashes: Vec<u64>,
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
            len: 0,
            stride,
            n_units,
        }
    }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// Physical term capacity of the primary buffers, derived from their
    /// actual allocated length (never tracked separately, since `planes`
    /// and `coeffs` get swapped with the auxiliary buffers by `swap_in_aux`
    /// and a separately-tracked counter would go stale across that swap).
    #[inline]
    fn cap(&self) -> usize { self.coeffs.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Live-region view of plane `p`.
    #[inline]
    pub fn plane(&self, p: usize) -> &[u64] {
        &self.planes[p][..self.len * self.stride]
    }

    /// Words for term `i` in plane `p`.
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

    /// Grow primary storage so it can hold at least `needed_len` terms,
    /// doubling (rather than growing to exactly `needed_len`) so repeated
    /// small appends across many gate applications amortize to O(1).
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

    /// Set the logical length after a pass has written `[0, new_len)`.
    /// Does not shrink underlying allocations.
    pub fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
    }

    /// Append one term (used by `add`/`__setitem__`/observable loading).
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

    /// Resize (grow-only) the auxiliary scratch buffers to hold at least
    /// `needed_len` terms. Callers in `soa::kernels` then destructure `self`
    /// into disjoint field borrows (`planes`, `coeffs`, `aux_planes`, ...)
    /// to write the compacted result and call `swap_in_aux` — this method
    /// deliberately doesn't return those borrows itself, since a method
    /// returning a borrow of one field ties up `&mut self` for that borrow's
    /// whole lifetime and blocks disjoint access to sibling fields.
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

    /// Swap the auxiliary buffers into the primary position and set the new
    /// logical length. The (now-stale) primary buffers become the new
    /// auxiliary scratch for next time.
    pub(crate) fn swap_in_aux(&mut self, new_len: usize) {
        std::mem::swap(&mut self.planes, &mut self.aux_planes);
        std::mem::swap(&mut self.coeffs, &mut self.aux_coeffs);
        debug_assert!(new_len <= self.cap());
        self.len = new_len;
    }

    /// Resize (grow-only) the flag/index scratch arrays to hold at least
    /// `needed_len`. See `ensure_aux_capacity` for why this doesn't return
    /// the borrows directly.
    pub(crate) fn ensure_scratch_capacity(&mut self, needed_len: usize) {
        if self.flags.len() < needed_len {
            self.flags.resize(needed_len, 0);
        }
        if self.index.len() < needed_len {
            self.index.resize(needed_len, 0);
        }
    }

    /// Resize (grow-only) the hash scratch array to hold at least
    /// `needed_len`. See `ensure_aux_capacity` for why this doesn't return
    /// the borrow directly.
    pub(crate) fn ensure_hashes_capacity(&mut self, needed_len: usize) {
        if self.hashes.len() < needed_len {
            self.hashes.resize(needed_len, 0);
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
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }

    /// Build a new `SoaTermSum<C2>` with identical term-identity planes and
    /// each coefficient mapped through `f` — a bulk columnar copy (both live
    /// plane regions via `copy_from_slice`, no per-term struct or Python
    /// object materialized). This is the seam for switching coefficient
    /// representations on the same term set, e.g. seeding the surrogate's
    /// `SoaTermSum<SymbolicCoeff>` from a numerical `SoaTermSum<f64>`
    /// observable via `SymbolicCoeff::from_real`.
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
            len: self.len,
            stride: self.stride,
            n_units: self.n_units,
        }
    }
}
