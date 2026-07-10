///
/// Defines the core algebra of Majorana monomials, products of Majorana operators
///
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
use std::hash::{Hash, Hasher};
use rustc_hash::FxHasher;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::{pyint_to_bitset, bitset_to_pyint};
use propaq_core::traits::AbstractTerm;
use propaq_core::soa::SoaBasis;

/// A Majorana monomial, a product of Majorana operators encoded as a mode bitmask.
///
/// Bit 2k is set if $\gamma_{2k}$ (even mode) is active on site k.
/// Bit 2k+1 is set if $\gamma_{2k+1}$ (odd mode) is active on site k.
///
/// Arguments:
///     modes: Integer bitmask encoding occupied Majorana modes.
///     n_modes: Total number of Majorana modes (must be even, equal to 2 * n_qubits).
///     is_number_preserving: Whether the monomial preserves particle number (default True).
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct MajoranaMonomial {
    pub modes: Bitset,
    #[pyo3(get)]
    pub n_modes: usize,
    #[pyo3(get)]
    pub is_number_preserving: bool,
    pub weight: u32,
    /// Cached parallel-prefix XOR-scan value used by `compute_weight_for`.
    /// `p` is linear in `modes` under XOR (see `weight_and_p_from_product`),
    /// so it can be combined via a single XOR on every product instead of
    /// recomputing the O(log n_qubits) scan from scratch each time. Purely
    /// a derived cache of `modes`/`n_modes` — excluded from `Eq`/`Hash`.
    pub p: Bitset,
}

impl MajoranaMonomial {
    fn commutes_with_impl(&self, other: &MajoranaMonomial) -> bool {
        if self.modes == other.modes {
            return true;
        }
        let a = self.modes.as_words();
        let b = other.modes.as_words();
        let overlap: u32 = (0..a.len().min(b.len()))
            .map(|i| (a[i] & b[i]).count_ones())
            .sum();
        (self.modes.count_ones() as usize * other.modes.count_ones() as usize
            + overlap as usize)
            % 2 == 0
    }

    fn trace_fock_state_impl(&self, fock_state: u64) -> f64 {
        let n_fermionic = self.n_modes / 2;
        let mut p = 0i32;
        let mut product = 1i32;

        for k in 0..n_fermionic {
            let low  = self.modes.bit(2 * k) as i32;
            let high = self.modes.bit(2 * k + 1) as i32;

            if low != high {
                return 0.0;
            }
            if low == 1 {
                // `fock_state` is a plain `u64`, so it only ever encodes the
                // occupation of qubits `0..64`; a qubit index at or beyond
                // that is implicitly unoccupied (matching `MajoranaBasis::trace`,
                // the SoA version, which already treats every fock word past
                // the first as zero for the same reason) — without this
                // guard, systems with more than 64 fermionic modes panic here
                // (shift overflow) whenever a long run of paired qubits
                // reaches index 64.
                let n_k = if k < 64 { ((fock_state >> k) & 1) as i32 } else { 0 };
                product *= 2 * n_k - 1;
                p += 1;
            }
        }

        let phase = if (p / 2) % 2 == 0 { 1 } else { -1 };
        (phase * product) as f64
    }

    /// Per-qubit `single = x ^ y` (unpaired Majorana site, needs a Z-string)
    /// and `occupied = x | y` (site touched at all), compressed from the
    /// mode bitmask. Cheap: two `compress_to_qubits` passes, no scan.
    fn compress_single_occupied(modes: &Bitset, n_qubits: usize) -> (Bitset, Bitset) {
        let x_bits = compress_to_qubits(modes, n_qubits, 0);
        let y_bits = compress_to_qubits(modes, n_qubits, 1);
        let occupied = &x_bits | &y_bits;
        let single = &x_bits ^ &y_bits;
        (single, occupied)
    }

    /// Inclusive parallel-prefix XOR-scan of `single` over `[0, n_qubits)`:
    /// `p[k] = single[0] ^ single[1] ^ ... ^ single[k]`. This is the
    /// expensive O(log n_qubits) `shl`-heavy part — linear in `single`
    /// under XOR, so callers on the hot path should prefer combining two
    /// already-scanned `p` values (`weight_and_p_from_product`) over calling
    /// this directly.
    fn scan_p(single: &Bitset, n_qubits: usize, qubit_mask: &Bitset) -> Bitset {
        let mut p = single.clone();
        let mut shift = 1usize;
        while shift < n_qubits {
            p = &p ^ &(&p.shl(shift) & qubit_mask);
            shift <<= 1;
        }
        p
    }

    /// Final weight from the compressed parts: complements `p` into the
    /// Jordan-Wigner Z-string parity, then counts non-identity qubits.
    fn weight_from_parts(single: &Bitset, occupied: &Bitset, p: &Bitset, qubit_mask: &Bitset) -> u32 {
        let string = if single.count_ones() & 1 == 1 {
            p ^ qubit_mask
        } else {
            p.clone()
        };
        (single | &(occupied ^ &string)).count_ones()
    }

    pub fn compute_weight_for(modes: &Bitset, n_modes: usize) -> u32 {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 { return 0; }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(modes, n_qubits);
        let p = Self::scan_p(&single, n_qubits, &qubit_mask);
        Self::weight_from_parts(&single, &occupied, &p, &qubit_mask)
    }

    /// Full (weight, p) computation from scratch — used at "fresh"
    /// construction sites (not the hot multiplication path) so `p` doesn't
    /// need a second, separate scan later.
    pub fn weight_and_p_for(modes: &Bitset, n_modes: usize) -> (u32, Bitset) {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 { return (0, Bitset::zero()); }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(modes, n_qubits);
        let p = Self::scan_p(&single, n_qubits, &qubit_mask);
        let weight = Self::weight_from_parts(&single, &occupied, &p, &qubit_mask);
        (weight, p)
    }

    /// Fast path used by `matmul_internal`: `p` for the product is exactly
    /// `self_p ^ other_p` (the prefix-scan is linear in `modes` under XOR),
    /// so this needs no scan at all — only the cheap `single`/`occupied`
    /// compression on the already-XORed `result_modes`.
    pub(crate) fn weight_and_p_from_product(
        result_modes: &Bitset,
        n_modes: usize,
        self_p: &Bitset,
        other_p: &Bitset,
    ) -> (u32, Bitset) {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 { return (0, Bitset::zero()); }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(result_modes, n_qubits);
        let p = self_p ^ other_p;
        let weight = Self::weight_from_parts(&single, &occupied, &p, &qubit_mask);
        (weight, p)
    }

    pub(crate) fn matmul_internal(&self, other: &MajoranaMonomial) -> (Complex64, MajoranaMonomial) {
        let result_modes = &self.modes ^ &other.modes;
        let (weight, p) = Self::weight_and_p_from_product(&result_modes, self.n_modes, &self.p, &other.p);
        let n_fermionic = self.n_modes / 2;
        let is_np = (0..n_fermionic).all(|k| result_modes.bit(2 * k) == result_modes.bit(2 * k + 1));
        let result = MajoranaMonomial {
            modes: result_modes,
            n_modes: self.n_modes,
            is_number_preserving: is_np,
            weight,
            p,
        };

        let r_a = hermiticity_exp(self.length());
        let r_b = hermiticity_exp(other.length());
        let r_c = hermiticity_exp(result.length());
        let total_parity = resorting_parity(self.modes.as_words(), other.modes.as_words());
        let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);

        let phase = match phase_exp {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        (phase, result)
    }
}

#[pymethods]
impl MajoranaMonomial {
    /// Construct a Majorana monomial from a mode bitmask.
    ///
    /// Arguments:
    ///     modes: Integer bitmask where bit 2k (2k+1) is set if Majorana mode gamma_{2k} (gamma_{2k+1}) is active.
    ///     n_modes: Total number of Majorana modes (must be even).
    ///     is_number_preserving: Whether the monomial preserves particle number.
    #[new]
    #[pyo3(signature = (modes, n_modes, is_number_preserving = true))]
    fn new(modes: &Bound<'_, PyAny>, n_modes: usize, is_number_preserving: bool) -> PyResult<Self> {
        let bitset = pyint_to_bitset(modes, n_modes)?;
        let (weight, p) = Self::weight_and_p_for(&bitset, n_modes);
        Ok(MajoranaMonomial { modes: bitset, n_modes, is_number_preserving, weight, p })
    }

    /// The active mode indices as a Python integer bitmask.
    #[getter]
    fn modes(&self, py: Python<'_>) -> PyResult<PyObject> {
        bitset_to_pyint(py, &self.modes)
    }

    /// The number of Majorana modes in the system 
    #[getter]
    fn n_modes(&self) -> usize {
        self.n_modes
    }

    /// Whether or not the monomial preserves particle number (i.e. fully paired).
    #[getter]
    fn is_number_preserving(&self) -> bool {
        self.is_number_preserving
    }
    
    /// Number of active Majorana modes in the monomial (popcount of the mode bitmask).
    #[getter]
    fn length(&self) -> usize {
        self.modes.count_ones() as usize
    }

    /// Pauli weight of this monomial under the Jordan-Wigner mapping.
    #[getter]
    fn weight(&self) -> u32 {
        self.weight
    }

    /// Number of Majorana modes shared with *other* (popcount of modes & other.modes).
    /// Arguments:
    ///     other: Another MajoranaMonomial to compare with.
    /// Returns:
    ///     The number of Majorana modes that are active in both self and other.
    fn overlap(&self, other: &MajoranaMonomial) -> u32 {
        (&self.modes & &other.modes).count_ones()
    }

    /// Return True if this monomial commutes with *other*.
    /// Arguments:
    ///     other: Another MajoranaMonomial to check commutation with.
    /// Returns:
    ///     True if self and other commute, False otherwise.
    pub fn commutes_with(&self, other: &MajoranaMonomial) -> bool {
        self.commutes_with_impl(other)
    }

    /// Pauli weight of the product monomial self @ other, without computing the full product.
    /// Arguments:
    ///     other: Another MajoranaMonomial to multiply with.
    /// Returns:
    ///     The Pauli weight of the resulting monomial from multiplying self and other.
    fn resulting_weight(&self, other: &MajoranaMonomial) -> u32 {
        let result_modes = &self.modes ^ &other.modes;
        Self::compute_weight_for(&result_modes, self.n_modes)
    }

    /// Multiply two Majorana monomials, returning (phase, product).
    ///
    /// The phase factor accounts for the anticommutation relations of Majorana operators.
    fn __matmul__(&self, other: &MajoranaMonomial) -> PyResult<(Complex64, MajoranaMonomial)> {
        Ok(self.matmul_internal(other))
    }

    /// Compute $\langle \psi |M| \psi \rangle$ for this Majorana monomial M.
    ///
    /// Returns 0.0 if M has any unpaired modes.
    /// For paired modes, returns the product of $(2n_k - 1)$ values for each occupied pair.
    ///
    /// Arguments:
    ///     fock_state: Computational basis state as a bitstring integer.
    /// Returns:
    ///     Expectation value of the Majorana monomial in the given Fock state.
    pub fn trace_with_fock_state(&self, fock_state: u64) -> f64 {
        self.trace_fock_state_impl(fock_state)
    }

    /// Serialize the mode bitmask as a little-endian byte string.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let byte_length = (self.n_modes + 7) / 8;
        let mut bytes = self.modes.to_le_bytes();
        bytes.resize(byte_length, 0);
        PyBytes::new(py, &bytes)
    }

    fn __hash__(&self) -> u64 {
        let mut h = FxHasher::default();
        self.modes.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &MajoranaMonomial) -> bool {
        self.modes == other.modes
    }
}

impl AbstractTerm for MajoranaMonomial {
    fn weight(&self) -> u32 { self.weight }
    fn commutes_with(&self, other: &Self) -> bool { self.commutes_with_impl(other) }
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) { MajoranaMonomial::matmul_internal(self, other) }
    fn trace_with_fock_state(&self, fock_state: u64) -> f64 { self.trace_fock_state_impl(fock_state) }
    fn to_bytes_vec(&self) -> Vec<u8> {
        let byte_length = (self.n_modes + 7) / 8;
        let mut bytes = self.modes.to_le_bytes();
        bytes.resize(byte_length, 0);
        bytes
    }
    fn partition_key(&self) -> u64 {
        let mut h = FxHasher::default();
        self.modes.hash(&mut h);
        h.finish()
    }
    fn is_number_preserving(&self) -> bool { self.is_number_preserving }
    fn system_size(&self) -> u64 { self.n_modes as u64 }
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self {
        let n_modes = system_size as usize;
        let modes = Bitset::from_le_bytes(bytes);
        let (weight, p) = Self::weight_and_p_for(&modes, n_modes);
        let n_fermionic = n_modes / 2;
        let is_np = (0..n_fermionic).all(|k| modes.bit(2 * k) == modes.bit(2 * k + 1));
        MajoranaMonomial { modes, n_modes, is_number_preserving: is_np, weight, p }
    }
}

impl PartialEq for MajoranaMonomial {
    fn eq(&self, other: &Self) -> bool { self.modes == other.modes }
}

impl Eq for MajoranaMonomial {}

impl Hash for MajoranaMonomial {
    fn hash<H: Hasher>(&self, state: &mut H) { self.modes.hash(state); }
}

/// One of `gen`'s (at most two) touched-mode positions in the `modes` plane,
/// located once per `commutes`/`product` call by `classify_gen`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GenSite {
    word: usize,
    /// Exactly one bit set: the mode's position within `word`.
    mask: u64,
}

#[inline]
fn site_bit_pos(site: &GenSite) -> usize {
    site.word * 64 + site.mask.trailing_zeros() as usize
}

#[inline]
fn site_bit(modes: &[u64], site: &GenSite) -> bool {
    modes[site.word] & site.mask != 0
}

/// Structural classification of `gen.modes`'s support
/// (`popcount(gen.modes)`), computed fresh at the top of `commutes`/
/// `product`. Unlike `PauliBasis`'s analogous fast path, real Majorana
/// generators aren't always narrow (an `_xx_plus_yy_terms` gate between
/// distant qubits spans a wide Jordan-Wigner string), so `Wide` still keeps
/// an exact `gen_len` (needed by both `commutes_fast` and `product_fast`
/// regardless of width) rather than discarding it the way `PauliBasis`'s
/// `Wide` does.
#[derive(Debug, PartialEq)]
enum GenShape {
    Identity,
    Weight1(GenSite),
    Weight2(GenSite, GenSite),
    Wide,
}

/// Single fused pass: accumulates `gen`'s full weight (needed even when
/// `Wide`) while opportunistically locating up to two set bits. Never a
/// separate popcount-then-locate pass, so a narrow generator touching a
/// late word costs no more than the weight scan alone already would, and a
/// wide generator costs exactly what the original unconditional scan cost.
#[inline]
fn classify_gen(gen: [&[u64]; 2]) -> (u32, GenShape) {
    let mut gen_len: u32 = 0;
    let mut site0: Option<GenSite> = None;
    let mut site1: Option<GenSite> = None;
    let mut wide = false;
    for (word, &w) in gen[0].iter().enumerate() {
        gen_len += w.count_ones();
        if wide {
            continue;
        }
        if gen_len > 2 {
            wide = true;
            continue;
        }
        let mut remaining = w;
        while remaining != 0 {
            let mask = 1u64 << remaining.trailing_zeros();
            let site = GenSite { word, mask };
            if site0.is_none() {
                site0 = Some(site);
            } else {
                site1 = Some(site);
            }
            remaining &= remaining - 1;
        }
    }
    let shape = if wide {
        GenShape::Wide
    } else {
        match (site0, site1) {
            (None, None) => GenShape::Identity,
            (Some(s), None) => GenShape::Weight1(s),
            (Some(a), Some(b)) => GenShape::Weight2(a, b),
            (None, Some(_)) => unreachable!("a site is only ever recorded as site1 after site0"),
        }
    };
    (gen_len, shape)
}

/// The cached `p`-plane's shape for a classified `gen`, derived from the
/// *qubit* relationship of its touched site(s) — **narrow `modes` does not
/// imply narrow `p`**. `single_k = modes[2k] ^ modes[2k+1]` and `p` is
/// `single`'s inclusive prefix-XOR-scan, so: two touched bits at the *same*
/// qubit (a number/Z-type term) cancel in `single` and give an all-zero `p`;
/// two touched bits at *adjacent* qubits give a `p` with exactly one set bit
/// (the pulse `[q1, q2)` collapses to one position when `q2 == q1 + 1`); a
/// lone touched bit (e.g. `from_x` on qubit 0) gives a `p` that is a
/// **suffix** from that qubit to the end of the register — genuinely wide,
/// not narrow, even though `modes` itself is weight-1. Two touched bits at
/// non-adjacent qubits give a `p` pulse spanning the gap — also wide here
/// (a further optimization could special-case a bounded run, but isn't
/// implemented — see the module-level design note).
#[derive(Debug, PartialEq)]
enum PPlaneShape {
    Zero,
    SingleBit { word: usize, mask: u64 },
    Wide,
}

#[inline]
fn classify_p_shape(shape: &GenShape) -> PPlaneShape {
    match shape {
        GenShape::Identity => PPlaneShape::Zero,
        GenShape::Weight1(_) => PPlaneShape::Wide,
        GenShape::Weight2(a, b) => {
            let qa = site_bit_pos(a) / 2;
            let qb = site_bit_pos(b) / 2;
            if qa == qb {
                PPlaneShape::Zero
            } else {
                let lo = qa.min(qb);
                let hi = qa.max(qb);
                if hi == lo + 1 {
                    PPlaneShape::SingleBit { word: lo / 64, mask: 1u64 << (lo % 64) }
                } else {
                    PPlaneShape::Wide
                }
            }
        }
        GenShape::Wide => PPlaneShape::Wide,
    }
}

/// Fast commute check: `(term_len * gen_len + overlap) % 2 == 0` collapses
/// to `overlap % 2 == 0` whenever `gen_len` is even (`even * anything` is
/// even), meaning `term_len` never needs to be computed. Real generators
/// (`_rz_terms`/`_cp_terms`/`_xx_plus_yy_terms`/`from_swap`) are always
/// even-weight; `from_x` is the one real odd-weight generator, handled by
/// falling back to `commutes_generic` (which needs `term_len`). `overlap`
/// itself is O(1) for narrow shapes, a full scan for `Wide` (this still
/// saves `term_len`'s scan even when `gen` is wide).
#[inline]
fn commutes_fast(term: [&[u64]; 2], gen: [&[u64]; 2], gen_len: u32, shape: &GenShape) -> Option<bool> {
    if gen_len % 2 != 0 {
        return None;
    }
    let overlap: u32 = match shape {
        GenShape::Identity => 0,
        GenShape::Weight1(s) => site_bit(term[0], s) as u32,
        GenShape::Weight2(a, b) => site_bit(term[0], a) as u32 + site_bit(term[0], b) as u32,
        GenShape::Wide => term[0].iter().zip(gen[0]).map(|(a, b)| (a & b).count_ones()).sum(),
    };
    Some(overlap % 2 == 0)
}

/// Fast product. Always succeeds (no generic fallback needed) because the
/// `result_len = gen_len + term_len - 2*overlap` identity (an unconditional
/// symmetric-difference cardinality fact, no even/odd precondition) and the
/// `gen_len`/`overlap` pass-fusion apply regardless of `gen`'s width —
/// `Wide` still benefits (one fused pass over `gen[0]`/`term[0]` instead of
/// three separate reductions, plus `result_len`'s scan of `out` eliminated
/// entirely), just without the O(1) narrow-site shortcuts for `overlap`/
/// `out[0]`'s construction. `term_len` is the one quantity that can never be
/// avoided — a genuinely global property of an arbitrary term. `out[1]` (the
/// `p`-plane) is handled separately via `classify_p_shape`, since its
/// locality doesn't follow from `modes`'s narrowness (see `PPlaneShape`).
#[inline]
fn product_fast(
    term: [&[u64]; 2],
    gen: [&[u64]; 2],
    gen_len: u32,
    shape: &GenShape,
    out: [&mut [u64]; 2],
) -> Complex64 {
    let (term_len, overlap): (u32, u32) = match shape {
        GenShape::Identity => {
            out[0].copy_from_slice(term[0]);
            (term[0].iter().map(|w| w.count_ones()).sum(), 0)
        }
        GenShape::Weight1(s) => {
            out[0].copy_from_slice(term[0]);
            out[0][s.word] ^= s.mask;
            (term[0].iter().map(|w| w.count_ones()).sum(), site_bit(term[0], s) as u32)
        }
        GenShape::Weight2(a, b) => {
            out[0].copy_from_slice(term[0]);
            out[0][a.word] ^= a.mask;
            out[0][b.word] ^= b.mask;
            let overlap = site_bit(term[0], a) as u32 + site_bit(term[0], b) as u32;
            (term[0].iter().map(|w| w.count_ones()).sum(), overlap)
        }
        GenShape::Wide => {
            let mut term_len = 0u32;
            let mut overlap = 0u32;
            for i in 0..out[0].len() {
                let g = gen[0][i];
                let t = term[0][i];
                out[0][i] = g ^ t;
                term_len += t.count_ones();
                overlap += (g & t).count_ones();
            }
            (term_len, overlap)
        }
    };

    match classify_p_shape(shape) {
        PPlaneShape::Zero => out[1].copy_from_slice(term[1]),
        PPlaneShape::SingleBit { word, mask } => {
            out[1].copy_from_slice(term[1]);
            out[1][word] ^= mask;
        }
        PPlaneShape::Wide => {
            for i in 0..out[1].len() {
                out[1][i] = gen[1][i] ^ term[1][i];
            }
        }
    }

    // Exact identity, no even/odd precondition: |gen △ term| = |gen| + |term| - 2|gen ∩ term|.
    let result_len = (gen_len as i64 + term_len as i64 - 2 * overlap as i64) as usize;
    let r_a = hermiticity_exp(gen_len as usize);
    let r_b = hermiticity_exp(term_len as usize);
    let r_c = hermiticity_exp(result_len);
    let total_parity = resorting_parity(gen[0], term[0]);
    let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);
    match phase_exp {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!(),
    }
}

fn commutes_generic(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
    // Mirrors `commutes_with_impl(self=term, other=gen)`.
    if term[0] == gen[0] {
        return true;
    }
    let overlap: u32 = term[0].iter().zip(gen[0]).map(|(a, b)| (a & b).count_ones()).sum();
    let term_len: usize = term[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
    let gen_len: usize = gen[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
    (term_len * gen_len + overlap as usize) % 2 == 0
}

// Unlike `commutes_generic` (a real fallback for odd-`gen_len` generators in
// release builds too), `product_fast` always succeeds — this is only used by
// `product`'s debug-mode differential cross-check, so it's genuinely dead in
// release builds; gate it accordingly rather than warn on every build.
#[cfg(debug_assertions)]
fn product_generic(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
    // gen @ term, matching `matmul_internal(self=gen, other=term)`. `p`
    // combines by a plain XOR (linear in `modes`, see
    // `weight_and_p_from_product`) — no rescan needed, so unlike
    // `weight`/`term_from_planes` this needs no `n_units`.
    for i in 0..out[0].len() {
        out[0][i] = gen[0][i] ^ term[0][i];
        out[1][i] = gen[1][i] ^ term[1][i];
    }
    let gen_len = gen[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
    let term_len = term[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
    let result_len = out[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
    let r_a = hermiticity_exp(gen_len);
    let r_b = hermiticity_exp(term_len);
    let r_c = hermiticity_exp(result_len);
    let total_parity = resorting_parity(gen[0], term[0]);
    let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);
    match phase_exp {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!(),
    }
}

/// SoA engine seam for Majorana monomials. Plane 0 is `modes` (the term's
/// identity); plane 1 is the cached prefix-XOR-scan `p` — not part of
/// identity (`key_hash`/`key_eq` ignore it), but it must travel with a term
/// through sort/compaction/append the same way `modes` does, since it's what lets
/// `weight`/`product` avoid an O(log n_qubits) rescan on every call (see the
/// `p` field's doc comment on `MajoranaMonomial` above).
pub struct MajoranaBasis;

impl SoaBasis for MajoranaBasis {
    type Term = MajoranaMonomial;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        let (gen_len, shape) = classify_gen(gen);
        if let Some(fast) = commutes_fast(term, gen, gen_len, &shape) {
            debug_assert_eq!(
                fast, commutes_generic(term, gen),
                "MajoranaBasis::commutes fast/generic mismatch"
            );
            return fast;
        }
        commutes_generic(term, gen)
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        let [o0, o1] = out;
        let (gen_len, shape) = classify_gen(gen);
        let phase = product_fast(term, gen, gen_len, &shape, [&mut *o0, &mut *o1]);
        #[cfg(debug_assertions)]
        {
            let mut ref_0 = vec![0u64; o0.len()];
            let mut ref_1 = vec![0u64; o1.len()];
            let ref_phase = product_generic(term, gen, [&mut ref_0, &mut ref_1]);
            debug_assert!(
                (phase - ref_phase).norm() < 1e-9,
                "MajoranaBasis::product fast/generic phase mismatch"
            );
            debug_assert_eq!(*o0, ref_0[..], "MajoranaBasis::product fast/generic modes mismatch");
            debug_assert_eq!(*o1, ref_1[..], "MajoranaBasis::product fast/generic p mismatch");
        }
        phase
    }

    fn weight(term: [&[u64]; 2], n_units: usize) -> u32 {
        let n_qubits = n_units / 2;
        if n_qubits == 0 {
            return 0;
        }
        let modes = Bitset::from_words(term[0].to_vec());
        let p = Bitset::from_words(term[1].to_vec());
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = MajoranaMonomial::compress_single_occupied(&modes, n_qubits);
        MajoranaMonomial::weight_from_parts(&single, &occupied, &p, &qubit_mask)
    }

    fn trace(term: [&[u64]; 2], n_units: usize, fock: u64) -> f64 {
        // `trace_fock_state_impl` only reads `modes`/`n_modes`; the other
        // fields are irrelevant to it, so a throwaway monomial is fine here.
        let modes = Bitset::from_words(term[0].to_vec());
        let m = MajoranaMonomial {
            modes,
            n_modes: n_units,
            is_number_preserving: false,
            weight: 0,
            p: Bitset::zero(),
        };
        m.trace_fock_state_impl(fock)
    }

    fn key_hash(term: [&[u64]; 2]) -> u64 {
        // Only `modes` is identity; `p` is a derived cache.
        let mut h = FxHasher::default();
        term[0].hash(&mut h);
        h.finish()
    }

    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool {
        a[0] == b[0]
    }

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> MajoranaMonomial {
        let modes = Bitset::from_words(term[0].to_vec());
        let p = Bitset::from_words(term[1].to_vec());
        let n_qubits = n_units / 2;
        let weight = if n_qubits == 0 {
            0
        } else {
            let qubit_mask = Bitset::all_ones_upto(n_qubits);
            let (single, occupied) = MajoranaMonomial::compress_single_occupied(&modes, n_qubits);
            MajoranaMonomial::weight_from_parts(&single, &occupied, &p, &qubit_mask)
        };
        let is_np = (0..n_qubits).all(|k| modes.bit(2 * k) == modes.bit(2 * k + 1));
        MajoranaMonomial { modes, n_modes: n_units, is_number_preserving: is_np, weight, p }
    }

    fn term_into_planes(term: &MajoranaMonomial, _n_units: usize, out: [&mut [u64]; 2]) {
        let mw = term.modes.as_words();
        out[0].fill(0);
        out[0][..mw.len()].copy_from_slice(mw);
        let pw = term.p.as_words();
        out[1].fill(0);
        out[1][..pw.len()].copy_from_slice(pw);
    }
}

fn compress_to_qubits(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("bmi2") {
        return unsafe { compress_to_qubits_bmi2(modes, n_qubits, offset) };
    }
    compress_to_qubits_scalar(modes, n_qubits, offset)
}

/// We can use BMI2's PEXT instruction to compress 
/// the even and odd bits of the Majorana mode bitmask quickly.
/// Each modes word covers 64 mode bits = 32 qubits.
/// Two consecutive modes words interleave into one qubit word:
/// qubit_word[q] = pext(modes_word[2q], mask) | (pext(modes_word[2q+1], mask) << 32)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn compress_to_qubits_bmi2(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    use std::arch::x86_64::_pext_u64;
    let mask: u64 = if offset == 0 { 0x5555_5555_5555_5555 } else { 0xAAAA_AAAA_AAAA_AAAA };
    let n_qubit_words = (n_qubits + 63) / 64;
    let mut words = vec![0u64; n_qubit_words];
    let mode_words = modes.as_words();
    for qw in 0..n_qubit_words {
        let lo = mode_words.get(2 * qw).copied().unwrap_or(0);
        let hi = mode_words.get(2 * qw + 1).copied().unwrap_or(0);
        words[qw] = _pext_u64(lo, mask) | (_pext_u64(hi, mask) << 32);
    }
    Bitset::from_words(words)
}

fn compress_to_qubits_scalar(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
    let n_words = (n_qubits + 63) / 64;
    let mut words = vec![0u64; n_words];
    for k in 0..n_qubits {
        if modes.bit(2 * k + offset) != 0 {
            words[k / 64] |= 1u64 << (k % 64);
        }
    }
    Bitset::from_words(words)
}

fn hermiticity_exp(length: usize) -> i32 {
    if matches!(length % 4, 0 | 1) { 0 } else { 1 }
}

/// Operates directly on word slices (rather than `&Bitset`) so the SoA
/// `MajoranaBasis::product` kernel can call it on `SoaTermSum` plane rows
/// without allocating a temporary `Bitset` per call on the propagation hot
/// path. Correct for slices of any (possibly unequal) length: an
/// all-zero-valued `a` or `b` naturally drives every term in the sum to
/// zero, so unlike the earlier `&Bitset` version this needs no empty-input
/// short-circuit (`Bitset` could be zero-*length*; a fixed-stride slice
/// never is, just possibly all-zero-*valued*).
fn resorting_parity(a_words: &[u64], b_words: &[u64]) -> bool {
    let total: u64 = a_words.iter().map(|w| w.count_ones() as u64).sum();
    let mut running = 0u64;
    let mut count = 0u64;

    for (wi, &bw) in b_words.iter().enumerate() {
        let a_word = a_words.get(wi).copied().unwrap_or(0);
        running += a_word.count_ones() as u64;
        let above_higher = total - running;
        let mut bword = bw;
        while bword != 0 {
            let bi = bword.trailing_zeros() as usize;
            let above_same = if bi < 63 {
                (a_word >> (bi + 1)).count_ones() as u64
            } else {
                0
            };
            count += above_same + above_higher;
            bword &= bword - 1;
        }
    }
    (count & 1) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mon(bits: u64, n_modes: usize) -> MajoranaMonomial {
        let modes = Bitset::from_le_bytes(&bits.to_le_bytes());
        let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
        MajoranaMonomial { modes, n_modes, is_number_preserving: true, weight, p }
    }

    fn mon_bits(bits: Vec<u64>, n_modes: usize) -> MajoranaMonomial {
        let modes = Bitset::from_words(bits);
        let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
        MajoranaMonomial { modes, n_modes, is_number_preserving: true, weight, p }
    }

    #[test]
    fn hermiticity_exp_all_residues() {
        for (len, expected) in [(0,0),(1,0),(2,1),(3,1),(4,0),(5,0),(6,1),(7,1),(8,0)] {
            assert_eq!(hermiticity_exp(len), expected, "hermiticity_exp({len})");
        }
    }

    #[test]
    fn parity_disjoint_no_inversions() {
        let a = Bitset::from_le_bytes(&[0b0011]);
        let b = Bitset::from_le_bytes(&[0b1100]);
        assert!(!resorting_parity(a.as_words(), b.as_words()));
    }

    #[test]
    fn parity_single_inversion() {
        let a = Bitset::from_le_bytes(&[0b0010]);
        let b = Bitset::from_le_bytes(&[0b0001]);
        assert!(resorting_parity(a.as_words(), b.as_words()));
    }

    #[test]
    fn parity_two_inversions_even() {
        let a = Bitset::from_le_bytes(&[0b1100]);
        let b = Bitset::from_le_bytes(&[0b0011]);
        assert!(!resorting_parity(a.as_words(), b.as_words()));
    }

    #[test]
    fn parity_empty_b_is_false() {
        let a = Bitset::from_le_bytes(&[0xFF]);
        let b = Bitset::zero();
        assert!(!resorting_parity(a.as_words(), b.as_words()));
    }

    #[test]
    fn weight_identity() { assert_eq!(mon(0, 8).weight, 0); }

    #[test]
    fn weight_single_gamma() { assert_eq!(mon(0b01, 8).weight, 1); }

    #[test]
    fn weight_number_operator() { assert_eq!(mon(0b11, 8).weight, 1); }

    #[test]
    fn weight_four_x_modes() { assert_eq!(mon(0b0101_0101, 8).weight, 4); }

    #[test]
    fn weight_large_n_modes() { assert_eq!(mon(0b01, 128).weight, 1); }

    #[test]
    fn weight_multi_word_mode() {
        let m = mon_bits(vec![0u64, 1u64], 128);
        assert_eq!(m.weight, 33);
    }

    #[test]
    fn trace_identity_any_fock() {
        let m = mon(0, 8);
        assert_eq!(m.trace_fock_state_impl(0), 1.0);
        assert_eq!(m.trace_fock_state_impl(0b1111), 1.0);
    }

    #[test]
    fn trace_unpaired_mode_is_zero() {
        let m = mon(0b01, 8);
        assert_eq!(m.trace_fock_state_impl(0), 0.0);
        assert_eq!(m.trace_fock_state_impl(1), 0.0);
    }

    #[test]
    fn trace_site0_empty_fock() { assert_eq!(mon(0b11, 8).trace_fock_state_impl(0), -1.0); }

    #[test]
    fn trace_site0_occupied_fock() { assert_eq!(mon(0b11, 8).trace_fock_state_impl(1), 1.0); }

    #[test]
    fn trace_two_sites_all_combinations() {
        let m = mon(0b1111, 8);
        assert_eq!(m.trace_fock_state_impl(0b00), -1.0);
        assert_eq!(m.trace_fock_state_impl(0b01),  1.0);
        assert_eq!(m.trace_fock_state_impl(0b10),  1.0);
        assert_eq!(m.trace_fock_state_impl(0b11), -1.0);
    }

    /// Cross-checks a product's incrementally-computed `weight` (and cached
    /// `p`) against a from-scratch recomputation on the result's own modes
    /// — the oracle for every weight/`p`-related test in this module.
    fn assert_weight_and_p_correct(result: &MajoranaMonomial) {
        let expected_weight = MajoranaMonomial::compute_weight_for(&result.modes, result.n_modes);
        assert_eq!(result.weight, expected_weight, "weight mismatch for modes={:?}", result.modes);
        let (_, expected_p) = MajoranaMonomial::weight_and_p_for(&result.modes, result.n_modes);
        assert_eq!(result.p, expected_p, "p drifted for modes={:?}", result.modes);
    }

    #[test]
    fn matmul_identity_on_left() {
        let identity = mon(0, 8);
        let m = mon(0b0011, 8);
        let (phase, result) = identity.matmul_internal(&m);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes, m.modes);
        assert_weight_and_p_correct(&result);
    }

    #[test]
    fn matmul_identity_on_right() {
        let m = mon(0b0011, 8);
        let identity = mon(0, 8);
        let (phase, result) = m.matmul_internal(&identity);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes, m.modes);
        assert_weight_and_p_correct(&result);
    }

    #[test]
    fn matmul_self_is_identity() {
        let m = mon(0b0111, 8);
        let (phase, result) = m.matmul_internal(&m);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(result.modes.is_zero());
        assert_weight_and_p_correct(&result);
    }

    #[test]
    fn matmul_disjoint_phase_is_minus_one() {
        let a = mon(0b0011, 8);
        let b = mon(0b1100, 8);
        let (phase, result) = a.matmul_internal(&b);
        assert!((phase - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes.count_ones(), 4);
        assert_weight_and_p_correct(&result);
    }

    #[test]
    fn commutes_with_itself() {
        let m = mon(0b0011, 8);
        assert!(m.commutes_with_impl(&m));
    }

    #[test]
    fn commutes_disjoint_even_lengths() {
        let a = mon(0b0011, 8);
        let b = mon(0b1100, 8);
        assert!(a.commutes_with_impl(&b));
    }

    #[test]
    fn anticommutes_single_overlap_even_lengths() {
        let a = mon(0b0011, 8);
        let b = mon(0b0110, 8);
        assert!(!a.commutes_with_impl(&b));
    }

    #[test]
    fn commutes_single_modes_disjoint() {
        let a = mon(0b0001, 8);
        let b = mon(0b0010, 8);
        assert!(!a.commutes_with_impl(&b));
    }

    // --- Incremental weight/p correctness (see `weight_and_p_from_product`) ---
    //
    // The fast path in `matmul_internal` relies on the parallel-prefix
    // XOR-scan being linear under XOR of its input: `p(A^B) == p(A) ^ p(B)`.
    // This was verified analytically and against 3.3M+ trials of an
    // independent simulation before implementation; the tests below port
    // that verification into the suite so a future change can't silently
    // break it. Every test compares `matmul_internal`'s incrementally
    // computed `weight`/`p` against `compute_weight_for`/`weight_and_p_for`
    // (the from-scratch, unchanged-since-inception reference) as the oracle.

    /// Deterministic splitmix64 PRNG — avoids a `rand` dev-dependency for
    /// what's otherwise a one-file, test-only need.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn random_bitset(rng: &mut Rng, n_modes: usize) -> Bitset {
        let n_words = (n_modes + 63) / 64;
        let mut words: Vec<u64> = (0..n_words).map(|_| rng.next_u64()).collect();
        let rem = n_modes % 64;
        if rem != 0 {
            let mask = (1u64 << rem) - 1;
            *words.last_mut().unwrap() &= mask;
        }
        Bitset::from_words(words)
    }

    fn random_mon(rng: &mut Rng, n_modes: usize) -> MajoranaMonomial {
        let modes = random_bitset(rng, n_modes);
        let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
        MajoranaMonomial { modes, n_modes, is_number_preserving: true, weight, p }
    }

    #[test]
    fn weight_matches_reference_exhaustive_small() {
        // Exhaustive over `a`, strided over `b`, for every small system size
        // (mirrors the pre-implementation simulation's coverage).
        for n_qubits in 1usize..=6 {
            let n_modes = 2 * n_qubits;
            let space = 1u64 << n_modes;
            let stride = (space / 37).max(1);
            for a_bits in 0..space {
                let a = mon(a_bits, n_modes);
                let mut b_bits = 0u64;
                while b_bits < space {
                    let b = mon(b_bits, n_modes);
                    let (_, result) = a.matmul_internal(&b);
                    assert_weight_and_p_correct(&result);
                    b_bits += stride;
                }
            }
        }
    }

    #[test]
    fn weight_matches_reference_randomized_multiword() {
        let mut rng = Rng(0xC0FFEE_D15EA5E5);
        for &n_qubits in &[30usize, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200] {
            let n_modes = 2 * n_qubits;
            for _ in 0..300 {
                let a = random_mon(&mut rng, n_modes);
                let b = random_mon(&mut rng, n_modes);
                let (_, result) = a.matmul_internal(&b);
                assert_weight_and_p_correct(&result);
            }
        }
    }

    #[test]
    fn weight_and_p_no_drift_over_chained_updates() {
        // Simulates a term being multiplied by 200 successive gate
        // generators in sequence, checking after every step that neither
        // the incrementally-tracked weight nor the cached `p` has drifted
        // from a full from-scratch recomputation.
        let mut rng = Rng(0x1234_5678_9ABC_DEF0);
        for &n_qubits in &[8usize, 33, 65, 128] {
            let n_modes = 2 * n_qubits;
            let mut term = random_mon(&mut rng, n_modes);
            for _ in 0..200 {
                let generator = random_mon(&mut rng, n_modes);
                let (_, next) = generator.matmul_internal(&term);
                assert_weight_and_p_correct(&next);
                term = next;
            }
        }
    }

    // --- `MajoranaBasis` (SoA word-plane kernels) vs `MajoranaMonomial`
    // (AoS, exhaustively tested above) cross-checks. This is the seam most
    // at risk in the SoA rewrite, since `weight`/`product` depend on the
    // cached `p` plane travelling correctly alongside `modes`.

    fn planes_of(m: &MajoranaMonomial, stride: usize) -> (Vec<u64>, Vec<u64>) {
        let mut g0 = vec![0u64; stride];
        let mut g1 = vec![0u64; stride];
        MajoranaBasis::term_into_planes(m, m.n_modes, [&mut g0, &mut g1]);
        (g0, g1)
    }

    fn assert_majorana_basis_matches(a: &MajoranaMonomial, b: &MajoranaMonomial, stride: usize) {
        let (a0, a1) = planes_of(a, stride);
        let (b0, b1) = planes_of(b, stride);
        let a_planes = [a0.as_slice(), a1.as_slice()];
        let b_planes = [b0.as_slice(), b1.as_slice()];
        let ctx = || format!("a.modes={a0:?} b.modes={b0:?}");

        assert_eq!(
            MajoranaBasis::commutes(a_planes, b_planes),
            a.commutes_with_impl(b),
            "commutes mismatch for {}", ctx(),
        );
        assert_eq!(MajoranaBasis::weight(a_planes, a.n_modes), a.weight, "weight mismatch for {}", ctx());

        // gen=a, term=b => a @ b, matching `a.matmul_internal(b)`.
        let (expected_phase, expected_result) = a.matmul_internal(b);
        let mut out0 = vec![0u64; stride];
        let mut out1 = vec![0u64; stride];
        let phase = MajoranaBasis::product(b_planes, a_planes, [&mut out0, &mut out1]);
        assert!((phase - expected_phase).norm() < 1e-10, "phase mismatch for {}", ctx());
        let result = MajoranaBasis::term_from_planes([&out0, &out1], a.n_modes);
        assert_eq!(result.modes, expected_result.modes, "product modes mismatch for {}", ctx());
        assert_eq!(result.p, expected_result.p, "product p mismatch for {}", ctx());
        assert_eq!(result.weight, expected_result.weight, "product weight mismatch for {}", ctx());

        for fock in 0u64..16 {
            assert_eq!(
                MajoranaBasis::trace(a_planes, a.n_modes, fock),
                a.trace_fock_state_impl(fock),
                "trace mismatch for {} fock={fock}", ctx(),
            );
        }

        assert_eq!(MajoranaBasis::key_eq(a_planes, b_planes), *a == *b, "key_eq mismatch for {}", ctx());
        if MajoranaBasis::key_eq(a_planes, b_planes) {
            assert_eq!(
                MajoranaBasis::key_hash(a_planes), MajoranaBasis::key_hash(b_planes),
                "key_eq monomials must key_hash equally for {}", ctx(),
            );
        }
    }

    #[test]
    fn majorana_basis_matches_aos_exhaustive_small() {
        for n_qubits in 1usize..=4 {
            let n_modes = 2 * n_qubits;
            let stride = MajoranaBasis::stride_words(n_modes);
            let space = 1u64 << n_modes;
            for a_bits in 0..space {
                let a = mon(a_bits, n_modes);
                for b_bits in 0..space {
                    let b = mon(b_bits, n_modes);
                    assert_majorana_basis_matches(&a, &b, stride);
                }
            }
        }
    }

    #[test]
    fn majorana_basis_matches_aos_randomized_multiword() {
        let mut rng = Rng(0xFEED_FACE_C0FF_EE00);
        for &n_qubits in &[30usize, 33, 64, 100, 128] {
            let n_modes = 2 * n_qubits;
            let stride = MajoranaBasis::stride_words(n_modes);
            for _ in 0..100 {
                let a = random_mon(&mut rng, n_modes);
                let b = random_mon(&mut rng, n_modes);
                assert_majorana_basis_matches(&a, &b, stride);
            }
        }
    }

    #[test]
    fn majorana_basis_key_eq_and_hash_ignore_p_plane() {
        // Two monomials with identical modes must be key_eq (and key_hash
        // equally) regardless of what garbage sits in the (unused-for-
        // identity) p plane — merge's parallel-batch correctness depends on
        // key_eq/key_hash agreeing, and neither may read `p`.
        let stride = 1;
        let a = mon(0b0101, 8);
        let (a0, a1) = planes_of(&a, stride);
        let mut a1_garbage = a1.clone();
        a1_garbage[0] ^= 0xDEAD_BEEF;
        assert!(MajoranaBasis::key_eq([&a0, &a1], [&a0, &a1_garbage]));
        assert_eq!(
            MajoranaBasis::key_hash([&a0, &a1]), MajoranaBasis::key_hash([&a0, &a1_garbage]),
        );
        let c = mon(0b1111, 8);
        let (c0, c1) = planes_of(&c, stride);
        assert!(!MajoranaBasis::key_eq([&a0, &a1], [&c0, &c1]));
    }

    // --- Narrow/even-generator fast path (`classify_gen`/`classify_p_shape`/
    // `commutes_fast`/`product_fast`): the exhaustive/randomized tests above
    // already exercise the dispatch (they call `MajoranaBasis::commutes`/
    // `product` directly), but only up to `n_qubits=128` with fully random
    // bitmasks. These target the realistic gate shapes specifically —
    // same-qubit, adjacent-qubit, the qubit-63/64 boundary, non-adjacent
    // (`p`-wide) pairs, a real wide-even `_xx_plus_yy_terms` pattern, and
    // `from_x`'s real odd-weight pattern — plus a dedicated large randomized
    // sweep and a standalone check of the `result_len` identity.

    fn set_bit(words: &mut [u64], bit: usize) {
        words[bit / 64] |= 1u64 << (bit % 64);
    }

    #[test]
    fn classify_gen_locates_sites_correctly() {
        const N_MODES: usize = 200;
        let stride = MajoranaBasis::stride_words(N_MODES);

        let (g0, g1) = planes_of(&mon_bits(vec![0u64; stride], N_MODES), stride);
        assert_eq!(classify_gen([g0.as_slice(), g1.as_slice()]), (0, GenShape::Identity));

        // Weight 1: mode bit 130 (word 2, bit 2; qubit 65).
        let mut words = vec![0u64; stride];
        set_bit(&mut words, 130);
        let (g0, g1) = planes_of(&mon_bits(words, N_MODES), stride);
        assert_eq!(
            classify_gen([g0.as_slice(), g1.as_slice()]),
            (1, GenShape::Weight1(GenSite { word: 2, mask: 1 << 2 })),
        );

        // Weight 2, same qubit: mode bits 130 and 131 (both qubit 65).
        let mut words = vec![0u64; stride];
        set_bit(&mut words, 130);
        set_bit(&mut words, 131);
        let (g0, g1) = planes_of(&mon_bits(words, N_MODES), stride);
        assert_eq!(
            classify_gen([g0.as_slice(), g1.as_slice()]),
            (2, GenShape::Weight2(GenSite { word: 2, mask: 1 << 2 }, GenSite { word: 2, mask: 1 << 3 })),
        );

        // Weight 3: falls back to Wide, but `gen_len` is still exact.
        let mut words = vec![0u64; stride];
        set_bit(&mut words, 130);
        set_bit(&mut words, 131);
        set_bit(&mut words, 132);
        let (g0, g1) = planes_of(&mon_bits(words, N_MODES), stride);
        assert_eq!(classify_gen([g0.as_slice(), g1.as_slice()]), (3, GenShape::Wide));
    }

    #[test]
    fn classify_p_shape_matches_qubit_adjacency() {
        // Same qubit -> Zero.
        let same = GenShape::Weight2(GenSite { word: 2, mask: 1 << 2 }, GenSite { word: 2, mask: 1 << 3 });
        assert_eq!(classify_p_shape(&same), PPlaneShape::Zero);

        // Adjacent qubits (63 and 64: mode bits 127 and 128) -> SingleBit at qubit 63.
        let adjacent = GenShape::Weight2(GenSite { word: 1, mask: 1u64 << 63 }, GenSite { word: 2, mask: 1 });
        assert_eq!(classify_p_shape(&adjacent), PPlaneShape::SingleBit { word: 0, mask: 1u64 << 63 });

        // Non-adjacent qubits -> Wide.
        let gap = GenShape::Weight2(GenSite { word: 0, mask: 1 << 4 }, GenSite { word: 0, mask: 1 << 8 });
        assert_eq!(classify_p_shape(&gap), PPlaneShape::Wide);

        // Weight1 / Wide always fall back for the p-plane.
        assert_eq!(classify_p_shape(&GenShape::Weight1(GenSite { word: 0, mask: 1 })), PPlaneShape::Wide);
        assert_eq!(classify_p_shape(&GenShape::Wide), PPlaneShape::Wide);
        assert_eq!(classify_p_shape(&GenShape::Identity), PPlaneShape::Zero);
    }

    #[test]
    fn result_len_identity_matches_symmetric_difference_cardinality() {
        for gen_bits in 0u32..64 {
            for term_bits in 0u32..64 {
                let gen_len = gen_bits.count_ones() as i64;
                let term_len = term_bits.count_ones() as i64;
                let overlap = (gen_bits & term_bits).count_ones() as i64;
                let expected = (gen_bits ^ term_bits).count_ones() as i64;
                assert_eq!(
                    gen_len + term_len - 2 * overlap, expected,
                    "gen={gen_bits:#08b} term={term_bits:#08b}",
                );
            }
        }
    }

    #[test]
    fn majorana_fast_path_weight1_high_word() {
        let n_qubits = 100;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let mut gwords = vec![0u64; stride];
        set_bit(&mut gwords, 130); // qubit 65's even mode, alone (from_x-on-qubit-65-shaped)
        let gen = mon_bits(gwords, n_modes);
        assert_eq!(gen.modes.count_ones() % 2, 1);

        assert_majorana_basis_matches(&gen, &mon_bits(vec![0u64; stride], n_modes), stride);
        let mut rng = Rng(0x1111_2222_3333_4444);
        for _ in 0..30 {
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }

    #[test]
    fn majorana_fast_path_weight2_same_qubit_late_word() {
        let n_qubits = 100;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let mut gwords = vec![0u64; stride];
        set_bit(&mut gwords, 130); // qubit 65, both modes (an _rz_terms/_cp_terms-shaped number term)
        set_bit(&mut gwords, 131);
        let gen = mon_bits(gwords, n_modes);
        assert_eq!(gen.modes.count_ones() % 2, 0);

        let mut rng = Rng(0x5555_6666_7777_8888);
        for _ in 0..30 {
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }

    #[test]
    fn majorana_fast_path_weight2_adjacent_qubit_boundary() {
        let n_qubits = 100;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        // Qubit 63 (modes 126,127) and qubit 64 (modes 128,129) straddle
        // both the qubit-63/64 boundary and a mode-bit word boundary (bit
        // 127 is word 1's top bit; bit 128 is word 2's bottom bit) — an
        // `_xx_plus_yy_terms`/`from_swap`-shaped adjacent-qubit hop.
        let mut gwords = vec![0u64; stride];
        set_bit(&mut gwords, 127);
        set_bit(&mut gwords, 128);
        let gen = mon_bits(gwords, n_modes);

        let mut rng = Rng(0x9999_AAAA_BBBB_CCCC);
        for _ in 0..30 {
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }

    #[test]
    fn majorana_fast_path_weight2_nonadjacent_qubit_p_wide_fallback() {
        let n_qubits = 100;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        // Qubit 10 and qubit 20: not currently emitted by the compiler, but
        // must stay correct defensively (`out[1]` must fall back to Wide).
        let mut gwords = vec![0u64; stride];
        set_bit(&mut gwords, 20);
        set_bit(&mut gwords, 40);
        let gen = mon_bits(gwords, n_modes);

        let mut rng = Rng(0xDDDD_EEEE_FFFF_0001);
        for _ in 0..30 {
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }

    #[test]
    fn majorana_fast_path_wide_even_xx_plus_yy_shaped_multiword() {
        // Reproduces `_xx_plus_yy_terms`'s real bit pattern (endpoints + the
        // full JW string between) for distant qubits spanning multiple
        // words: always even-weight, but wide (not narrow).
        let n_qubits = 80;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let (lo, hi) = (5usize, 70usize);
        let mut gwords = vec![0u64; stride];
        set_bit(&mut gwords, 2 * lo);
        for k in (lo + 1)..hi {
            set_bit(&mut gwords, 2 * k);
            set_bit(&mut gwords, 2 * k + 1);
        }
        set_bit(&mut gwords, 2 * hi + 1);
        let gen = mon_bits(gwords, n_modes);
        assert_eq!(gen.modes.count_ones() % 2, 0, "xx_plus_yy-shaped generators are always even-weight");

        let mut rng = Rng(0xA5A5_5A5A_1234_9876);
        for _ in 0..20 {
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }

    #[test]
    fn majorana_fast_path_from_x_odd_weight_matches_generic() {
        // `from_x`'s exact pattern: `modes = (1 << (2*i+1)) - 1`, always
        // odd-weight — the one real generator that exercises `commutes`'s
        // odd-`gen_len` fallback, for `i=0` (a real Weight1 shape) and
        // `i=50` (a wide odd shape).
        let n_qubits = 128;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let mut rng = Rng(0x0FF5_E7BA_DC0F_FEED);
        for &i in &[0usize, 50] {
            let mut gwords = vec![0u64; stride];
            for b in 0..(2 * i + 1) {
                set_bit(&mut gwords, b);
            }
            let gen = mon_bits(gwords, n_modes);
            assert_eq!(gen.modes.count_ones() % 2, 1, "from_x generators are always odd-weight");
            for _ in 0..20 {
                let term = random_mon(&mut rng, n_modes);
                assert_majorana_basis_matches(&gen, &term, stride);
            }
        }
    }

    #[test]
    fn majorana_fast_path_randomized_cross_word() {
        let n_qubits = 128;
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let mut rng = Rng(0x1357_9BDF_2468_ACE0);

        for _ in 0..10_000 {
            let gen_weight = (rng.next_u64() % 4) as usize;
            let mut gwords = vec![0u64; stride];
            for _ in 0..gen_weight {
                let bit = (rng.next_u64() as usize) % n_modes;
                set_bit(&mut gwords, bit);
            }
            let gen = mon_bits(gwords, n_modes);
            let term = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&gen, &term, stride);
        }
    }
}
