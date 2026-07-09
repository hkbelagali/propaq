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
                let n_k = ((fock_state >> k) & 1) as i32;
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
        let total_parity = resorting_parity(&self.modes, &other.modes);
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

fn resorting_parity(a: &Bitset, b: &Bitset) -> bool {
    let a_words = a.as_words();
    let b_words = b.as_words();
    if a_words.is_empty() || b_words.is_empty() {
        return false;
    }

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
        assert!(!resorting_parity(&a, &b));
    }

    #[test]
    fn parity_single_inversion() {
        let a = Bitset::from_le_bytes(&[0b0010]);
        let b = Bitset::from_le_bytes(&[0b0001]);
        assert!(resorting_parity(&a, &b));
    }

    #[test]
    fn parity_two_inversions_even() {
        let a = Bitset::from_le_bytes(&[0b1100]);
        let b = Bitset::from_le_bytes(&[0b0011]);
        assert!(!resorting_parity(&a, &b));
    }

    #[test]
    fn parity_empty_b_is_false() {
        let a = Bitset::from_le_bytes(&[0xFF]);
        let b = Bitset::zero();
        assert!(!resorting_parity(&a, &b));
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
}
