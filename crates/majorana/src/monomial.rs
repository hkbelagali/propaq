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

    fn trace_fock_state_impl(&self, fock_state: &Bitset) -> f64 {
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
                let n_k = fock_state.bit(k) as i32;
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

    pub fn weight_and_p_for(modes: &Bitset, n_modes: usize) -> (u32, Bitset) {
        let n_qubits = n_modes / 2;
        if n_qubits == 0 { return (0, Bitset::zero()); }
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = Self::compress_single_occupied(modes, n_qubits);
        let p = Self::scan_p(&single, n_qubits, &qubit_mask);
        let weight = Self::weight_from_parts(&single, &occupied, &p, &qubit_mask);
        (weight, p)
    }

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
    pub fn trace_with_fock_state(&self, fock_state: &Bound<'_, PyAny>) -> PyResult<f64> {
        let bs = pyint_to_bitset(fock_state, self.n_modes)?;
        Ok(self.trace_fock_state_impl(&bs))
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
    fn trace_with_fock_state(&self, fock_state: &Bitset) -> f64 { self.trace_fock_state_impl(fock_state) }
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

pub struct MajoranaBasis;

impl SoaBasis for MajoranaBasis {
    type Term = MajoranaMonomial;

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        // Mirrors `commutes_with_impl(self=term, other=gen)`.
        if term[0] == gen[0] {
            return true;
        }
        let overlap: u32 = term[0].iter().zip(gen[0]).map(|(a, b)| (a & b).count_ones()).sum();
        let term_len: usize = term[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        let gen_len: usize = gen[0].iter().map(|w| w.count_ones()).sum::<u32>() as usize;
        (term_len * gen_len + overlap as usize) % 2 == 0
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
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

    fn weight(term: [&[u64]; 2], n_units: usize) -> u32 {
        let n_qubits = n_units / 2;
        if n_qubits == 0 {
            return 0;
        }
        let modes = Bitset::from_slice(term[0]);
        let p = Bitset::from_slice(term[1]);
        let qubit_mask = Bitset::all_ones_upto(n_qubits);
        let (single, occupied) = MajoranaMonomial::compress_single_occupied(&modes, n_qubits);
        MajoranaMonomial::weight_from_parts(&single, &occupied, &p, &qubit_mask)
    }

    fn trace(term: [&[u64]; 2], n_units: usize, fock: &[u64]) -> f64 {
        // `trace_fock_state_impl` only reads `modes`/`n_modes`; the other
        // fields are irrelevant to it, so a throwaway monomial is fine here.
        let modes = Bitset::from_slice(term[0]);
        let m = MajoranaMonomial {
            modes,
            n_modes: n_units,
            is_number_preserving: false,
            weight: 0,
            p: Bitset::zero(),
        };
        let fock_bs = Bitset::from_slice(fock);
        m.trace_fock_state_impl(&fock_bs)
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
        let modes = Bitset::from_slice(term[0]);
        let p = Bitset::from_slice(term[1]);
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

    fn fock(bits: u64) -> Bitset {
        Bitset::from_le_bytes(&bits.to_le_bytes())
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
        assert_eq!(m.trace_fock_state_impl(&fock(0)), 1.0);
        assert_eq!(m.trace_fock_state_impl(&fock(0b1111)), 1.0);
    }

    #[test]
    fn trace_unpaired_mode_is_zero() {
        let m = mon(0b01, 8);
        assert_eq!(m.trace_fock_state_impl(&fock(0)), 0.0);
        assert_eq!(m.trace_fock_state_impl(&fock(1)), 0.0);
    }

    #[test]
    fn trace_site0_empty_fock() { assert_eq!(mon(0b11, 8).trace_fock_state_impl(&fock(0)), -1.0); }

    #[test]
    fn trace_site0_occupied_fock() { assert_eq!(mon(0b11, 8).trace_fock_state_impl(&fock(1)), 1.0); }

    #[test]
    fn trace_two_sites_all_combinations() {
        let m = mon(0b1111, 8);
        assert_eq!(m.trace_fock_state_impl(&fock(0b00)), -1.0);
        assert_eq!(m.trace_fock_state_impl(&fock(0b01)),  1.0);
        assert_eq!(m.trace_fock_state_impl(&fock(0b10)),  1.0);
        assert_eq!(m.trace_fock_state_impl(&fock(0b11)), -1.0);
    }

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

        for fock_bits in 0u64..16 {
            let fock_words = [fock_bits];
            assert_eq!(
                MajoranaBasis::trace(a_planes, a.n_modes, &fock_words),
                a.trace_fock_state_impl(&fock(fock_bits)),
                "trace mismatch for {} fock={fock_bits}", ctx(),
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
}
