use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
use std::hash::{Hash, Hasher};

use crate::bitset::Bitset;

/// we can't use u64 directly because that would only allow us to represent up to 32 fermionic modes 
/// we would likely need more than this, so we use a custom arbitrary-length bitset implementation, where we can control the boolean operations
/// In order to do this, we define conversion functions between Python integers and our Bitset
pub(crate) fn pyint_to_bitset(obj: &Bound<'_, PyAny>, _n_modes: usize) -> PyResult<Bitset> {
    let bit_length: usize = obj.call_method0("bit_length")?.extract()?;
    let byte_len = (bit_length + 7) / 8;
    let bytes: Vec<u8> = obj.call_method1("to_bytes", (byte_len, "little"))?.extract()?;
    Ok(Bitset::from_le_bytes(&bytes))
}

pub(crate) fn bitset_to_pyint(py: Python<'_>, bs: &Bitset) -> PyResult<PyObject> {
    let bytes = bs.to_le_bytes();
    let builtins = py.import("builtins")?;
    let int_type = builtins.getattr("int")?;
    Ok(int_type.call_method1("from_bytes", (bytes.as_slice(), "little"))?.into())
}

#[pyclass]
#[derive(Clone)]
pub struct MajoranaMonomial {
    pub modes: Bitset,
    #[pyo3(get)]
    pub n_modes: usize,
    #[pyo3(get)]
    pub is_number_preserving: bool,
}

#[pymethods]
impl MajoranaMonomial {
    #[new]
    #[pyo3(signature = (modes, n_modes, is_number_preserving = true))]
    fn new(modes: &Bound<'_, PyAny>, n_modes: usize, is_number_preserving: bool) -> PyResult<Self> {
        let bitset = pyint_to_bitset(modes, n_modes)?;
        Ok(MajoranaMonomial { modes: bitset, n_modes, is_number_preserving })
    }

    #[getter]
    fn modes(&self, py: Python<'_>) -> PyResult<PyObject> {
        bitset_to_pyint(py, &self.modes)
    }

    #[getter]
    fn length(&self) -> usize {
        self.modes.count_ones() as usize
    }

    #[getter]
    fn weight(&self) -> u32 {
        self.compute_weight()
    }

    fn overlap(&self, other: &MajoranaMonomial) -> u32 {
        (&self.modes & &other.modes).count_ones()
    }

    pub fn commutes_with(&self, other: &MajoranaMonomial) -> bool {
        if self.modes == other.modes {
            return true;
        }
        (self.length() * other.length() + self.overlap(other) as usize) % 2 == 0
    }

    fn resulting_weight(&self, other: &MajoranaMonomial) -> u32 {
        let result_modes = &self.modes ^ &other.modes;
        let temp = MajoranaMonomial {
            modes: result_modes,
            n_modes: self.n_modes,
            is_number_preserving: true,
        };
        temp.compute_weight()
    }

    fn __matmul__(&self, other: &MajoranaMonomial) -> PyResult<(Complex64, MajoranaMonomial)> {
        Ok(self.matmul_internal(other))
    }

    pub fn trace_with_fock_state(&self, fock_state: u64) -> f64 {
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

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let byte_length = (self.n_modes + 7) / 8;
        let mut bytes = self.modes.to_le_bytes();
        bytes.resize(byte_length, 0);
        PyBytes::new(py, &bytes)
    }

    fn __hash__(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.modes.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &MajoranaMonomial) -> bool {
        self.modes == other.modes
    }
}

/// We define stuff that we'll use in other parts of the Rust code here, so that we can keep the Python bindings clean and prevent duplicate stuff
impl MajoranaMonomial {
    /// Pauli weight of this monomial after Jordan-Wigner transformation.
    pub fn compute_weight(&self) -> u32 {
        let n_qubits = self.n_modes / 2;
        if n_qubits == 0 { return 0; }

        let qubit_mask = Bitset::all_ones_upto(n_qubits);

        let x_bits = compress_to_qubits(&self.modes, n_qubits, 0);
        let y_bits = compress_to_qubits(&self.modes, n_qubits, 1);

        let occupied = &x_bits | &y_bits;
        let single   = &x_bits ^ &y_bits;

        let mut p = single.clone();
        let mut shift = 1usize;
        while shift < n_qubits {
            p = &p ^ &(&p.shl(shift) & &qubit_mask);
            shift <<= 1;
        }

        let string = if single.count_ones() & 1 == 1 {
            &p ^ &qubit_mask
        } else {
            p
        };

        (&occupied | &string).count_ones()
    }

    pub(crate) fn matmul_internal(&self, other: &MajoranaMonomial) -> (Complex64, MajoranaMonomial) {
        let result_modes = &self.modes ^ &other.modes;
        let result = MajoranaMonomial {
            modes: result_modes,
            n_modes: self.n_modes,
            is_number_preserving: true,
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

impl PartialEq for MajoranaMonomial {
    fn eq(&self, other: &Self) -> bool { self.modes == other.modes }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mon(bits: u64, n_modes: usize) -> MajoranaMonomial {
        MajoranaMonomial {
            modes: Bitset::from_le_bytes(&bits.to_le_bytes()),
            n_modes,
            is_number_preserving: true,
        }
    }

    fn mon_bits(bits: Vec<u64>, n_modes: usize) -> MajoranaMonomial {
        MajoranaMonomial {
            modes: Bitset::from_words(bits),
            n_modes,
            is_number_preserving: true,
        }
    }

    #[test]
    fn hermiticity_exp_all_residues() {
        for (len, expected) in [(0,0),(1,0),(2,1),(3,1),(4,0),(5,0),(6,1),(7,1),(8,0)] {
            assert_eq!(hermiticity_exp(len), expected, "hermiticity_exp({len})");
        }
    }

    #[test]
    fn parity_disjoint_no_inversions() {
        // a={0,1} b={2,3}: no b-bit has any a-bit above it
        let a = Bitset::from_le_bytes(&[0b0011]);
        let b = Bitset::from_le_bytes(&[0b1100]);
        assert!(!resorting_parity(&a, &b));
    }

    #[test]
    fn parity_single_inversion() {
        // a={1} b={0}: a has one bit (1) above b's bit (0) → count=1, odd
        let a = Bitset::from_le_bytes(&[0b0010]);
        let b = Bitset::from_le_bytes(&[0b0001]);
        assert!(resorting_parity(&a, &b));
    }

    #[test]
    fn parity_two_inversions_even() {
        // a={2,3} b={0,1}: each b-bit has two a-bits above it → count=4, even
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
    fn weight_identity() {
        assert_eq!(mon(0, 8).compute_weight(), 0);
    }

    #[test]
    fn weight_single_gamma() {
        // bit 0 only (gamma_0): X on site 0 in JW → weight 1
        assert_eq!(mon(0b01, 8).compute_weight(), 1);
    }

    #[test]
    fn weight_number_operator() {
        // bits 0,1 (gamma_0 gamma_1 = number operator on site 0) → weight 1
        assert_eq!(mon(0b11, 8).compute_weight(), 1);
    }

    #[test]
    fn weight_four_x_modes() {
        // bits 0,2,4,6 (gamma_0 on each of 4 sites) → weight 4
        assert_eq!(mon(0b0101_0101, 8).compute_weight(), 4);
    }

    #[test]
    fn weight_large_n_modes() {
        // n_modes=128, single mode bit 0 → weight 1
        assert_eq!(mon(0b01, 128).compute_weight(), 1);
    }

    #[test]
    fn weight_multi_word_mode() {
        // bit 64 → gamma_0 on site 32 in a 64-qubit JW chain.
        // The JW string spans qubits 0..=32 → weight 33.
        let m = mon_bits(vec![0u64, 1u64], 128);
        assert_eq!(m.compute_weight(), 33);
    }

    #[test]
    fn trace_identity_any_fock() {
        let m = mon(0, 8);
        assert_eq!(m.trace_with_fock_state(0), 1.0);
        assert_eq!(m.trace_with_fock_state(0b1111), 1.0);
    }

    #[test]
    fn trace_unpaired_mode_is_zero() {
        // only bit 0 (gamma_0, no matching gamma_1) → number-changing → trace 0
        let m = mon(0b01, 8);
        assert_eq!(m.trace_with_fock_state(0), 0.0);
        assert_eq!(m.trace_with_fock_state(1), 0.0);
    }

    #[test]
    fn trace_site0_empty_fock() {
        // modes=0b11 (site 0), fock_state=0 (site 0 empty): 2*0-1 = -1, phase=1 → -1.0
        assert_eq!(mon(0b11, 8).trace_with_fock_state(0), -1.0);
    }

    #[test]
    fn trace_site0_occupied_fock() {
        // modes=0b11, fock_state=1 (site 0 occupied): 2*1-1 = 1, phase=1 → 1.0
        assert_eq!(mon(0b11, 8).trace_with_fock_state(1), 1.0);
    }

    #[test]
    fn trace_two_sites_all_combinations() {
        // modes=0b1111 (sites 0 and 1): p=2, phase = if (2/2)%2==0 → 1%2=1 → -1
        let m = mon(0b1111, 8);
        // fock=0b00: product=(-1)(-1)=1  → -1 * 1 = -1
        assert_eq!(m.trace_with_fock_state(0b00), -1.0);
        // fock=0b01: product=(1)(-1)=-1  → -1 * (-1) = 1... wait
        // phase=-1, product=-1 → -1 * -1 = 1? No: returns (phase*product) as f64
        // phase=-1, product=(2*1-1)*(2*0-1)=1*(-1)=-1 → (-1)*(-1) = 1
        assert_eq!(m.trace_with_fock_state(0b01), 1.0);
        // fock=0b10: product=(-1)*(1)=-1 → (-1)*(-1)=1
        assert_eq!(m.trace_with_fock_state(0b10), 1.0);
        // fock=0b11: product=(1)*(1)=1 → (-1)*(1)=-1
        assert_eq!(m.trace_with_fock_state(0b11), -1.0);
    }

    #[test]
    fn matmul_identity_on_left() {
        let identity = mon(0, 8);
        let m = mon(0b0011, 8);
        let (phase, result) = identity.matmul_internal(&m);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes, m.modes);
    }

    #[test]
    fn matmul_identity_on_right() {
        let m = mon(0b0011, 8);
        let identity = mon(0, 8);
        let (phase, result) = m.matmul_internal(&identity);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes, m.modes);
    }

    #[test]
    fn matmul_self_is_identity() {
        let m = mon(0b0111, 8);
        let (phase, result) = m.matmul_internal(&m);
        assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(result.modes.is_zero());
    }

    #[test]
    fn matmul_disjoint_phase_is_minus_one() {
        let a = mon(0b0011, 8);
        let b = mon(0b1100, 8);
        let (phase, result) = a.matmul_internal(&b);
        assert!((phase - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
        assert_eq!(result.modes.count_ones(), 4);
    }

    #[test]
    fn commutes_with_itself() {
        let m = mon(0b0011, 8);
        assert!(m.commutes_with(&m));
    }

    #[test]
    fn commutes_disjoint_even_lengths() {
        let a = mon(0b0011, 8);
        let b = mon(0b1100, 8);
        // length 2 * length 2 + overlap 0 = 4, even → commutes
        assert!(a.commutes_with(&b));
    }

    #[test]
    fn anticommutes_single_overlap_even_lengths() {
        let a = mon(0b0011, 8); // bits 0,1
        let b = mon(0b0110, 8); // bits 1,2
        // length 2 * length 2 + overlap 1 = 5, odd → anticommutes
        assert!(!a.commutes_with(&b));
    }

    #[test]
    fn commutes_single_modes_disjoint() {
        let a = mon(0b0001, 8);
        let b = mon(0b0010, 8);
        // length 1 * length 1 + overlap 0 = 1, odd → anticommutes
        assert!(!a.commutes_with(&b));
    }
}
impl Eq for MajoranaMonomial {}

impl Hash for MajoranaMonomial {
    fn hash<H: Hasher>(&self, state: &mut H) { self.modes.hash(state); }
}

fn compress_to_qubits(modes: &Bitset, n_qubits: usize, offset: usize) -> Bitset {
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

pub(crate) fn resorting_parity(a: &Bitset, b: &Bitset) -> bool {
    let mut count = 0u64;
    let mut remaining = b.clone();
    loop {
        let pos = remaining.trailing_zeros();
        if pos == usize::MAX { break; }
        count += a.count_ones_above(pos);
        remaining.clear_bit(pos);
    }
    (count & 1) != 0
}
