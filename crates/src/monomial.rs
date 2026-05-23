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

fn resorting_parity(a: &Bitset, b: &Bitset) -> bool {
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
