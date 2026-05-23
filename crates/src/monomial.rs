use pyo3::prelude::*;
use pyo3::types::PyBytes;
use num_complex::Complex64;
use std::hash::{Hash, Hasher};

#[pyclass]
#[derive(Clone)]
pub struct MajoranaMonomial {
    #[pyo3(get)]
    pub modes: u64,
    #[pyo3(get)]
    pub n_modes: usize,
    #[pyo3(get)]
    pub is_number_preserving: bool,
}

#[pymethods]
impl MajoranaMonomial { 
    #[new]
    #[pyo3(signature = (modes, n_modes, is_number_preserving = true))]
    fn new(modes: u64, n_modes: usize, is_number_preserving: bool) -> Self {
        MajoranaMonomial {modes, n_modes, is_number_preserving}
    }

    #[getter]
    fn length(&self) -> usize {
        self.modes.count_ones() as usize
    }

    // Function for the weight attribute
    #[getter]
    fn weight(&self) -> u32 {
        self.compute_weight()
    }

    fn overlap(&self, other: &MajoranaMonomial) -> u32 { 
        (self.modes & other.modes).count_ones()
    }

    fn commutes_with(&self, other: &MajoranaMonomial) -> bool { 
        if self.modes == other.modes { 
            return true;
        }
        (self.length() * other.length() + self.overlap(other) as usize) % 2 == 0
    }

    fn resulting_weight(&self, other: &MajoranaMonomial) -> u32 { 
        let result_modes = self.modes ^ other.modes;
        let temp = MajoranaMonomial::new(result_modes, self.n_modes, true);
        temp.weight()
    }

    fn __matmul__(&self, other: &MajoranaMonomial) -> PyResult<(Complex64, MajoranaMonomial)> {
        let result_modes = self.modes ^ other.modes;
        let result = MajoranaMonomial::new(result_modes, self.n_modes, true);

        let r_a = hermiticity_exp(self.length());
        let r_b = hermiticity_exp(other.length());
        let r_c = hermiticity_exp(result.length());

        let total_parity = resorting_parity(self.modes, other.modes);

        let phase_exp = (r_a + r_b - r_c + 2 * (total_parity as i32)).rem_euclid(4);

        let phase = match phase_exp {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            3 => Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        };

        Ok((phase, result))
    }

    fn trace_with_fock_state(&self, fock_state: u64) -> f64 {
        let n_fermionic = self.n_modes / 2;
        let mut p = 0;
        let mut product = 1;

        for k in 0..n_fermionic { 
            let low = (self.modes >> (2 * k)) & 1;
            let high = (self.modes >> (2 * k + 1)) & 1;

            if low != high { 
                return 0.0;
            }

            if low == 1 { 
                let n_k = (fock_state >> k) & 1;
                product *= 2 * (n_k as i32) - 1;
                p += 1;
            }
        }

        let phase = if (p / 2) % 2 == 0 { 1 } else { -1 };
        (phase * product) as f64
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> { 
        let byte_length = (self.n_modes + 7) / 8;
        let bytes = self.modes.to_le_bytes();
        PyBytes::new(py, &bytes[..byte_length])
    }

    // Python facing hash and equality functions
    fn __hash__(&self) -> u64 { 
        let mut h = std::collections::hash_map::DefaultHasher::new();
        use std::hash::{Hash, Hasher};
        self.modes.hash(&mut h);
        h.finish()
    }

    fn __eq__(&self, other: &MajoranaMonomial) -> bool { 
        self.modes == other.modes
    }
}

// Public version of weight computation for use elsewhere
impl MajoranaMonomial {
    pub fn compute_weight(&self) -> u32 {
        let paired = self.modes | (self.modes >> 1);
        let even_mask: u64 = (0..self.n_modes).step_by(2).fold(0, |acc, i| acc | (1 << i));
        (paired & even_mask).count_ones()
    }
}

impl PartialEq for MajoranaMonomial {
    fn eq(&self, other: &Self) -> bool {
        self.modes == other.modes
    }
}

// Rust facing hash and equality stuff
impl Eq for MajoranaMonomial {}

impl Hash for MajoranaMonomial {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.modes.hash(state);
    }
}

fn hermiticity_exp(length: usize) -> i32 { 
    if matches!(length % 4, 0 | 1) { 0 } else { 1 } 
}

fn resorting_parity(a: u64, b: u64) -> bool { 
    let mut count = 0;
    let mut remaining = b;

    while remaining != 0 { 
        let lowest_bit = remaining & remaining.wrapping_neg();
        let pos = lowest_bit.trailing_zeros();
        count += (a >> (pos + 1)).count_ones();
        remaining ^= lowest_bit;
    }

    (count & 1) != 0
}
