use pyo3::prelude::*;

#[pyclass(subclass)]
#[derive(Clone)]
pub struct TruncationPolicy {
    #[pyo3(get, set)]
    pub weight_cutoff: u32,
    #[pyo3(get, set)]
    pub coeff_cutoff: f64,
}

#[pymethods]
impl TruncationPolicy {
    #[new]
    fn new(weight_cutoff: u32, coeff_cutoff: f64) -> Self {
        TruncationPolicy { weight_cutoff, coeff_cutoff }
    }

    fn should_truncate(&self, weight: u32, abs_coeff: f64) -> bool {
        weight > self.weight_cutoff || abs_coeff < self.coeff_cutoff
    }
}
