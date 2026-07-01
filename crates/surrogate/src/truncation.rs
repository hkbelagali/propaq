use pyo3::prelude::*;

/// Truncation policy for surrogate propagation.
///
/// Frequency truncation drops monomials whose trig factor count exceeds
/// `max_frequency`. A monomial with `l` factors has expected squared magnitude
/// `(1/2)^l` over uniform random angles, so this controls the approximation order.
///
/// `weight_cutoff` mirrors the numerical propagator's Pauli/Majorana weight cutoff.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncationPolicy {
    /// Drop monomials with more than this many trig factors (None = no limit).
    #[pyo3(get, set)]
    pub max_frequency: Option<usize>,
    /// Drop Pauli/Majorana terms with weight exceeding this value (None = no limit).
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
}

#[pymethods]
impl FrequencyTruncationPolicy {
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None))]
    pub fn new(max_frequency: Option<usize>, weight_cutoff: Option<u32>) -> Self {
        FrequencyTruncationPolicy { max_frequency, weight_cutoff }
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={})",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}
