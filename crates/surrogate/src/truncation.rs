///
/// Legacy frequency truncation policy
///
use pyo3::prelude::*;

use propaq_core::truncators::{FrequencyTruncator, TermBudget, Truncator, WeightTruncator};

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncationPolicy {
    /// Drop monomials with more than this many trig factors (None = no limit).
    #[pyo3(get, set)]
    pub max_frequency: Option<usize>,
    /// Drop Pauli/Majorana terms with weight exceeding this value (None = no limit).
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    /// Live-term floor below which truncation is suppressed entirely.
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl FrequencyTruncationPolicy {
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, min_terms=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        min_terms: Option<usize>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            min_terms,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, min_terms={})",
            self.max_frequency
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.min_terms
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

impl FrequencyTruncationPolicy {
    pub fn decompose(&self) -> Vec<Truncator> {
        let mut ops = Vec::new();
        if let Some(frequency) = self.max_frequency {
            ops.push(Truncator::Frequency(FrequencyTruncator {
                frequency: Some(frequency),
            }));
        }
        if let Some(weight) = self.weight_cutoff {
            ops.push(Truncator::Weight(WeightTruncator {
                weight: Some(weight),
            }));
        }
        if self.min_terms.is_some() {
            ops.push(Truncator::TermBudget(TermBudget {
                min_terms: self.min_terms,
            }));
        }
        ops
    }
}
