///
/// Legacy frequency truncation policy
///
use pyo3::prelude::*;

use propaq_core::truncators::{FrequencyTruncator, TermBudget, Truncator, WeightTruncator};

const DEFAULT_MAX_TERMS: usize = 10_000_000;

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
    pub truncation_range: (Option<usize>, Option<usize>),
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl FrequencyTruncationPolicy {
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, truncation_range=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
        }
    }

    /// The (min_terms, max_terms) pair controlling when and how aggressively truncation fires.
    #[getter]
    fn truncation_range(&self) -> (Option<usize>, Option<usize>) {
        self.truncation_range
    }

    #[setter]
    fn set_truncation_range(&mut self, value: (Option<usize>, Option<usize>)) {
        self.truncation_range = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, truncation_range=({}, {}))",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
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
        if self.truncation_range.0.is_some() || self.truncation_range.1.is_some() {
            ops.push(Truncator::TermBudget(TermBudget {
                min_terms: self.truncation_range.0,
                max_terms: self.truncation_range.1,
            }));
        }
        ops
    }
}
