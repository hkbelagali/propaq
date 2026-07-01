use pyo3::prelude::*;

const DEFAULT_MAX_TERMS: usize = 10_000_000;

/// Truncation policy for surrogate propagation.
///
/// Frequency truncation drops monomials whose trig factor count exceeds
/// `max_frequency`. A monomial with `l` factors has expected squared magnitude
/// `(1/2)^l` over uniform random angles, so this controls the approximation order.
///
/// `weight_cutoff` mirrors the numerical propagator's Pauli/Majorana weight cutoff.
///
/// `truncation_range` mirrors the numerical propagator's `TruncationPolicy`:
/// a `(min_terms, max_terms)` pair. A flush is triggered once the live term
/// count reaches `max_terms`, and the lossy `max_frequency`/`weight_cutoff`
/// filtering is skipped (only lossless deduplication runs) while the term
/// count is below `min_terms`. Defaults to `(None, 10_000_000)`.
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
