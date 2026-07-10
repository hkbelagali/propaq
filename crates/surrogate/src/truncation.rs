///
/// Legacy frequency truncation policy
///
use pyo3::prelude::*;

use propaq_core::truncators::{FlushSchedule, FrequencyTruncator, TermBudget, Truncator, WeightTruncator};

const DEFAULT_MAX_TERMS: usize = 10_000_000;
/// Default finer merge cadence - merge (dedup duplicate Pauli strings into
/// the maps) after every gate that adds a term, without truncating. Merging
/// is O(1) per term regardless of prior history under the symbolic DAG
/// coefficient representation, so eager merging keeps live term count
/// minimal at all times rather than drifting toward path count within a
/// truncation window.
const DEFAULT_MERGE_MAX_TERMS: usize = 1;

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
    /// Finer lossless merge cadence: when this many terms accumulate in the
    /// outboxes since the last flush, collapse duplicate Pauli strings into the
    /// partition maps (no truncation). `None` disables the finer cadence.
    #[pyo3(get, set)]
    pub merge_max_terms: Option<usize>,
}

#[pymethods]
impl FrequencyTruncationPolicy {
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, truncation_range=None, merge_max_terms=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
        merge_max_terms: Option<usize>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
            // Default-on. Assign `policy.merge_max_terms = None` after
            // construction to disable the finer cadence.
            merge_max_terms: merge_max_terms.or(Some(DEFAULT_MERGE_MAX_TERMS)),
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
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, truncation_range=({}, {}), merge_max_terms={})",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.merge_max_terms.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

impl FrequencyTruncationPolicy {
    /// Decompose this legacy policy into the composable `(FlushSchedule,
    /// [Truncator])` shape the surrogate propagator runs internally. Each cutoff
    /// becomes its own core truncator (a bound that is `None` on both sides is
    /// omitted), and `merge_max_terms` becomes the schedule.
    pub fn decompose(&self) -> (FlushSchedule, Vec<Truncator>) {
        let schedule = FlushSchedule { merge_max_terms: self.merge_max_terms };
        let mut ops = Vec::new();
        if let Some(frequency) = self.max_frequency {
            ops.push(Truncator::Frequency(FrequencyTruncator { frequency: Some(frequency) }));
        }
        if let Some(weight) = self.weight_cutoff {
            ops.push(Truncator::Weight(WeightTruncator { weight: Some(weight) }));
        }
        if self.truncation_range.0.is_some() || self.truncation_range.1.is_some() {
            ops.push(Truncator::TermBudget(TermBudget {
                min_terms: self.truncation_range.0,
                max_terms: self.truncation_range.1,
            }));
        }
        (schedule, ops)
    }
}
