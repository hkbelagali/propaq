///
/// Legacy frequency truncation policy
///
use pyo3::prelude::*;

use propaq_core::truncators::{
    FlushSchedule, FrequencyTruncator, MonomialBudget, TermBudget, Truncator, WeightTruncator,
};

const DEFAULT_MAX_TERMS: usize = 10_000_000;
const DEFAULT_MAX_MONOMIALS: usize = 10_000_000;
const DEFAULT_MIN_MONOMIALS: usize = 5_000_000;
/// Default finer merge cadence - once this many terms accumulate in the outboxes
/// since the last flush, do a lossless merge (dedup duplicate Pauli strings into
/// the maps) without truncating. Smaller than the truncation window (default
/// `DEFAULT_MAX_TERMS` = 10M) so several merges happen per truncation, keeping
/// within-window peak near the unique-term count instead of the path count.
const DEFAULT_MERGE_MAX_TERMS: usize = 2_000_000;

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
    pub monomial_range: (Option<usize>, Option<usize>),
    /// Finer lossless merge cadence: when this many terms accumulate in the
    /// outboxes since the last flush, collapse duplicate Pauli strings into the
    /// partition maps (no truncation). `None` disables the finer cadence.
    #[pyo3(get, set)]
    pub merge_max_terms: Option<usize>,
}

#[pymethods]
impl FrequencyTruncationPolicy {
    /// `monomial_range` defaults to `(Some(5_000_000), Some(10_000_000))` when
    /// omitted. To disable monomial-range-driven truncation entirely, set
    /// `policy.monomial_range = (None, None)` after construction.
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, truncation_range=None, monomial_range=None, merge_max_terms=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
        monomial_range: Option<(Option<usize>, Option<usize>)>,
        merge_max_terms: Option<usize>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
            monomial_range: monomial_range
                .unwrap_or((Some(DEFAULT_MIN_MONOMIALS), Some(DEFAULT_MAX_MONOMIALS))),
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

    /// The (min_monomials, max_monomials) pair controlling when a flush's
    /// monomial-level truncation fires (once live count exceeds
    /// `max_monomials`) and how far it reduces the count once it does.
    #[getter]
    fn monomial_range(&self) -> (Option<usize>, Option<usize>) {
        self.monomial_range
    }

    #[setter]
    fn set_monomial_range(&mut self, value: (Option<usize>, Option<usize>)) {
        self.monomial_range = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, truncation_range=({}, {}), monomial_range=({}, {}), merge_max_terms={})",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
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
        if self.monomial_range.0.is_some() || self.monomial_range.1.is_some() {
            ops.push(Truncator::MonomialBudget(MonomialBudget {
                min_monomials: self.monomial_range.0,
                max_monomials: self.monomial_range.1,
            }));
        }
        (schedule, ops)
    }
}
