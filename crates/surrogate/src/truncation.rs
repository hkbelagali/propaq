use pyo3::prelude::*;

const DEFAULT_MAX_TERMS: usize = 10_000_000;
const DEFAULT_MAX_MONOMIALS: usize = 10_000_000;
const DEFAULT_MIN_MONOMIALS: usize = 5_000_000;

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
///
/// `monomial_range` is a *second*, independent `(min_monomials, max_monomials)`
/// pair, on its own axis: term count is a poor proxy for a symbolic
/// coefficient's actual size — a handful of terms can carry the overwhelming
/// majority of monomials while term count barely moves, so
/// `truncation_range`'s term-count trigger alone can fail to fire until
/// memory has already exploded. A flush's monomial-level (frequency)
/// truncation isn't triggered until the live monomial count exceeds
/// `max_monomials`; once triggered, it removes monomials (highest frequency
/// first, on top of whatever `max_frequency` alone already trimmed) down to
/// `max_monomials` — the target it aims to land on, not `min_monomials`.
/// `min_monomials` is only a floor: since removal happens in whole
/// highest-frequency buckets, a bucket bigger than what's needed to reach
/// `max_monomials` gets a partial removal rather than being discarded
/// entirely, so in practice truncation lands at or just above
/// `max_monomials`, not somewhere down near `min_monomials`.
/// Defaults to `(5_000_000, 10_000_000)`.
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
}

#[pymethods]
impl FrequencyTruncationPolicy {
    /// `monomial_range` defaults to `(Some(5_000_000), Some(10_000_000))` when
    /// omitted. To disable monomial-range-driven truncation entirely, set
    /// `policy.monomial_range = (None, None)` after construction.
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, truncation_range=None, monomial_range=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
        monomial_range: Option<(Option<usize>, Option<usize>)>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
            monomial_range: monomial_range
                .unwrap_or((Some(DEFAULT_MIN_MONOMIALS), Some(DEFAULT_MAX_MONOMIALS))),
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
    /// `max_monomials`) and how far it reduces the count once it does
    /// (down to `max_monomials`; `min_monomials` is only a floor against a
    /// single oversized top-frequency bucket removal overshooting further
    /// than necessary).
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
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, truncation_range=({}, {}), monomial_range=({}, {}))",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}
