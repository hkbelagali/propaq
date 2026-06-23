use pyo3::prelude::*;

const DEFAULT_MAX_TERMS: usize = 10_000_000;

/// Controls when and how terms are discarded during propagation.
///
/// Arguments:
///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
///         None disables weight-based truncation.
///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
///     truncation_range: (min_terms, max_terms) pair. Truncation fires when the term
///         count reaches max_terms and will not reduce it below min_terms.
///         Defaults to (None, $10^7$).
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TruncationPolicy {
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    #[pyo3(get, set)]
    pub coeff_cutoff: f64,
    pub truncation_range: (Option<usize>, Option<usize>),
}

#[pymethods]
impl TruncationPolicy {
    /// Initialize the truncation policy.
    ///
    /// Arguments:
    ///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
    ///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
    ///     truncation_range: (min_terms, max_terms) pair. Truncation is triggered when the
    ///         term count reaches max_terms and will not reduce it below min_terms.
    #[new]
    #[pyo3(signature = (weight_cutoff=None, coeff_cutoff=0.0, truncation_range=None))]
    fn new(
        weight_cutoff: Option<u32>,
        coeff_cutoff: f64,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
    ) -> Self {
        TruncationPolicy {
            weight_cutoff,
            coeff_cutoff,
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

    /// Return True if a term with *weight* and |coefficient| *abs_coeff* should be discarded.
    fn should_truncate(&self, weight: u32, abs_coeff: f64) -> bool {
        self.weight_cutoff.map_or(false, |wc| weight > wc) || abs_coeff < self.coeff_cutoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_high_weight() {
        let p = TruncationPolicy { weight_cutoff: Some(2), coeff_cutoff: 0.1, truncation_range: (None, None) };
        assert!(p.should_truncate(3, 1.0));
    }

    #[test]
    fn truncate_low_coeff() {
        let p = TruncationPolicy { weight_cutoff: Some(5), coeff_cutoff: 0.5, truncation_range: (None, None) };
        assert!(p.should_truncate(2, 0.49));
    }

    #[test]
    fn keep_within_both_cutoffs() {
        let p = TruncationPolicy { weight_cutoff: Some(5), coeff_cutoff: 0.1, truncation_range: (None, None) };
        assert!(!p.should_truncate(3, 0.5));
    }

    #[test]
    fn weight_boundary_exact_keeps() {
        // weight == cutoff → not strictly greater → keep
        let p = TruncationPolicy { weight_cutoff: Some(3), coeff_cutoff: 0.0, truncation_range: (None, None) };
        assert!(!p.should_truncate(3, 0.1));
        assert!(p.should_truncate(4, 0.1));
    }

    #[test]
    fn coeff_boundary_exact_keeps() {
        // abs_coeff == cutoff → not strictly less → keep
        let p = TruncationPolicy { weight_cutoff: Some(10), coeff_cutoff: 0.5, truncation_range: (None, None) };
        assert!(!p.should_truncate(1, 0.5));
        assert!(p.should_truncate(1, 0.4999));
    }

    #[test]
    fn truncate_both_conditions() {
        let p = TruncationPolicy { weight_cutoff: Some(2), coeff_cutoff: 0.5, truncation_range: (None, None) };
        assert!(p.should_truncate(5, 0.1));
    }

    #[test]
    fn zero_cutoffs_keep_nothing_nonzero_weight() {
        let p = TruncationPolicy { weight_cutoff: Some(0), coeff_cutoff: 0.0, truncation_range: (None, None) };
        assert!(p.should_truncate(1, 1.0));  // weight 1 > 0
        assert!(!p.should_truncate(0, 1.0)); // weight 0, coeff fine
    }

    #[test]
    fn none_weight_cutoff_never_truncates_on_weight() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.0, truncation_range: (None, None) };
        assert!(!p.should_truncate(100, 1.0));
        assert!(!p.should_truncate(1000, 0.5));
    }

    #[test]
    fn none_weight_cutoff_still_truncates_on_coeff() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.5, truncation_range: (None, None) };
        assert!(p.should_truncate(100, 0.1));
        assert!(!p.should_truncate(100, 0.6));
    }

    #[test]
    fn truncation_range_default() {
        let p = TruncationPolicy::new(None, 0.0, None);
        assert_eq!(p.truncation_range, (None, Some(10_000_000)));
    }

    #[test]
    fn truncation_range_min_only() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.0, truncation_range: (Some(100), None) };
        assert_eq!(p.truncation_range.0, Some(100));
        assert_eq!(p.truncation_range.1, None);
    }

    #[test]
    fn truncation_range_max_only() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.0, truncation_range: (None, Some(10_000_000)) };
        assert_eq!(p.truncation_range.0, None);
        assert_eq!(p.truncation_range.1, Some(10_000_000));
    }

    #[test]
    fn truncation_range_both() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.0, truncation_range: (Some(50), Some(1_000)) };
        assert_eq!(p.truncation_range, (Some(50), Some(1_000)));
    }
}
