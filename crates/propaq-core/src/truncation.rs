use pyo3::prelude::*;

#[pyclass(subclass)]
#[derive(Clone)]
pub struct TruncationPolicy {
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    #[pyo3(get, set)]
    pub coeff_cutoff: f64,
}

#[pymethods]
impl TruncationPolicy {
    #[new]
    #[pyo3(signature = (weight_cutoff=None, coeff_cutoff=0.0))]
    fn new(weight_cutoff: Option<u32>, coeff_cutoff: f64) -> Self {
        TruncationPolicy { weight_cutoff, coeff_cutoff }
    }

    fn should_truncate(&self, weight: u32, abs_coeff: f64) -> bool {
        self.weight_cutoff.map_or(false, |wc| weight > wc) || abs_coeff < self.coeff_cutoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_high_weight() {
        let p = TruncationPolicy { weight_cutoff: Some(2), coeff_cutoff: 0.1 };
        assert!(p.should_truncate(3, 1.0));
    }

    #[test]
    fn truncate_low_coeff() {
        let p = TruncationPolicy { weight_cutoff: Some(5), coeff_cutoff: 0.5 };
        assert!(p.should_truncate(2, 0.49));
    }

    #[test]
    fn keep_within_both_cutoffs() {
        let p = TruncationPolicy { weight_cutoff: Some(5), coeff_cutoff: 0.1 };
        assert!(!p.should_truncate(3, 0.5));
    }

    #[test]
    fn weight_boundary_exact_keeps() {
        // weight == cutoff → not strictly greater → keep
        let p = TruncationPolicy { weight_cutoff: Some(3), coeff_cutoff: 0.0 };
        assert!(!p.should_truncate(3, 0.1));
        assert!(p.should_truncate(4, 0.1));
    }

    #[test]
    fn coeff_boundary_exact_keeps() {
        // abs_coeff == cutoff → not strictly less → keep
        let p = TruncationPolicy { weight_cutoff: Some(10), coeff_cutoff: 0.5 };
        assert!(!p.should_truncate(1, 0.5));
        assert!(p.should_truncate(1, 0.4999));
    }

    #[test]
    fn truncate_both_conditions() {
        let p = TruncationPolicy { weight_cutoff: Some(2), coeff_cutoff: 0.5 };
        assert!(p.should_truncate(5, 0.1));
    }

    #[test]
    fn zero_cutoffs_keep_nothing_nonzero_weight() {
        let p = TruncationPolicy { weight_cutoff: Some(0), coeff_cutoff: 0.0 };
        assert!(p.should_truncate(1, 1.0));  // weight 1 > 0
        assert!(!p.should_truncate(0, 1.0)); // weight 0, coeff fine
    }

    #[test]
    fn none_weight_cutoff_never_truncates_on_weight() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.0 };
        assert!(!p.should_truncate(100, 1.0));
        assert!(!p.should_truncate(1000, 0.5));
    }

    #[test]
    fn none_weight_cutoff_still_truncates_on_coeff() {
        let p = TruncationPolicy { weight_cutoff: None, coeff_cutoff: 0.5 };
        assert!(p.should_truncate(100, 0.1));
        assert!(!p.should_truncate(100, 0.6));
    }
}
