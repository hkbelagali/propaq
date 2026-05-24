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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_high_weight() {
        let p = TruncationPolicy { weight_cutoff: 2, coeff_cutoff: 0.1 };
        assert!(p.should_truncate(3, 1.0));
    }

    #[test]
    fn truncate_low_coeff() {
        let p = TruncationPolicy { weight_cutoff: 5, coeff_cutoff: 0.5 };
        assert!(p.should_truncate(2, 0.49));
    }

    #[test]
    fn keep_within_both_cutoffs() {
        let p = TruncationPolicy { weight_cutoff: 5, coeff_cutoff: 0.1 };
        assert!(!p.should_truncate(3, 0.5));
    }

    #[test]
    fn weight_boundary_exact_keeps() {
        // weight == cutoff → not strictly greater → keep
        let p = TruncationPolicy { weight_cutoff: 3, coeff_cutoff: 0.0 };
        assert!(!p.should_truncate(3, 0.1));
        assert!(p.should_truncate(4, 0.1));
    }

    #[test]
    fn coeff_boundary_exact_keeps() {
        // abs_coeff == cutoff → not strictly less → keep
        let p = TruncationPolicy { weight_cutoff: 10, coeff_cutoff: 0.5 };
        assert!(!p.should_truncate(1, 0.5));
        assert!(p.should_truncate(1, 0.4999));
    }

    #[test]
    fn truncate_both_conditions() {
        let p = TruncationPolicy { weight_cutoff: 2, coeff_cutoff: 0.5 };
        assert!(p.should_truncate(5, 0.1));
    }

    #[test]
    fn zero_cutoffs_keep_nothing_nonzero_weight() {
        let p = TruncationPolicy { weight_cutoff: 0, coeff_cutoff: 0.0 };
        assert!(p.should_truncate(1, 1.0));  // weight 1 > 0
        assert!(!p.should_truncate(0, 1.0)); // weight 0, coeff fine
    }
}
