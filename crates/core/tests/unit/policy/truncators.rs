use super::*;

#[test]
fn resolve_config_last_wins_and_none_disables() {
    let cfg = resolve_config(&[
        Truncator::Frequency(FrequencyTruncator { frequency: Some(9) }),
        Truncator::Frequency(FrequencyTruncator { frequency: Some(5) }), // last wins
        Truncator::Coefficient(CoefficientTruncator {
            coefficient: Some(1e-8),
        }),
        Truncator::Weight(WeightTruncator { weight: Some(12) }),
        Truncator::Weight(WeightTruncator { weight: None }), // None disables
        Truncator::TermBudget(TermBudget {
            min_terms: Some(1),
            max_terms: Some(100),
        }),
    ]);
    assert_eq!(cfg.frequency, Some(5));
    assert_eq!(cfg.coefficient, Some(1e-8));
    assert_eq!(cfg.weight, None);
    assert_eq!((cfg.min_terms, cfg.max_terms), (Some(1), Some(100)));
}

#[test]
fn resolve_config_monomial_budget_last_wins_and_none_disables() {
    let cfg = resolve_config(&[
        Truncator::MonomialBudget(MonomialBudget {
            min_monomials: Some(1),
            max_monomials: Some(1_000),
        }),
        Truncator::MonomialBudget(MonomialBudget {
            min_monomials: None,
            max_monomials: Some(500),
        }), // last wins wholesale
    ]);
    assert_eq!((cfg.min_monomials, cfg.max_monomials), (None, Some(500)));
}

#[test]
fn resolve_config_empty_is_all_none() {
    let cfg = resolve_config(&[]);
    assert_eq!(cfg.frequency, None);
    assert_eq!(cfg.weight, None);
    assert_eq!(cfg.max_terms, None);
    assert_eq!(cfg.min_monomials, None);
    assert_eq!(cfg.max_monomials, None);
}

#[test]
fn is_surrogate_only_flags_frequency() {
    assert!(Truncator::Frequency(FrequencyTruncator { frequency: Some(3) }).is_surrogate_only());
    assert!(!Truncator::Weight(WeightTruncator { weight: Some(2) }).is_surrogate_only());
    assert!(!Truncator::TermBudget(TermBudget {
        min_terms: None,
        max_terms: Some(9)
    })
    .is_surrogate_only());
    assert!(!Truncator::Coefficient(CoefficientTruncator {
        coefficient: Some(1e-3)
    })
    .is_surrogate_only());
}

#[test]
fn is_surrogate_only_flags_monomial_budget() {
    assert!(Truncator::MonomialBudget(MonomialBudget {
        min_monomials: None,
        max_monomials: Some(9)
    })
    .is_surrogate_only());
}

#[test]
fn is_surrogate_only_flags_simplify() {
    assert!(Truncator::Simplify(Simplify { enabled: true }).is_surrogate_only());
}

#[test]
fn resolve_config_simplify_last_wins_and_defaults_to_false() {
    let cfg = resolve_config(&[]);
    assert!(
        !cfg.simplify,
        "simplify must default to false when no Simplify truncator is present"
    );

    let cfg = resolve_config(&[
        Truncator::Simplify(Simplify { enabled: true }),
        Truncator::Simplify(Simplify { enabled: false }), // last wins
    ]);
    assert!(!cfg.simplify);

    let cfg = resolve_config(&[Truncator::Simplify(Simplify { enabled: true })]);
    assert!(cfg.simplify);
}

#[test]
fn truncate_high_weight() {
    let p = TruncationPolicy {
        weight_cutoff: Some(2),
        coeff_cutoff: 0.1,
        truncation_range: (None, None),
    };
    assert!(p.should_truncate(3, 1.0));
}

#[test]
fn truncate_low_coeff() {
    let p = TruncationPolicy {
        weight_cutoff: Some(5),
        coeff_cutoff: 0.5,
        truncation_range: (None, None),
    };
    assert!(p.should_truncate(2, 0.49));
}

#[test]
fn keep_within_both_cutoffs() {
    let p = TruncationPolicy {
        weight_cutoff: Some(5),
        coeff_cutoff: 0.1,
        truncation_range: (None, None),
    };
    assert!(!p.should_truncate(3, 0.5));
}

#[test]
fn weight_boundary_exact_keeps() {
    let p = TruncationPolicy {
        weight_cutoff: Some(3),
        coeff_cutoff: 0.0,
        truncation_range: (None, None),
    };
    assert!(!p.should_truncate(3, 0.1));
    assert!(p.should_truncate(4, 0.1));
}

#[test]
fn coeff_boundary_exact_keeps() {
    let p = TruncationPolicy {
        weight_cutoff: Some(10),
        coeff_cutoff: 0.5,
        truncation_range: (None, None),
    };
    assert!(!p.should_truncate(1, 0.5));
    assert!(p.should_truncate(1, 0.4999));
}

#[test]
fn truncate_both_conditions() {
    let p = TruncationPolicy {
        weight_cutoff: Some(2),
        coeff_cutoff: 0.5,
        truncation_range: (None, None),
    };
    assert!(p.should_truncate(5, 0.1));
}

#[test]
fn zero_cutoffs_keep_nothing_nonzero_weight() {
    let p = TruncationPolicy {
        weight_cutoff: Some(0),
        coeff_cutoff: 0.0,
        truncation_range: (None, None),
    };
    assert!(p.should_truncate(1, 1.0)); // weight 1 > 0
    assert!(!p.should_truncate(0, 1.0)); // weight 0, coeff fine
}

#[test]
fn none_weight_cutoff_never_truncates_on_weight() {
    let p = TruncationPolicy {
        weight_cutoff: None,
        coeff_cutoff: 0.0,
        truncation_range: (None, None),
    };
    assert!(!p.should_truncate(100, 1.0));
    assert!(!p.should_truncate(1000, 0.5));
}

#[test]
fn none_weight_cutoff_still_truncates_on_coeff() {
    let p = TruncationPolicy {
        weight_cutoff: None,
        coeff_cutoff: 0.5,
        truncation_range: (None, None),
    };
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
    let p = TruncationPolicy {
        weight_cutoff: None,
        coeff_cutoff: 0.0,
        truncation_range: (Some(100), None),
    };
    assert_eq!(p.truncation_range.0, Some(100));
    assert_eq!(p.truncation_range.1, None);
}

#[test]
fn truncation_range_max_only() {
    let p = TruncationPolicy {
        weight_cutoff: None,
        coeff_cutoff: 0.0,
        truncation_range: (None, Some(10_000_000)),
    };
    assert_eq!(p.truncation_range.0, None);
    assert_eq!(p.truncation_range.1, Some(10_000_000));
}

#[test]
fn truncation_range_both() {
    let p = TruncationPolicy {
        weight_cutoff: None,
        coeff_cutoff: 0.0,
        truncation_range: (Some(50), Some(1_000)),
    };
    assert_eq!(p.truncation_range, (Some(50), Some(1_000)));
}
