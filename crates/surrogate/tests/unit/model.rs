use super::*;
use crate::symcoeff::{GateParam, SymbolicCoeff};
use num_complex::Complex64;
use propaq_core::coeff::CoeffRepr;

fn build_shared_model() -> (SurrogateModel, Vec<SymbolicCoeff>, Vec<f64>) {
    let phase = Complex64::new(0.0, -1.0);
    let mut base = SymbolicCoeff::from_scalar(1.0);
    let _ = base.apply_rotation(&GateParam::symbolic(0), phase);

    let overlaps = [1.5f64, -0.5, 2.0];
    let coeffs: Vec<SymbolicCoeff> = overlaps
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut c = base.clone();
            let _ = c.apply_rotation(&GateParam::symbolic(1 + i as u32), phase);
            c
        })
        .collect();

    let (tape, roots) = SymbolicCoeff::compile_batch(coeffs.clone());
    let terms: Vec<SurrogateTerm> = overlaps
        .iter()
        .zip(&roots)
        .map(|(&overlap, &root)| SurrogateTerm { overlap, root })
        .collect();

    (
        SurrogateModel::new(terms, tape, 4),
        coeffs,
        overlaps.to_vec(),
    )
}

#[test]
fn evaluate_matches_the_old_per_term_compile_algorithm() {
    let (model, coeffs, overlaps) = build_shared_model();
    let params = [0.3, 0.7, 1.1, 1.9];
    let lut = SurrogateModel::make_lut(&params);

    let expected: f64 = overlaps
        .iter()
        .zip(&coeffs)
        .map(|(&overlap, c)| overlap * c.compile().evaluate(&lut))
        .sum();

    let got = model.evaluate(&params);
    assert!(
        (got - expected).abs() < 1e-12,
        "got {got}, expected {expected}"
    );
}

#[test]
fn n_monomials_matches_the_original_node_count_sum() {
    let (model, coeffs, _overlaps) = build_shared_model();
    let expected: u128 = coeffs.iter().map(|c| c.monomial_count()).sum();
    assert_eq!(model.n_monomials(), expected);
}

#[test]
fn n_monomials_saturates_instead_of_wrapping_when_summing_many_huge_terms() {
    let mut coeffs = Vec::new();
    for i in 0..3 {
        let mut c = SymbolicCoeff::from_scalar(1.0 + i as f64);
        for _ in 0..135 {
            let other = c.clone();
            c.add_assign(other);
        }
        assert_eq!(
            c.monomial_count(),
            u128::MAX,
            "each term must itself be saturated already"
        );
        coeffs.push(c);
    }
    let (tape, roots) = SymbolicCoeff::compile_batch(coeffs);
    let terms: Vec<SurrogateTerm> = roots
        .iter()
        .map(|&root| SurrogateTerm { overlap: 1.0, root })
        .collect();
    let model = SurrogateModel::new(terms, tape, 1);
    assert_eq!(model.n_monomials(), u128::MAX);
}

#[test]
fn save_load_round_trips_evaluate_output() {
    let (model, _coeffs, _overlaps) = build_shared_model();
    let params = [0.3, 0.7, 1.1, 1.9];
    let before = model.evaluate(&params);

    let path = std::env::temp_dir().join(format!(
        "propaq_surrogate_model_test_{}.bin",
        std::process::id()
    ));
    let path_str = path.to_str().unwrap();
    model.save(path_str).expect("save should succeed");
    let loaded = SurrogateModel::load(path_str).expect("load should succeed");
    let _ = std::fs::remove_file(&path);

    assert_eq!(loaded.n_params, model.n_params);
    assert_eq!(loaded.n_terms(), model.n_terms());
    let after = loaded.evaluate(&params);
    assert!(
        (after - before).abs() < 1e-12,
        "round-tripped {after} vs original {before}"
    );
}
