use super::*;
use crate::mps::build_environments;
use crate::mps::tests::{dense_statevector, random_mps};

#[allow(clippy::needless_range_loop)]
fn dense_expectation(psi: &[Complex64], n: usize, sites: &[(usize, PauliOp)]) -> Complex64 {
    let dim = 1usize << n;
    let mut out = psi.to_vec();
    for &(q, op) in sites {
        let m = op.matrix();
        let mut next = vec![Complex64::new(0.0, 0.0); dim];
        for basis in 0..dim {
            let s = (basis >> q) & 1;
            for sp in 0..2 {
                let amp = m[s][sp]; // <s|O|s'>, applied to ket component s'
                if amp == Complex64::new(0.0, 0.0) {
                    continue;
                }
                let src_basis = (basis & !(1 << q)) | (sp << q);
                next[basis] += amp * out[src_basis];
            }
        }
        out = next;
    }
    psi.iter()
        .zip(out.iter())
        .map(|(bra, ket)| bra.conj() * ket)
        .sum()
}

fn all_pauli_combinations(n: usize) -> Vec<Vec<(usize, PauliOp)>> {
    let mut result = vec![vec![]];
    for q in 0..n {
        let mut next = Vec::new();
        for combo in &result {
            for op in [None, Some(PauliOp::X), Some(PauliOp::Y), Some(PauliOp::Z)] {
                let mut c = combo.clone();
                if let Some(o) = op {
                    c.push((q, o));
                }
                next.push(c);
            }
        }
        result = next;
    }
    result
}

#[test]
fn window_expectation_matches_dense_exhaustive_n4() {
    let n = 4;
    let mps = random_mps(&[1, 2, 3, 2, 1], 123);
    let env = build_environments(&mps);
    let psi = dense_statevector(&mps);
    let norm2 = env.l[env.l.len() - 1][0];

    for sites in all_pauli_combinations(n) {
        let got = window_expectation(&mps, &env, &sites) / norm2;
        let expected = dense_expectation(&psi, n, &sites) / dense_norm(&psi);
        assert!(
            (got - expected).norm() < 1e-8,
            "sites={sites:?} got={got:?} expected={expected:?}"
        );
    }
}

fn dense_norm(psi: &[Complex64]) -> Complex64 {
    psi.iter().map(|a| a.conj() * a).sum()
}

#[test]
fn hand_picked_two_site_case() {
    let a0 = crate::mps::MpsTensor::new(
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.2, -0.3),
            Complex64::new(0.7, 0.1),
        ],
        1,
        2,
        2,
    );
    let a1 = crate::mps::MpsTensor::new(
        vec![
            Complex64::new(0.4, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-0.5, 0.2),
            Complex64::new(0.1, -0.1),
        ],
        2,
        2,
        1,
    );
    let mps = Mps::new(vec![a0, a1]).unwrap();
    let env = build_environments(&mps);
    let psi = dense_statevector(&mps);
    let norm2 = dense_norm(&psi);

    for sites in all_pauli_combinations(2) {
        let got = window_expectation(&mps, &env, &sites) / norm2;
        let expected = dense_expectation(&psi, 2, &sites) / norm2;
        assert!(
            (got - expected).norm() < 1e-8,
            "sites={sites:?} got={got:?} expected={expected:?}"
        );
    }
}
