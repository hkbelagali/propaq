//! Per-term window contraction and the parallel weighted sum across an
//! entire `TermSum`.

use num_complex::Complex64;
use rayon::prelude::*;

use propaq_core::store::{split_planes, Position, TermSum};

use crate::mps::{apply_transfer, Environments, Mps};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Chunk-size floor below which the per-row loop runs serially
const PAR_MIN_LEN: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliOp {
    X,
    Y,
    Z,
}

impl PauliOp {
    fn matrix(self) -> [[Complex64; 2]; 2] {
        let z = ZERO;
        let o = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        match self {
            PauliOp::X => [[z, o], [o, z]],
            PauliOp::Y => [[z, -i], [i, z]],
            PauliOp::Z => [[o, z], [z, -o]],
        }
    }
}

/// Decodes a `PauliTermSum` row's nontrivial sites directly from its sparse
/// positions, without materializing a `PauliString` or any dense plane.
///
/// The two planes are each ascending, so merging them yields the sites in
/// ascending order, which is what `window_expectation` walks.
fn decode_sites(row: &[Position], plane_span: usize, n_units: usize) -> Vec<(usize, PauliOp)> {
    let (xs, zs) = split_planes(row, plane_span);
    let mut sites = Vec::with_capacity(xs.len() + zs.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < xs.len() || j < zs.len() {
        let x_q = xs.get(i).map(|&p| p as usize);
        let z_q = zs.get(j).map(|&p| p as usize - plane_span);
        let (q, op) = match (x_q, z_q) {
            (Some(a), Some(b)) if a == b => {
                i += 1;
                j += 1;
                (a, PauliOp::Y)
            }
            (Some(a), Some(b)) if a < b => {
                i += 1;
                (a, PauliOp::X)
            }
            (Some(_), Some(b)) => {
                j += 1;
                (b, PauliOp::Z)
            }
            (Some(a), None) => {
                i += 1;
                (a, PauliOp::X)
            }
            (None, Some(b)) => {
                j += 1;
                (b, PauliOp::Z)
            }
            (None, None) => unreachable!("the loop condition guarantees one side is nonempty"),
        };
        if q < n_units {
            sites.push((q, op));
        }
    }
    sites
}

/// `<Psi|P|Psi>` for a single Pauli string
pub fn window_expectation(mps: &Mps, env: &Environments, sites: &[(usize, PauliOp)]) -> Complex64 {
    if sites.is_empty() {
        return env.l[env.l.len() - 1][0];
    }
    let s_min = sites[0].0;
    let s_max = sites[sites.len() - 1].0;

    let mut current = env.l[s_min].clone();
    let mut idx = 0;
    for k in s_min..=s_max {
        let op = if idx < sites.len() && sites[idx].0 == k {
            let m = sites[idx].1.matrix();
            idx += 1;
            Some(m)
        } else {
            None
        };
        let bond_in = mps.tensors[k].chi_l;
        current = apply_transfer(&current, bond_in, &mps.tensors[k], op.as_ref());
    }

    let r = &env.r[s_max + 1];
    current.iter().zip(r.iter()).map(|(a, b)| a * b).sum()
}

/// `sum_i coeff_i * <Psi|P_i|Psi>` over every row of `terms`
pub fn hybrid_expectation_sum(mps: &Mps, env: &Environments, terms: &TermSum<f64>) -> Complex64 {
    let n = terms.len();
    let plane_span = terms.plane_span();
    let compute_row = |i: usize| -> Complex64 {
        let sites = decode_sites(terms.row_positions(i), plane_span, mps.n_sites);
        window_expectation(mps, env, &sites) * *terms.coeff(i)
    };

    if n >= PAR_MIN_LEN {
        (0..n)
            .into_par_iter()
            .fold(|| ZERO, |acc, i| acc + compute_row(i))
            .reduce(|| ZERO, |a, b| a + b)
    } else {
        (0..n).map(compute_row).fold(ZERO, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mps::tests::{dense_statevector, random_mps};
    use crate::mps::build_environments;

    fn dense_expectation(psi: &[Complex64], n: usize, sites: &[(usize, PauliOp)]) -> Complex64 {
        let dim = 1usize << n;
        // Build the dense 2x2 matrices per active site, apply to psi via
        // repeated single-qubit matrix application, then take <psi|P|psi>.
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
        psi.iter().zip(out.iter()).map(|(bra, ket)| bra.conj() * ket).sum()
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
}
