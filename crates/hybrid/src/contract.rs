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
/// positions
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

/// \(\langle\Psi|P|\Psi\rangle\) for a single Pauli string
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

/// \(\sum_i c_i \langle\Psi|P_i|\Psi\rangle\) over every row of `terms`
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
#[path = "../tests/unit/contract.rs"]
mod tests;
