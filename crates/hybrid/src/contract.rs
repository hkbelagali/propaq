//! Per-term window contraction and the parallel weighted sum across an
//! entire `TermSum`.
//!
//! The contraction does not know the basis. The basis gives one matrix for
//! each unit label (a `LocalTable`). A unit label is the `bits_per_unit`
//! bits of one unit of a term key. Label 0 must be the identity.

use num_complex::Complex64;
use rayon::prelude::*;

use propaq_core::store::{split_planes, Position, TermSum};

use crate::mps::{apply_transfer, Environments, Mps};

const ZERO: Complex64 = Complex64::new(0.0, 0.0);

/// Chunk-size floor below which the per-row loop runs serially
const PAR_MIN_LEN: usize = 512;

/// One site of a term: the unit index and its label.
pub type Site = (usize, u32);

/// The widest label table the contraction takes: 8 bits per unit.
pub const MAX_LABELS: usize = 256;

/// One `dim x dim` matrix for each unit label, row-major.
/// Entry `[label][sigma * dim + sigma_p]` is `<sigma|M_label|sigma_p>`.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalTable {
    dim: usize,
    n_labels: usize,
    data: Vec<Complex64>,
}

impl LocalTable {
    /// Builds a table from flat data of length `n_labels * dim * dim`.
    pub fn new(dim: usize, data: Vec<Complex64>) -> Result<Self, String> {
        if dim == 0 {
            return Err("local dimension must be at least 1".to_string());
        }
        let block = dim * dim;
        if data.is_empty() || !data.len().is_multiple_of(block) {
            return Err(format!(
                "local table has {} entries, not a multiple of dim^2 = {block}",
                data.len()
            ));
        }
        let n_labels = data.len() / block;
        if n_labels > MAX_LABELS {
            return Err(format!(
                "local table has {n_labels} labels, at most {MAX_LABELS} are allowed"
            ));
        }
        let table = LocalTable {
            dim,
            n_labels,
            data,
        };
        // Label 0 must be the identity: the contraction skips such units.
        for s in 0..dim {
            for sp in 0..dim {
                let want = if s == sp { Complex64::new(1.0, 0.0) } else { ZERO };
                if table.matrix(0)[s * dim + sp] != want {
                    return Err("local matrix of label 0 must be the identity".to_string());
                }
            }
        }
        Ok(table)
    }

    /// The qubit table of the Pauli basis. Label bit 0 is x, bit 1 is z.
    /// Labels: 0 = I, 1 = X, 2 = Z, 3 = Y (the Hermitian `i^(x z) X^x Z^z`).
    pub fn pauli() -> Self {
        let z = ZERO;
        let o = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let data = vec![
            o, z, z, o, // I
            z, o, o, z, // X
            o, z, z, -o, // Z
            z, -i, i, z, // Y
        ];
        LocalTable::new(2, data).expect("the Pauli table is valid")
    }

    /// The physical dimension of one unit.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of labels in the table.
    pub fn n_labels(&self) -> usize {
        self.n_labels
    }

    /// The matrix of `label`, flat row-major.
    #[inline]
    pub fn matrix(&self, label: u32) -> &[Complex64] {
        let block = self.dim * self.dim;
        let k = label as usize;
        &self.data[k * block..(k + 1) * block]
    }
}

/// Decodes a Pauli `TermSum` row's nontrivial sites directly from its sparse
/// positions. The label of a site is `x | z << 1`.
fn decode_plane_sites(row: &[Position], plane_span: usize, n_units: usize) -> Vec<Site> {
    let (xs, zs) = split_planes(row, plane_span);
    let mut sites = Vec::with_capacity(xs.len() + zs.len());
    let (mut i, mut j) = (0usize, 0usize);
    while i < xs.len() || j < zs.len() {
        let x_q = xs.get(i).map(|&p| p as usize);
        let z_q = zs.get(j).map(|&p| p as usize - plane_span);
        let (q, label) = match (x_q, z_q) {
            (Some(a), Some(b)) if a == b => {
                i += 1;
                j += 1;
                (a, 3)
            }
            (Some(a), Some(b)) if a < b => {
                i += 1;
                (a, 1)
            }
            (Some(_), Some(b)) => {
                j += 1;
                (b, 2)
            }
            (Some(a), None) => {
                i += 1;
                (a, 1)
            }
            (None, Some(b)) => {
                j += 1;
                (b, 2)
            }
            (None, None) => unreachable!("the loop condition guarantees one side is nonempty"),
        };
        if q < n_units {
            sites.push((q, label));
        }
    }
    sites
}

/// Decodes the nonzero units of a raw-word key. Unit `u` owns the bits
/// `u * bits .. (u + 1) * bits`. `bits` divides 64, so no unit crosses a word.
pub fn decode_word_sites(words: &[u64], bits: usize, n_units: usize) -> Vec<Site> {
    let mask: u64 = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let per_word = 64 / bits;
    let mut sites = Vec::new();
    for (wi, &word) in words.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            let unit_in_word = rest.trailing_zeros() as usize / bits;
            let shift = unit_in_word * bits;
            let label = (rest >> shift) & mask;
            rest &= !(mask << shift);
            let unit = wi * per_word + unit_in_word;
            if unit < n_units {
                sites.push((unit, label as u32));
            }
        }
    }
    sites
}

/// \(\langle\Psi|P|\Psi\rangle\) for a single term. `sites` holds the
/// nonidentity units in increasing order.
pub fn window_expectation(
    mps: &Mps,
    env: &Environments,
    table: &LocalTable,
    sites: &[Site],
) -> Complex64 {
    if sites.is_empty() {
        return env.l[env.l.len() - 1][0];
    }
    let s_min = sites[0].0;
    let s_max = sites[sites.len() - 1].0;

    let mut current = env.l[s_min].clone();
    let mut idx = 0;
    for (k, tensor) in mps.tensors.iter().enumerate().take(s_max + 1).skip(s_min) {
        let op = if idx < sites.len() && sites[idx].0 == k {
            let m = table.matrix(sites[idx].1);
            idx += 1;
            Some(m)
        } else {
            None
        };
        let bond_in = tensor.chi_l;
        current = apply_transfer(&current, bond_in, tensor, op);
    }

    let r = &env.r[s_max + 1];
    current.iter().zip(r.iter()).map(|(a, b)| a * b).sum()
}

/// Sums `row(i)` over `0..n`, in parallel when `n` is large.
fn parallel_sum(n: usize, row: impl Fn(usize) -> Complex64 + Sync + Send) -> Complex64 {
    if n >= PAR_MIN_LEN {
        (0..n)
            .into_par_iter()
            .fold(|| ZERO, |acc, i| acc + row(i))
            .reduce(|| ZERO, |a, b| a + b)
    } else {
        (0..n).map(row).fold(ZERO, |a, b| a + b)
    }
}

/// Checks that the table and the MPS agree on the unit dimension.
pub fn check_dims(mps: &Mps, table: &LocalTable) -> Result<(), String> {
    if mps.local_dim != table.dim() {
        return Err(format!(
            "the MPS has physical dimension {} but the basis has local dimension {}",
            mps.local_dim,
            table.dim()
        ));
    }
    Ok(())
}

/// \(\sum_i c_i \langle\Psi|P_i|\Psi\rangle\) over every row of a Pauli `terms`
/// store (two planes, x and z).
pub fn hybrid_expectation_sum(
    mps: &Mps,
    env: &Environments,
    table: &LocalTable,
    terms: &TermSum<f64>,
) -> Complex64 {
    let plane_span = terms.plane_span();
    parallel_sum(terms.len(), |i| {
        let sites = decode_plane_sites(terms.row_positions(i), plane_span, mps.n_sites);
        window_expectation(mps, env, table, &sites) * *terms.coeff(i)
    })
}

/// The same sum for terms in raw words: row `i` is
/// `words[i * stride .. (i + 1) * stride]` with `bits` bits per unit.
pub fn hybrid_expectation_sum_raw(
    mps: &Mps,
    env: &Environments,
    table: &LocalTable,
    words: &[u64],
    stride: usize,
    bits: usize,
    coeffs: &[Complex64],
) -> Complex64 {
    parallel_sum(coeffs.len(), |i| {
        let row = &words[i * stride..(i + 1) * stride];
        let sites = decode_word_sites(row, bits, mps.n_sites);
        window_expectation(mps, env, table, &sites) * coeffs[i]
    })
}

#[cfg(test)]
#[path = "../tests/unit/contract.rs"]
mod tests;
