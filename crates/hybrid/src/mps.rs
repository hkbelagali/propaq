//! Plain-Rust matrix-product-state representation and boundary-environment
//! precompute, shared by every Pauli term in a hybrid expectation-value call.
//!
//! Index convention used throughout this crate: every boundary/environment
//! vector (`Environments::l[k]`, `Environments::r[k]`, and the running
//! accumulator in `apply_transfer`) is a flattened `bond x bond` matrix
//! indexed `[ket_index * bond + bra_index]`. The ket index always contracts
//! against an un-conjugated tensor entry, the bra index against a conjugated one
//! 

use ndarray::ArrayView2;
use num_complex::Complex64;

const ZERO: Complex64 = Complex64::new(0.0, 0.0);
const ONE: Complex64 = Complex64::new(1.0, 0.0);

/// One MPS site tensor, row-major over `(left_bond, physical, right_bond)`.
#[derive(Clone)]
pub struct MpsTensor {
    data: Vec<Complex64>,
    pub chi_l: usize,
    pub phys: usize,
    pub chi_r: usize,
}

impl MpsTensor {
    pub fn new(data: Vec<Complex64>, chi_l: usize, phys: usize, chi_r: usize) -> Self {
        assert_eq!(
            data.len(),
            chi_l * phys * chi_r,
            "MpsTensor data length mismatch"
        );
        MpsTensor {
            data,
            chi_l,
            phys,
            chi_r,
        }
    }

    #[inline]
    pub fn at(&self, l: usize, s: usize, r: usize) -> Complex64 {
        self.data[(l * self.phys + s) * self.chi_r + r]
    }

    /// The tensor's raw flat data, `(left_bond, physical, right_bond)`
    /// row-major
    #[inline]
    fn data(&self) -> &[Complex64] {
        &self.data
    }
}

/// An open-boundary MPS: `n_sites` tensors with `chi_l[0] == chi_r[n_sites-1] == 1`
/// and adjacent bond dimensions agreeing.
pub struct Mps {
    pub tensors: Vec<MpsTensor>,
    pub n_sites: usize,
}

impl Mps {
    pub fn new(tensors: Vec<MpsTensor>) -> Result<Self, String> {
        let n_sites = tensors.len();
        if n_sites == 0 {
            return Err("Mps must have at least one site".to_string());
        }
        if tensors[0].chi_l != 1 {
            return Err(format!(
                "left boundary bond must be 1, got {}",
                tensors[0].chi_l
            ));
        }
        if tensors[n_sites - 1].chi_r != 1 {
            return Err(format!(
                "right boundary bond must be 1, got {}",
                tensors[n_sites - 1].chi_r
            ));
        }
        for k in 0..n_sites {
            if tensors[k].phys != 2 {
                return Err(format!(
                    "site {k} has physical dimension {}, expected 2",
                    tensors[k].phys
                ));
            }
            if k + 1 < n_sites && tensors[k].chi_r != tensors[k + 1].chi_l {
                return Err(format!(
                    "bond mismatch between site {k} (chi_r={}) and site {} (chi_l={})",
                    tensors[k].chi_r,
                    k + 1,
                    tensors[k + 1].chi_l
                ));
            }
        }
        Ok(Mps { tensors, n_sites })
    }
}

/// Precomputed left/right boundary environments, one per site plus one at
/// each open end.
pub struct Environments {
    pub l: Vec<Vec<Complex64>>,
    pub r: Vec<Vec<Complex64>>,
}

/// Applies one site's bra-ket transfer step to a running left-to-right
/// environment, optionally sandwiching the physical index with a 2x2
/// operator
#[allow(clippy::needless_range_loop)]
pub fn apply_transfer(
    current: &[Complex64],
    bond_in: usize,
    tensor: &MpsTensor,
    op: Option<&[[Complex64; 2]; 2]>,
) -> Vec<Complex64> {
    let phys = tensor.phys;
    let bond_out = tensor.chi_r;

    let current_mat = ArrayView2::from_shape((bond_in, bond_in), current).unwrap();

    let a_eff_owned: Vec<Complex64>;
    let a_eff_slice: &[Complex64] = match op {
        Some(o) => {
            let mut v = vec![ZERO; bond_in * phys * bond_out];
            for l in 0..bond_in {
                for sigma in 0..phys {
                    for a in 0..bond_out {
                        let mut acc = ZERO;
                        for sigma_p in 0..phys {
                            acc += tensor.at(l, sigma_p, a) * o[sigma][sigma_p];
                        }
                        v[(l * phys + sigma) * bond_out + a] = acc;
                    }
                }
            }
            a_eff_owned = v;
            &a_eff_owned
        }
        None => tensor.data(),
    };
    let a_eff_mat = ArrayView2::from_shape((bond_in, phys * bond_out), a_eff_slice).unwrap();

    let tmp = current_mat.t().dot(&a_eff_mat);
    let tmp_slice = tmp.as_slice().expect("dot() result is contiguous");

    let tmp_reshaped = ArrayView2::from_shape((bond_in * phys, bond_out), tmp_slice).unwrap();

    let a_conj: Vec<Complex64> = tensor.data().iter().map(|c| c.conj()).collect();
    let a_conj_mat = ArrayView2::from_shape((bond_in * phys, bond_out), &a_conj).unwrap();

    let out = tmp_reshaped.t().dot(&a_conj_mat);
    out.as_slice().expect("dot() result is contiguous").to_vec()
}

fn apply_transfer_reverse(
    next: &[Complex64],
    bond_out: usize,
    tensor: &MpsTensor,
) -> Vec<Complex64> {
    let phys = tensor.phys;
    let bond_in = tensor.chi_l;

    let a_mat = ArrayView2::from_shape((bond_in * phys, bond_out), tensor.data()).unwrap();
    let next_mat = ArrayView2::from_shape((bond_out, bond_out), next).unwrap();

    let tmp = a_mat.dot(&next_mat);
    let tmp_slice = tmp.as_slice().expect("dot() result is contiguous");

    let tmp_reshaped = ArrayView2::from_shape((bond_in, phys * bond_out), tmp_slice).unwrap();

    let a_conj: Vec<Complex64> = tensor.data().iter().map(|c| c.conj()).collect();
    let a_conj_mat = ArrayView2::from_shape((bond_in, phys * bond_out), &a_conj).unwrap();

    let out = tmp_reshaped.dot(&a_conj_mat.t());
    out.as_slice().expect("dot() result is contiguous").to_vec()
}

/// Builds the left and right boundary environments for `mps`, O(n_sites) total
pub fn build_environments(mps: &Mps) -> Environments {
    let n = mps.n_sites;

    let mut l = Vec::with_capacity(n + 1);
    l.push(vec![ONE]);
    for k in 0..n {
        let bond_in = mps.tensors[k].chi_l;
        let next = apply_transfer(&l[k], bond_in, &mps.tensors[k], None);
        l.push(next);
    }

    let mut r: Vec<Vec<Complex64>> = vec![Vec::new(); n + 1];
    r[n] = vec![ONE];
    for k in (0..n).rev() {
        let bond_out = mps.tensors[k].chi_r;
        r[k] = apply_transfer_reverse(&r[k + 1], bond_out, &mps.tensors[k]);
    }

    Environments { l, r }
}

/// $\langle\Psi|\Psi\rangle$, read off the fully-contracted left environment
pub fn norm_squared(env: &Environments) -> Complex64 {
    env.l[env.l.len() - 1][0]
}

#[cfg(test)]
#[path = "../tests/unit/mps.rs"]
pub(crate) mod tests;
