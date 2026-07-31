//! Plain-Rust matrix-product-state representation and boundary-environment
//! precompute, shared by every Pauli term in a hybrid expectation-value call.
//!
//! Index convention used throughout this crate: every boundary/environment
//! vector (`Environments::l[k]`, `Environments::r[k]`, and the running
//! accumulator in `apply_transfer`) is a flattened `bond x bond` matrix
//! indexed `[ket_index * bond + bra_index]`. The ket index always contracts
//! against an un-conjugated tensor entry, the bra index against a conjugated one

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
        assert_eq!(data.len(), chi_l * phys * chi_r, "MpsTensor data length mismatch");
        MpsTensor { data, chi_l, phys, chi_r }
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
            return Err(format!("left boundary bond must be 1, got {}", tensors[0].chi_l));
        }
        if tensors[n_sites - 1].chi_r != 1 {
            return Err(format!("right boundary bond must be 1, got {}", tensors[n_sites - 1].chi_r));
        }
        for k in 0..n_sites {
            if tensors[k].phys != 2 {
                return Err(format!("site {k} has physical dimension {}, expected 2", tensors[k].phys));
            }
            if k + 1 < n_sites && tensors[k].chi_r != tensors[k + 1].chi_l {
                return Err(format!(
                    "bond mismatch between site {k} (chi_r={}) and site {} (chi_l={})",
                    tensors[k].chi_r, k + 1, tensors[k + 1].chi_l
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

fn apply_transfer_reverse(next: &[Complex64], bond_out: usize, tensor: &MpsTensor) -> Vec<Complex64> {
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

/// `<Psi|Psi>`, read off the fully-contracted left environment
pub fn norm_squared(env: &Environments) -> Complex64 {
    env.l[env.l.len() - 1][0]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn rand_c(seed: &mut u64) -> Complex64 {
        // xorshift64
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let re = ((*seed >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let im = ((*seed >> 32) as f64 / u32::MAX as f64) * 2.0 - 1.0;
        Complex64::new(re, im)
    }

    pub(crate) fn random_mps(bonds: &[usize], seed: u64) -> Mps {
        let mut s = seed | 1;
        let n = bonds.len() - 1;
        let tensors = (0..n)
            .map(|k| {
                let chi_l = bonds[k];
                let chi_r = bonds[k + 1];
                let data = (0..chi_l * 2 * chi_r).map(|_| rand_c(&mut s)).collect();
                MpsTensor::new(data, chi_l, 2, chi_r)
            })
            .collect();
        Mps::new(tensors).unwrap()
    }

    pub(crate) fn dense_statevector(mps: &Mps) -> Vec<Complex64> {
        let n = mps.n_sites;
        let dim = 1usize << n;
        let mut out = vec![ZERO; dim];
        // basis bit k (0 = least significant) selects qubit k's physical index.
        for basis in 0..dim {
            let mut vec_l = vec![ONE]; // length chi_l of site 0 == 1
            for k in 0..n {
                let t = &mps.tensors[k];
                let s = (basis >> k) & 1;
                let mut vec_next = vec![ZERO; t.chi_r];
                for l in 0..t.chi_l {
                    let c = vec_l[l];
                    if c == ZERO {
                        continue;
                    }
                    for r in 0..t.chi_r {
                        vec_next[r] += c * t.at(l, s, r);
                    }
                }
                vec_l = vec_next;
            }
            out[basis] = vec_l[0];
        }
        out
    }

    fn dense_norm_squared(psi: &[Complex64]) -> Complex64 {
        psi.iter().map(|a| a.conj() * a).sum()
    }

    #[test]
    fn norm_squared_matches_dense() {
        let mps = random_mps(&[1, 3, 4, 3, 1], 42);
        let env = build_environments(&mps);
        let psi = dense_statevector(&mps);
        let expected = dense_norm_squared(&psi);
        let got = norm_squared(&env);
        assert!((got - expected).norm() < 1e-9, "got {got:?}, expected {expected:?}");
    }

    #[test]
    fn left_and_right_full_contraction_agree() {
        let mps = random_mps(&[1, 2, 3, 2, 1], 7);
        let env = build_environments(&mps);
        let from_l = env.l[env.l.len() - 1][0];
        let from_r = env.r[0][0];
        assert!((from_l - from_r).norm() < 1e-9);
    }
}
