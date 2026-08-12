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

#[allow(clippy::needless_range_loop)]
pub(crate) fn dense_statevector(mps: &Mps) -> Vec<Complex64> {
    let n = mps.n_sites;
    let dim = 1usize << n;
    let mut out = vec![ZERO; dim];

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
    assert!(
        (got - expected).norm() < 1e-9,
        "got {got:?}, expected {expected:?}"
    );
}

#[test]
fn left_and_right_full_contraction_agree() {
    let mps = random_mps(&[1, 2, 3, 2, 1], 7);
    let env = build_environments(&mps);
    let from_l = env.l[env.l.len() - 1][0];
    let from_r = env.r[0][0];
    assert!((from_l - from_r).norm() < 1e-9);
}
