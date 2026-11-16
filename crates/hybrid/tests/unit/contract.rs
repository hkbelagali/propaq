use super::*;
use crate::mps::build_environments;
use crate::mps::{Mps, MpsTensor};
use crate::mps::tests::{dense_statevector, random_mps};

#[derive(Clone, Copy, Debug)]
enum PauliOp {
    X,
    Y,
    Z,
}

impl PauliOp {
    /// Hard-coded matrices, independent of `LocalTable::pauli`.
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

    /// The label of the op in `LocalTable::pauli` (x = bit 0, z = bit 1).
    fn label(self) -> u32 {
        match self {
            PauliOp::X => 1,
            PauliOp::Z => 2,
            PauliOp::Y => 3,
        }
    }
}

fn to_sites(sites: &[(usize, PauliOp)]) -> Vec<Site> {
    sites.iter().map(|&(q, op)| (q, op.label())).collect()
}

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
    let table = LocalTable::pauli();
    let psi = dense_statevector(&mps);
    let norm2 = env.l[env.l.len() - 1][0];

    for sites in all_pauli_combinations(n) {
        let got = window_expectation(&mps, &env, &table, &to_sites(&sites)) / norm2;
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
    let table = LocalTable::pauli();
    let psi = dense_statevector(&mps);
    let norm2 = dense_norm(&psi);

    for sites in all_pauli_combinations(2) {
        let got = window_expectation(&mps, &env, &table, &to_sites(&sites)) / norm2;
        let expected = dense_expectation(&psi, 2, &sites) / norm2;
        assert!(
            (got - expected).norm() < 1e-8,
            "sites={sites:?} got={got:?} expected={expected:?}"
        );
    }
}

#[test]
fn pauli_table_has_the_expected_matrices() {
    let t = LocalTable::pauli();
    assert_eq!(t.dim(), 2);
    assert_eq!(t.n_labels(), 4);
    for (label, op) in [(1, PauliOp::X), (2, PauliOp::Z), (3, PauliOp::Y)] {
        let m = op.matrix();
        for (s, row) in m.iter().enumerate() {
            for (sp, entry) in row.iter().enumerate() {
                assert_eq!(t.matrix(label)[s * 2 + sp], *entry);
            }
        }
    }
}

#[test]
fn table_rejects_a_non_identity_label_zero() {
    let o = Complex64::new(1.0, 0.0);
    assert!(LocalTable::new(2, vec![o, ZERO, ZERO, -o]).is_err());
    assert!(LocalTable::new(2, vec![o, ZERO, ZERO]).is_err());
}

#[test]
fn word_decoder_reads_units_of_four_bits() {
    // Unit 0: label 5. Unit 3: label 0xA. Unit 17 (second word): label 1.
    let words = [0x5u64 | (0xAu64 << 12), 1u64 << 4];
    let sites = decode_word_sites(&words, 4, 40);
    assert_eq!(sites, vec![(0, 5), (3, 0xA), (17, 1)]);
    // Units at or past n_units are dropped.
    assert_eq!(decode_word_sites(&words, 4, 17), vec![(0, 5), (3, 0xA)]);
}

#[test]
fn word_decoder_reads_units_of_two_bits() {
    let words = [0b11u64 | (0b01u64 << 62)];
    assert_eq!(decode_word_sites(&words, 2, 32), vec![(0, 3), (31, 1)]);
}

#[test]
fn raw_sum_matches_term_by_term_dense_with_a_qudit_table() {
    // Two sites of dimension 3. Labels 0..3: I, and three fixed matrices.
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let mut data = Vec::new();
    for k in 0..4 {
        for s in 0..3 {
            for sp in 0..3 {
                data.push(match k {
                    0 => if s == sp { c(1.0, 0.0) } else { ZERO },
                    _ => c((s + 2 * sp + k) as f64 * 0.1, (s as f64 - sp as f64) * 0.05 * k as f64),
                });
            }
        }
    }
    let table = LocalTable::new(3, data).unwrap();
    let mut seed = 7u64;
    let mut tensor = |l: usize, r: usize| {
        let d = (0..l * 3 * r)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                c((seed >> 40) as f64 / 1e7 - 0.8, ((seed >> 20) & 0xFFFFF) as f64 / 1e6 - 0.5)
            })
            .collect();
        MpsTensor::new(d, l, 3, r)
    };
    let mps = Mps::new(vec![tensor(1, 2), tensor(2, 1)]).unwrap();
    assert_eq!(mps.local_dim, 3);
    let env = build_environments(&mps);
    // Keys with 2 bits per unit: unit u at bits 2u.
    let words = [0b0100u64, 0b1101, 0b0000, 0b1010];
    let coeffs = [c(0.5, 0.1), c(-1.0, 2.0), c(0.25, 0.0), c(0.0, -1.5)];
    let got = hybrid_expectation_sum_raw(&mps, &env, &table, &words, 1, 2, &coeffs);
    let mut want = ZERO;
    for (w, cf) in words.iter().zip(coeffs.iter()) {
        let (l0, l1) = ((w & 3) as u32, ((w >> 2) & 3) as u32);
        let mut acc = ZERO;
        for s0 in 0..3 {
            for s1 in 0..3 {
                for t0 in 0..3 {
                    for t1 in 0..3 {
                        let amp = |s0: usize, s1: usize| -> Complex64 {
                            (0..2)
                                .map(|a| mps.tensors[0].at(0, s0, a) * mps.tensors[1].at(a, s1, 0))
                                .sum()
                        };
                        acc += amp(s0, s1).conj()
                            * table.matrix(l0)[s0 * 3 + t0]
                            * table.matrix(l1)[s1 * 3 + t1]
                            * amp(t0, t1);
                    }
                }
            }
        }
        want += acc * cf;
    }
    assert!((got - want).norm() < 1e-10, "got {got:?} want {want:?}");
}

#[test]
fn dimension_mismatch_is_an_error() {
    let mps = random_mps(&[1, 2, 1], 5);
    let o = Complex64::new(1.0, 0.0);
    let mut data = vec![ZERO; 9];
    data[0] = o;
    data[4] = o;
    data[8] = o;
    let table = LocalTable::new(3, data).unwrap();
    assert!(check_dims(&mps, &table).is_err());
    assert!(check_dims(&mps, &LocalTable::pauli()).is_ok());
}
