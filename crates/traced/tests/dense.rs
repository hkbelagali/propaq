//! The built-in Pauli and Majorana bases against dense matrices.
//!
//! Each basis string has a matrix on at most 3 qubits. The test builds the
//! matrices from the layout rules in docs/architecture/tracer.md, section 4.
//! It checks the `Basis` functions against plain matrix algebra.

use num_complex::Complex64 as C;
use propaq_core::basis::{Basis, BasisKind};
use propaq_core::strings::BasisString;
use propaq_traced::majorana::Majorana;
use propaq_traced::pauli::Pauli;

type M = Vec<Vec<C>>;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

fn m2(a: [[C; 2]; 2]) -> M {
    a.iter().map(|r| r.to_vec()).collect()
}

fn eye(d: usize) -> M {
    (0..d).map(|i| (0..d).map(|j| if i == j { c(1.0, 0.0) } else { c(0.0, 0.0) }).collect()).collect()
}

fn mul(a: &M, b: &M) -> M {
    let d = a.len();
    let mut r = vec![vec![c(0.0, 0.0); d]; d];
    for i in 0..d {
        for k in 0..d {
            for j in 0..d {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}

fn scale(a: &M, s: C) -> M {
    a.iter().map(|r| r.iter().map(|x| x * s).collect()).collect()
}

fn kron(a: &M, b: &M) -> M {
    let (da, db) = (a.len(), b.len());
    let mut r = vec![vec![c(0.0, 0.0); da * db]; da * db];
    for i in 0..da {
        for j in 0..da {
            for k in 0..db {
                for l in 0..db {
                    r[i * db + k][j * db + l] = a[i][j] * b[k][l];
                }
            }
        }
    }
    r
}

fn close(a: &M, b: &M) -> bool {
    a.iter().zip(b).all(|(x, y)| x.iter().zip(y).all(|(p, q)| (p - q).norm() < 1e-12))
}

fn px() -> M { m2([[c(0., 0.), c(1., 0.)], [c(1., 0.), c(0., 0.)]]) }
fn py() -> M { m2([[c(0., 0.), c(0., -1.)], [c(0., 1.), c(0., 0.)]]) }
fn pz() -> M { m2([[c(1., 0.), c(0., 0.)], [c(0., 0.), c(-1., 0.)]]) }

/// Kron of per-qubit matrices. `ops[0]` acts on the lowest index bit.
fn kron_le(ops: &[M]) -> M {
    let mut it = ops.iter().rev();
    let mut r = it.next().unwrap().clone();
    for o in it {
        r = kron(&r, o);
    }
    r
}

/// Matrix of a Pauli string. Bit 2q is x, bit 2q+1 is z.
fn pauli_matrix(bits: u64, n: usize) -> M {
    let ops: Vec<M> = (0..n)
        .map(|q| {
            let (x, z) = ((bits >> (2 * q)) & 1, (bits >> (2 * q + 1)) & 1);
            match (x, z) {
                (0, 0) => eye(2),
                (1, 0) => px(),
                (0, 1) => pz(),
                _ => py(),
            }
        })
        .collect();
    kron_le(&ops)
}

/// Matrix of a Majorana monomial. Bit i is gamma_i.
fn majorana_matrix(bits: u64, n: usize) -> M {
    let gamma = |i: usize| {
        let (k, odd) = (i / 2, i % 2 == 1);
        let ops: Vec<M> = (0..n)
            .map(|q| if q < k { pz() } else if q == k { if odd { py() } else { px() } } else { eye(2) })
            .collect();
        kron_le(&ops)
    };
    let modes: Vec<usize> = (0..2 * n).filter(|i| (bits >> i) & 1 == 1).collect();
    let mut m = eye(1 << n);
    for &i in &modes {
        m = mul(&m, &gamma(i));
    }
    let r = if matches!(modes.len() % 4, 2 | 3) { c(0.0, 1.0) } else { c(1.0, 0.0) };
    scale(&m, r)
}

/// Number of qubits on which the matrix is not the identity.
fn support(m: &M, n: usize) -> u32 {
    let mut count = 0;
    for q in 0..n {
        let acts = [px(), py(), pz()].iter().any(|p| {
            let ops: Vec<M> = (0..n).map(|k| if k == q { p.clone() } else { eye(2) }).collect();
            let full = kron_le(&ops);
            !close(&mul(&full, m), &mul(m, &full))
        });
        if acts {
            count += 1;
        }
    }
    count
}

fn check<const W: usize, B: Basis<W>>(n: usize, matrix: fn(u64, usize) -> M, kind: BasisKind) {
    assert_eq!(B::KIND, kind);
    let dim = 1usize << n;
    let labels = 1u64 << (2 * n);
    for g in 1..labels {
        for sign in [1.0, -1.0] {
            let gen = BasisString::<W>::from_words(word::<W>(g));
            let ctx = B::make_signed_gen_context(&gen, sign);
            assert_eq!(B::generator(&ctx), &gen);
            let gm = matrix(g, n);
            for t in 0..labels {
                let tm = matrix(t, n);
                let key = BasisString::<W>::from_words(word::<W>(t));
                // Commutation: G T = +-T G.
                let (gt, tg) = (mul(&gm, &tm), mul(&tm, &gm));
                let anti = close(&gt, &scale(&tg, c(-1.0, 0.0)));
                assert!(anti || close(&gt, &tg), "g={g} t={t}");
                assert_eq!(B::anticommutes(&ctx, &key), anti, "anticommutes g={g} t={t}");
                assert_eq!(B::character(&ctx, &key), u32::from(anti));
                // Product: sign * G T = phase * T'.
                let (out, phase) = B::product(&ctx, &key);
                let out_bits = out.words()[0];
                let want = scale(&gt, c(sign, 0.0));
                assert!(close(&want, &scale(&matrix(out_bits, n), phase)), "product g={g} t={t} sign={sign}");
                // Fold: same character as `anticommutes`.
                let fold = B::linear_fold(&ctx).unwrap();
                assert_eq!(fold.eval(&key), u32::from(anti), "fold g={g} t={t}");
            }
        }
    }
    for t in 0..labels {
        let tm = matrix(t, n);
        let key = BasisString::<W>::from_words(word::<W>(t));
        assert_eq!(B::weight(&key, n), support(&tm, n), "weight t={t}");
        for diag in 0..dim as u64 {
            let want = tm[diag as usize][diag as usize].re;
            let got = B::trace(&key, n, &[diag]);
            assert!((got - want).abs() < 1e-12, "trace t={t} diag={diag}: {got} vs {want}");
        }
    }
}

fn word<const W: usize>(bits: u64) -> [u64; W] {
    let mut w = [0u64; W];
    w[0] = bits;
    w
}

#[test]
fn pauli_matches_dense() {
    for n in 1..=3 {
        check::<1, Pauli>(n, pauli_matrix, BasisKind::Pauli);
    }
    check::<4, Pauli>(2, pauli_matrix, BasisKind::Pauli);
}

#[test]
fn majorana_matches_dense() {
    for n in 1..=3 {
        check::<1, Majorana>(n, majorana_matrix, BasisKind::Majorana);
    }
    check::<4, Majorana>(2, majorana_matrix, BasisKind::Majorana);
}

#[test]
fn constants_are_the_pm1_class() {
    assert_eq!(<Pauli as Basis<2>>::BRANCHES, 2);
    assert_eq!(<Pauli as Basis<2>>::CHAR_MODULUS, 2);
    assert_eq!(<Majorana as Basis<2>>::PHASE_ORDER, 4);
}
