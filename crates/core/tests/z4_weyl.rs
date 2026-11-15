//! Tests for the generic run loop and the m-branch path.
//!
//! The test basis is the Z_4 Weyl-Heisenberg group on qudits.
//! Unit u owns bits 4u..4u+3 as (x0, x1, z0, z1). A digit is lo + 2 * hi.
//! The Weyl operator is W(x, z) = X^x Z^z with X|j> = |j+1>, Z|j> = i^j |j>.
//! The branch table comes from the spec: U = exp(-i theta H / 2), H = (W_g + W_g^dag) / 2.
//!
//! Three references are used:
//!   * a dense matrix evolution (checks the table and the branch rule),
//!   * a per-row dictionary evolution (checks the loop, noise and cutoffs),
//!   * the same basis without a linear fold (checks the column evaluation).

use std::collections::HashMap;

use num_complex::Complex64;

use propaq_core::basis::{Basis, BasisKind};
use propaq_core::fold::LinearFold;
use propaq_core::inverted_index::InvertedIndex;
use propaq_core::noise_resolver::ResolvedNoise;
use propaq_core::operator_index::OperatorIndex;
use propaq_core::run::{run_at_width, RunGate, RunInputs};
use propaq_core::strings::BasisString;
use propaq_core::termsum::EmitCutoff;

const I: Complex64 = Complex64::new(0.0, 1.0);

fn ipow(e: u32) -> Complex64 {
    match e % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => I,
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    }
}

// ---------------------------------------------------------------- digits

type Digits = Vec<(u32, u32)>; // (x, z) per unit

fn digits<const W: usize>(t: &BasisString<W>, n: usize) -> Digits {
    (0..n)
        .map(|u| {
            let b = |k: usize| u32::from(t.test(4 * u + k));
            (b(0) + 2 * b(1), b(2) + 2 * b(3))
        })
        .collect()
}

fn key_of<const W: usize>(d: &Digits) -> BasisString<W> {
    let mut s = BasisString::<W>::zero();
    for (u, &(x, z)) in d.iter().enumerate() {
        for (k, v) in [x & 1, x >> 1, z & 1, z >> 1].into_iter().enumerate() {
            if v != 0 {
                s.set(4 * u + k);
            }
        }
    }
    s
}

// ---------------------------------------------------------------- the basis

const N_UNITS: usize = 2;

pub struct GenCtx<const W: usize> {
    gen: BasisString<W>,
    g: Digits,
    q: u32,
}

/// Z_4 Weyl basis with a linear fold.
pub struct Z4;
/// The same basis, but `linear_fold` returns `None`.
pub struct Z4NoFold;

fn make_ctx<const W: usize>(gen: &BasisString<W>) -> GenCtx<W> {
    let g = digits(gen, 16.min(W * 16));
    let q = g.iter().map(|&(x, z)| x * z).sum::<u32>() % 4;
    GenCtx { gen: *gen, g, q }
}

fn chi_r<const W: usize>(ctx: &GenCtx<W>, t: &BasisString<W>) -> (u32, u32) {
    let td = digits(t, ctx.g.len());
    let zg_xt: u32 = ctx.g.iter().zip(&td).map(|(g, t)| g.1 * t.0).sum();
    let zt_xg: u32 = ctx.g.iter().zip(&td).map(|(g, t)| t.1 * g.0).sum();
    ((zg_xt + 4 * 64 - zt_xg) % 4, zt_xg % 4)
}

fn z4_table(theta: f64, q: u32) -> Vec<Complex64> {
    let mu = if q.is_multiple_of(2) { Complex64::new(1.0, 0.0) } else { Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4) };
    let lam: Vec<Complex64> = (0..4).map(|j| mu * ipow(j)).collect();
    let mut tab = vec![Complex64::new(0.0, 0.0); 16];
    for s in 0..4u32 {
        let chi = ipow(s);
        let f: Vec<Complex64> = lam
            .iter()
            .map(|l| (Complex64::new(0.0, 0.25 * theta) * ((chi - 1.0) * l + (chi.conj() - 1.0) * l.conj())).exp())
            .collect();
        for k in 0..4u32 {
            let mut ck = Complex64::new(0.0, 0.0);
            for j in 0..4 {
                ck += f[j] * lam[j].powi(-(k as i32));
            }
            ck /= 4.0;
            tab[(s * 4 + k) as usize] = ck * ipow(q * k * (k.max(1) - 1) / 2 % 4);
        }
    }
    tab
}

macro_rules! impl_z4 {
    ($ty:ty, $fold:expr) => {
        impl<const W: usize> Basis<W> for $ty {
            const KIND: BasisKind = BasisKind::Custom;
            const BRANCHES: usize = 4;
            const CHAR_MODULUS: u32 = 4;
            const PHASE_ORDER: u32 = 4;
            type GenContext = GenCtx<W>;

            fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> GenCtx<W> {
                assert_eq!(sign, 1.0);
                make_ctx(gen)
            }
            fn generator(ctx: &GenCtx<W>) -> &BasisString<W> {
                &ctx.gen
            }
            fn anticommutes(ctx: &GenCtx<W>, t: &BasisString<W>) -> bool {
                chi_r(ctx, t).0 != 0
            }
            fn fold_generator(ctx: &GenCtx<W>) -> &BasisString<W> {
                &ctx.gen
            }
            fn product(ctx: &GenCtx<W>, t: &BasisString<W>) -> (BasisString<W>, Complex64) {
                <Self as Basis<W>>::branch(ctx, t, 1)
            }
            fn weight(t: &BasisString<W>, n_units: usize) -> u32 {
                digits(t, n_units).iter().filter(|&&(x, z)| x != 0 || z != 0).count() as u32
            }
            fn trace(t: &BasisString<W>, n_units: usize, diag: &[u64]) -> f64 {
                let d = digits(t, n_units);
                if d.iter().any(|&(x, _)| x != 0) {
                    return 0.0;
                }
                let b = diag.first().copied().unwrap_or(0);
                let e: u32 = d.iter().enumerate().map(|(u, &(_, z))| z * ((b >> u) & 1) as u32).sum();
                ipow(e).re
            }
            fn character(ctx: &GenCtx<W>, t: &BasisString<W>) -> u32 {
                chi_r(ctx, t).0
            }
            fn linear_fold(ctx: &GenCtx<W>) -> Option<LinearFold<W>> {
                if !$fold {
                    return None;
                }
                // chi = zg.xt - zt.xg (mod 4). Each bit gets a weight in 0..3.
                let mut masks = [BasisString::<W>::zero(); 3];
                for (u, &(xg, zg)) in ctx.g.iter().enumerate() {
                    let w = [zg % 4, (2 * zg) % 4, (4 - xg % 4) % 4, (4 - (2 * xg) % 4) % 4];
                    for (k, &wk) in w.iter().enumerate() {
                        if wk != 0 {
                            masks[wk as usize - 1].set(4 * u + [0, 1, 2, 3][k]);
                        }
                    }
                }
                Some(LinearFold { modulus: 4, masks, use_parity: false })
            }
            fn branch(ctx: &GenCtx<W>, t: &BasisString<W>, k: usize) -> (BasisString<W>, Complex64) {
                let td = digits(t, ctx.g.len());
                let (_, r) = chi_r(ctx, t);
                let key: Digits = td
                    .iter()
                    .zip(&ctx.g)
                    .map(|(t, g)| ((t.0 + k as u32 * g.0) % 4, (t.1 + k as u32 * g.1) % 4))
                    .collect();
                (key_of::<W>(&key), ipow(k as u32 * r))
            }
            fn branch_table(ctx: &GenCtx<W>, theta: f64) -> Vec<Complex64> {
                z4_table(theta, ctx.q)
            }
        }
    };
}
impl_z4!(Z4, true);
impl_z4!(Z4NoFold, false);

// ------------------------------------------------------------ random + dense

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn digits(&mut self, n: usize) -> Digits {
        (0..n).map(|_| (self.below(4) as u32, self.below(4) as u32)).collect()
    }
}

type Mat = Vec<Complex64>; // row-major, dim x dim
const DIM: usize = 16;

fn matmul(a: &Mat, b: &Mat) -> Mat {
    let mut c = vec![Complex64::new(0.0, 0.0); DIM * DIM];
    for i in 0..DIM {
        for k in 0..DIM {
            let aik = a[i * DIM + k];
            if aik.norm_sqr() == 0.0 {
                continue;
            }
            for j in 0..DIM {
                c[i * DIM + j] += aik * b[k * DIM + j];
            }
        }
    }
    c
}

fn dagger(a: &Mat) -> Mat {
    let mut c = vec![Complex64::new(0.0, 0.0); DIM * DIM];
    for i in 0..DIM {
        for j in 0..DIM {
            c[j * DIM + i] = a[i * DIM + j].conj();
        }
    }
    c
}

/// W(x, z)|j> = i^(z.j) |j + x> on each unit. Basis index = j0 + 4 j1.
fn dense_weyl(d: &Digits) -> Mat {
    let mut m = vec![Complex64::new(0.0, 0.0); DIM * DIM];
    for j in 0..DIM {
        let js = [j % 4, j / 4];
        let mut phase = 0u32;
        let mut out = [0usize; 2];
        for u in 0..N_UNITS {
            phase += d[u].1 * js[u] as u32;
            out[u] = (js[u] + d[u].0 as usize) % 4;
        }
        m[(out[0] + 4 * out[1]) * DIM + j] += ipow(phase);
    }
    m
}

fn expm_i(h: &Mat, scale: f64) -> Mat {
    // exp(-i scale h) by Taylor series. ||scale h|| is small in the tests.
    let mut term: Mat = (0..DIM * DIM).map(|k| if k / DIM == k % DIM { Complex64::new(1.0, 0.0) } else { Complex64::new(0.0, 0.0) }).collect();
    let mut sum = term.clone();
    let a: Mat = h.iter().map(|v| v * Complex64::new(0.0, -scale)).collect();
    for n in 1..60 {
        term = matmul(&term, &a);
        term.iter_mut().for_each(|v| *v /= n as f64);
        sum.iter_mut().zip(&term).for_each(|(s, t)| *s += t);
    }
    sum
}

fn dense_of(terms: &HashMap<Vec<u32>, Complex64>) -> Mat {
    let mut m = vec![Complex64::new(0.0, 0.0); DIM * DIM];
    for (k, c) in terms {
        let d: Digits = (0..N_UNITS).map(|u| (k[2 * u], k[2 * u + 1])).collect();
        let w = dense_weyl(&d);
        m.iter_mut().zip(&w).for_each(|(a, b)| *a += *c * b);
    }
    m
}

fn flat(d: &Digits) -> Vec<u32> {
    d.iter().flat_map(|&(x, z)| [x, z]).collect()
}

fn max_diff(a: &Mat, b: &Mat) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max)
}

fn nonzero_gen(rng: &mut Rng) -> Digits {
    loop {
        let g = rng.digits(N_UNITS);
        if g.iter().any(|&(x, z)| x != 0 || z != 0) {
            return g;
        }
    }
}

// ---------------------------------------------------------------- tests

#[test]
fn the_branch_rule_matches_dense_evolution_for_one_rotation() {
    let mut rng = Rng(7);
    let theta = 0.37;
    for _ in 0..200 {
        let g = nonzero_gen(&mut rng);
        let t = rng.digits(N_UNITS);
        let wg = dense_weyl(&g);
        let h: Mat = wg.iter().zip(dagger(&wg)).map(|(a, b)| a + b).collect();
        let u = expm_i(&h, theta / 4.0);
        let lhs = matmul(&matmul(&dagger(&u), &dense_weyl(&t)), &u);

        let ctx = make_ctx::<1>(&key_of::<1>(&g));
        let tk = key_of::<1>(&t);
        let chi = <Z4 as Basis<1>>::character(&ctx, &tk) as usize;
        let table = <Z4 as Basis<1>>::branch_table(&ctx, theta);
        let mut rhs: HashMap<Vec<u32>, Complex64> = HashMap::new();
        for k in 0..4usize {
            let f = table[chi * 4 + k];
            if f.norm_sqr() == 0.0 {
                continue;
            }
            let (child, ph) = if k == 0 { (tk, Complex64::new(1.0, 0.0)) } else { <Z4 as Basis<1>>::branch(&ctx, &tk, k) };
            *rhs.entry(flat(&digits(&child, N_UNITS))).or_default() += f * ph;
        }
        assert!(max_diff(&lhs, &dense_of(&rhs)) < 1e-10, "g={g:?} t={t:?}");
        if chi == 0 {
            assert!(max_diff(&lhs, &dense_weyl(&t)) < 1e-10);
        }
    }
}

fn random_terms(rng: &mut Rng, n: usize) -> Vec<(Vec<u64>, f64)> {
    let mut seen = HashMap::new();
    for _ in 0..n {
        let d = rng.digits(N_UNITS);
        let c = (rng.below(2000) as f64 - 1000.0) / 1000.0;
        seen.insert(flat(&d), (key_of::<1>(&d).words().to_vec(), c));
    }
    seen.into_values().collect()
}

fn terms_to_map(terms: Vec<(Vec<u64>, Complex64)>) -> HashMap<Vec<u32>, Complex64> {
    let mut m: HashMap<Vec<u32>, Complex64> = HashMap::new();
    for (w, c) in terms {
        let key = key_of_words(&w);
        *m.entry(flat(&digits(&key, N_UNITS))).or_default() += c;
    }
    m
}

fn key_of_words(w: &[u64]) -> BasisString<1> {
    BasisString::from_words([w[0]])
}

fn gates(rng: &mut Rng, n: usize) -> Vec<RunGate<Complex64>> {
    (0..n)
        .map(|i| RunGate {
            qiskit_gate_idx: Some(i),
            generator: key_of::<1>(&nonzero_gen(rng)).words().to_vec(),
            angle: 0.2 + 0.1 * rng.below(8) as f64,
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run<A: Basis<1>>(
    obs: &[(Vec<u64>, f64)],
    layers: &[Vec<RunGate<Complex64>>],
    partitions: usize,
    cutoff: &EmitCutoff,
    noise: Option<&ResolvedNoise>,
) -> propaq_core::run::RunOutput<Complex64> {
    let obs: Vec<(Vec<u64>, Complex64)> =
        obs.iter().map(|(w, c)| (w.clone(), Complex64::new(*c, 0.0))).collect();
    run_at_width::<A, Complex64, 1, u8>(&RunInputs {
        observable: &obs,
        layers,
        n_units: N_UNITS,
        partitions,
        cutoff,
        noise,
        collect_counts: true,
        diag_state: None,
        want_terms: true,
        log_gates: true,
        progress: None,
    })
    .expect("run")
}

#[test]
fn the_run_loop_matches_dense_evolution() {
    let mut rng = Rng(11);
    for partitions in [1usize, 3] {
        let obs = random_terms(&mut rng, 6);
        let layers = vec![gates(&mut rng, 3), gates(&mut rng, 3)];
        let out = run::<Z4>(&obs, &layers, partitions, &EmitCutoff::none(), None);

        let mut start: HashMap<Vec<u32>, Complex64> = HashMap::new();
        for (w, c) in &obs {
            start.insert(flat(&digits(&key_of_words(w), N_UNITS)), Complex64::new(*c, 0.0));
        }
        let mut dense = dense_of(&start);
        for gate in layers.iter().flatten() {
            let g = digits(&key_of_words(&gate.generator), N_UNITS);
            let wg = dense_weyl(&g);
            let h: Mat = wg.iter().zip(dagger(&wg)).map(|(a, b)| a + b).collect();
            let u = expm_i(&h, gate.angle / 4.0);
            dense = matmul(&matmul(&dagger(&u), &dense), &u);
        }
        let got = dense_of(&terms_to_map(out.terms.unwrap()));
        assert!(max_diff(&dense, &got) < 1e-9, "partitions = {partitions}: {}", max_diff(&dense, &got));
        assert_eq!(out.result.n_terms.len(), 6);
        assert_eq!(out.gates.len(), 6);
    }
}

#[test]
fn the_column_fold_and_the_row_scan_agree() {
    let mut rng = Rng(23);
    for partitions in [1usize, 2] {
        let obs = random_terms(&mut rng, 8);
        let layers = vec![gates(&mut rng, 4), gates(&mut rng, 4)];
        let cutoff = EmitCutoff { min_coeff: Some(1e-4), ..EmitCutoff::none() };
        let with_fold = run::<Z4>(&obs, &layers, partitions, &cutoff, None);
        let without = run::<Z4NoFold>(&obs, &layers, partitions, &cutoff, None);
        let (a, b) = (terms_to_map(with_fold.terms.unwrap()), terms_to_map(without.terms.unwrap()));
        assert_eq!(a.len(), b.len());
        for (k, v) in &a {
            assert!((v - b[k]).norm() < 1e-12);
        }
        assert_eq!(with_fold.result.n_terms, without.result.n_terms);
    }
}

#[test]
fn weight_noise_and_a_weight_cutoff_match_a_dictionary_reference() {
    let mut rng = Rng(31);
    let obs = random_terms(&mut rng, 6);
    let layers = vec![gates(&mut rng, 3), gates(&mut rng, 3)];
    let table = vec![1.0, 0.9, 0.7];
    let noise = ResolvedNoise::WeightTable(table.clone());
    let cutoff = EmitCutoff { max_weight: Some(1), ..EmitCutoff::none() };
    let out = run::<Z4>(&obs, &layers, 2, &cutoff, Some(&noise));

    // Reference: the same rules on a plain map.
    let weight = |k: &Vec<u32>| (0..N_UNITS).filter(|&u| k[2 * u] != 0 || k[2 * u + 1] != 0).count();
    let mut map: HashMap<Vec<u32>, Complex64> = HashMap::new();
    for (w, c) in &obs {
        let k = flat(&digits(&key_of_words(w), N_UNITS));
        if weight(&k) <= 1 {
            map.insert(k, Complex64::new(*c, 0.0));
        }
    }
    for layer in &layers {
        for (k, c) in map.iter_mut() {
            *c *= table[weight(k).min(table.len() - 1)];
        }
        for gate in layer {
            let gk = key_of_words(&gate.generator);
            let ctx = make_ctx::<1>(&gk);
            let tab = z4_table(gate.angle, ctx.q);
            let mut next: HashMap<Vec<u32>, Complex64> = HashMap::new();
            for (k, c) in &map {
                let t = key_of::<1>(&k.chunks(2).map(|p| (p[0], p[1])).collect());
                let chi = chi_r(&ctx, &t).0 as usize;
                if chi == 0 {
                    *next.entry(k.clone()).or_default() += *c;
                    continue;
                }
                *next.entry(k.clone()).or_default() += *c * tab[chi * 4];
                for kk in 1..4 {
                    let f = tab[chi * 4 + kk];
                    if f.norm_sqr() == 0.0 {
                        continue;
                    }
                    let (child, ph) = <Z4 as Basis<1>>::branch(&ctx, &t, kk);
                    let ck = flat(&digits(&child, N_UNITS));
                    if weight(&ck) > 1 {
                        continue;
                    }
                    *next.entry(ck).or_default() += *c * f * ph;
                }
            }
            map = next;
        }
    }
    let got = terms_to_map(out.terms.unwrap());
    for (k, v) in &map {
        let g = got.get(k).copied().unwrap_or_default();
        assert!((v - g).norm() < 1e-10, "{k:?}");
    }
    for (k, g) in &got {
        assert!(map.contains_key(k) || g.norm() < 1e-10, "{k:?}");
    }
}

// ------------------------------------------------------- fold evaluation

fn random_fold(rng: &mut Rng, modulus: u32, use_parity: bool) -> LinearFold<1> {
    let mut masks = [BasisString::<1>::zero(); 3];
    for (w, m) in masks.iter_mut().enumerate() {
        if modulus == 2 && w > 0 {
            continue;
        }
        for _ in 0..rng.below(12) {
            m.set(rng.below(64) as usize);
        }
    }
    LinearFold { modulus, masks, use_parity }
}

#[test]
fn eval_fold_matches_the_per_row_form() {
    let mut rng = Rng(99);
    // Many rows over few columns make some columns dense, others stay sparse.
    for (n_rows, span) in [(300usize, 64u64), (5000, 64), (700, 8)] {
        let mut store = OperatorIndex::<u16, 1>::with_default_width();
        for _ in 0..n_rows {
            let mut m = BasisString::<1>::zero();
            for _ in 0..1 + rng.below(6) {
                let wide = rng.below(100) == 0;
                m.set(rng.below(if wide { 64 } else { span }) as usize);
            }
            store.push(&m).unwrap();
        }
        let mut index = InvertedIndex::new(64);
        index.sync_to(&store);
        if n_rows == 5000 {
            assert!(index.dense_column_count() > 0, "the dense tier must be exercised");
        }
        if n_rows == 700 {
            assert!(index.dense_column_count() > 0 && index.dense_column_count() < 64, "the sparse tier must be exercised");
        }
        for (modulus, parity) in [(2u32, false), (2, true), (4, false)] {
            for _ in 0..30 {
                let fold = random_fold(&mut rng, modulus, parity);
                let (mut lo, mut hi) = (Vec::new(), Vec::new());
                assert!(index.eval_fold(&fold, &mut lo, &mut hi));
                for r in 0..n_rows {
                    let want = fold.eval(&store.row(r));
                    let got = ((lo[r >> 6] >> (r & 63)) & 1) as u32
                        + if hi.is_empty() { 0 } else { 2 * ((hi[r >> 6] >> (r & 63)) & 1) as u32 };
                    assert_eq!(got, want, "row {r}, M = {modulus}, parity = {parity}");
                }
            }
        }
    }
}

#[test]
fn eval_fold_declines_an_unsupported_modulus() {
    let index = InvertedIndex::new(64);
    let fold = LinearFold::<1> { modulus: 3, masks: [BasisString::zero(); 3], use_parity: false };
    assert!(!index.eval_fold(&fold, &mut Vec::new(), &mut Vec::new()));
}
