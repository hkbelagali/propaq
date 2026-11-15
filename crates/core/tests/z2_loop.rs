//! The generic run loop on a +-1 basis (BRANCHES == 2).
//!
//! The toy algebra: keys are bit sets, two keys anticommute when
//! parity(key & gen) is odd, and the product is `key ^ gen` with phase i.
//! The loop is checked against a per-term dictionary reference, for `f64`
//! and for `Complex64` coefficients, with noise and a coefficient cutoff.

use std::collections::HashMap;

use num_complex::Complex64;

use propaq_core::basis::{Basis, BasisKind};
use propaq_core::coeff::CoeffRepr;
use propaq_core::noise_resolver::ResolvedNoise;
use propaq_core::run::{run_at_width, RunGate, RunInputs};
use propaq_core::strings::BasisString;
use propaq_core::termsum::EmitCutoff;

struct Toy;

impl Basis<1> for Toy {
    const KIND: BasisKind = BasisKind::Custom;
    type GenContext = BasisString<1>;
    fn make_signed_gen_context(gen: &BasisString<1>, sign: f64) -> Self::GenContext {
        assert_eq!(sign, 1.0);
        *gen
    }
    fn generator(ctx: &Self::GenContext) -> &BasisString<1> {
        ctx
    }
    fn anticommutes(ctx: &Self::GenContext, m: &BasisString<1>) -> bool {
        m.parity_and(ctx)
    }
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<1> {
        ctx
    }
    fn product(ctx: &Self::GenContext, m: &BasisString<1>) -> (BasisString<1>, Complex64) {
        (*m ^ *ctx, Complex64::new(0.0, 1.0))
    }
    fn weight(m: &BasisString<1>, _n: usize) -> u32 {
        m.count() as u32
    }
    fn trace(_m: &BasisString<1>, _n: usize, _d: &[u64]) -> f64 {
        0.0
    }
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s >> 33
}

type Setup = (Vec<(Vec<u64>, f64)>, Vec<Vec<RunGate<f64>>>);

fn setup() -> Setup {
    let mut s = 4u64;
    let mut obs: HashMap<u64, f64> = HashMap::new();
    for _ in 0..5 {
        obs.insert(1 + lcg(&mut s) % 255, (lcg(&mut s) % 100) as f64 / 50.0 - 1.0);
    }
    let obs = obs.into_iter().map(|(k, c)| (vec![k], c)).collect();
    let layers = (0..3)
        .map(|_| {
            (0..4)
                .map(|i| RunGate { qiskit_gate_idx: Some(i), generator: vec![1 + lcg(&mut s) % 255], angle: 0.3 + 0.1 * (lcg(&mut s) % 6) as f64 })
                .collect()
        })
        .collect();
    (obs, layers)
}

fn reference(obs: &[(Vec<u64>, f64)], layers: &[Vec<RunGate<f64>>], table: &[f64], min_coeff: f64) -> HashMap<u64, f64> {
    let mut map: HashMap<u64, f64> = obs.iter().map(|(k, c)| (k[0], *c)).collect();
    for layer in layers {
        for (k, c) in map.iter_mut() {
            *c *= table[(k.count_ones() as usize).min(table.len() - 1)];
        }
        for g in layer {
            let gen = g.generator[0];
            let (sin, cos) = g.angle.sin_cos();
            let mut next: HashMap<u64, f64> = HashMap::new();
            for (&k, &c) in &map {
                if (k & gen).count_ones() % 2 == 0 {
                    *next.entry(k).or_default() += c;
                } else {
                    *next.entry(k).or_default() += c * cos;
                    // The product phase is +i, so the sine branch has factor -sin.
                    let child = -(c * sin);
                    if child.abs() >= min_coeff {
                        *next.entry(k ^ gen).or_default() += child;
                    }
                }
            }
            map = next;
        }
    }
    map
}

fn run<C: CoeffRepr>(obs: &[(Vec<u64>, f64)], layers: &[Vec<RunGate<C>>], partitions: usize, cutoff: &EmitCutoff, noise: &ResolvedNoise) -> Vec<(Vec<u64>, C)> {
    let obs: Vec<(Vec<u64>, C)> = obs.iter().map(|(w, c)| (w.clone(), C::from_real(*c))).collect();
    run_at_width::<Toy, C, 1, u8>(&RunInputs {
        observable: &obs,
        layers,
        n_units: 8,
        partitions,
        cutoff,
        noise: Some(noise),
        collect_counts: false,
        diag_state: None,
        want_terms: true,
        log_gates: false,
        progress: None,
    })
    .unwrap()
    .terms
    .unwrap()
}

#[test]
fn f64_and_complex_loops_match_the_reference() {
    let (obs, layers) = setup();
    let table = vec![1.0, 0.95, 0.9, 0.8];
    let noise = ResolvedNoise::WeightTable(table.clone());
    let cutoff = EmitCutoff { min_coeff: Some(1e-3), ..EmitCutoff::none() };
    let cl: Vec<Vec<RunGate<Complex64>>> = layers
        .iter()
        .map(|l| l.iter().map(|g| RunGate { qiskit_gate_idx: g.qiskit_gate_idx, generator: g.generator.clone(), angle: g.angle }).collect())
        .collect();
    for partitions in [1usize, 3] {
        let a = run::<f64>(&obs, &layers, partitions, &cutoff, &noise);
        let b = run::<Complex64>(&obs, &cl, partitions, &cutoff, &noise);
        let want = reference(&obs, &layers, &table, 1e-3);
        let am: HashMap<u64, f64> = a.into_iter().map(|(k, c)| (k[0], c)).collect();
        let bm: HashMap<u64, Complex64> = b.into_iter().map(|(k, c)| (k[0], c)).collect();
        // The cutoff drops tiny branches, so compare the large terms and the term set size loosely.
        for (k, v) in &want {
            if v.abs() > 1e-2 {
                assert!((am.get(k).copied().unwrap_or(0.0) - v).abs() < 5e-3, "f64 {k}");
                assert!((bm.get(k).copied().unwrap_or_default().re - v).abs() < 5e-3, "c64 {k}");
            }
        }
        for (k, c) in &am {
            assert!((bm[k].re - c).abs() < 1e-12 && bm[k].im.abs() < 1e-12);
        }
    }
}
