///
/// Partition scaling of the monoprop-shaped engine, against the SoA engine at
/// the same thread count.
///
/// Ignored by default. Run with:
///
/// ```text
/// cargo test --release -p propaq-pauli --test monoprop_partition_scaling \
///     -- --ignored --nocapture
/// ```
///
/// Both engines are given the same worker budget at each row: rayon's pool is
/// sized to `t`, the SoA kernels use it for their internal parallel passes, and
/// the partitioned engine is built with `t` partitions. The comparison is
/// therefore engine against engine at equal resources, which the earlier
/// single-partition A/B could not do.
///
/// Truncation is a weight cutoff, which converts exactly, so both engines keep
/// identical term sets and the wall times compare the same computation. Term
/// count and expectation agreement are asserted, not assumed.
///
use std::time::Instant;

use propaq_core::monomial::Monomial;
use propaq_core::operator::EmitCutoff;
use propaq_core::partitioned::PartitionedOperator;
use propaq_core::soa::{kernels, SoaTermSum};
use propaq_core::truncators::ResolvedConfig;
use propaq_pauli::algebra::{planes_of, to_monomial, PauliAlgebra};
use propaq_pauli::string::{PauliBasis, PauliString};

const NX: usize = 6;
const NY: usize = 6;
const N_QUBITS: usize = NX * NY;
const W: usize = 2;
const STRIDE: usize = 1;
const MAX_WEIGHT: u32 = 6;

type PartOp = PartitionedOperator<f64, u8, W>;

struct Gate {
    gen: PauliString,
    angle: f64,
}

fn pauli_from_masks(x: u64, z: u64) -> PauliString {
    let xb = propaq_core::bitset::Bitset::from_words(vec![x]);
    let zb = propaq_core::bitset::Bitset::from_words(vec![z]);
    let weight = (&xb | &zb).count_ones();
    PauliString { x: xb, z: zb, n_qubits: N_QUBITS, weight }
}

fn lattice_edges() -> Vec<(usize, usize)> {
    let site = |r: usize, c: usize| r * NX + c;
    let mut edges = Vec::new();
    for r in 0..NY {
        for c in 0..NX {
            if c + 1 < NX {
                edges.push((site(r, c), site(r, c + 1)));
            }
            if r + 1 < NY {
                edges.push((site(r, c), site(r + 1, c)));
            }
        }
    }
    edges
}

fn trotter_circuit(steps: usize, dt: f64) -> Vec<Gate> {
    let edges = lattice_edges();
    let mut gates = Vec::new();
    for _ in 0..steps {
        for &(a, b) in &edges {
            gates.push(Gate { gen: pauli_from_masks(0, (1 << a) | (1 << b)), angle: dt });
        }
        for q in 0..N_QUBITS {
            gates.push(Gate { gen: pauli_from_masks(1 << q, 0), angle: 0.5 * dt });
        }
    }
    gates
}

fn observable() -> PauliString {
    pauli_from_masks(0, 1 << (N_QUBITS - 1))
}

struct Run {
    wall_s: f64,
    terms: usize,
    expectation: f64,
    scan_s: f64,
    absorb_s: f64,
}

fn run_soa(gates: &[Gate], fock: &[u64]) -> Run {
    let cfg = ResolvedConfig { weight: Some(MAX_WEIGHT), ..Default::default() };
    let mut terms = SoaTermSum::<f64>::new(N_QUBITS, STRIDE);
    let (x, z) = planes_of(&observable(), STRIDE);
    terms.push([&x, &z], 1.0);

    let t0 = Instant::now();
    for g in gates {
        let (gx, gz) = planes_of(&g.gen, STRIDE);
        kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &g.angle, false);
        kernels::merge_and_truncate::<PauliBasis, f64>(&mut terms, Some(&cfg));
    }
    let wall_s = t0.elapsed().as_secs_f64();
    Run {
        wall_s,
        terms: terms.len(),
        expectation: kernels::expectation::<PauliBasis, f64>(&terms, fock),
        scan_s: 0.0,
        absorb_s: 0.0,
    }
}

fn run_partitioned(gates: &[Gate], fock: &[u64], n_partitions: usize) -> Run {
    let cutoff = EmitCutoff { max_weight: Some(MAX_WEIGHT), min_coeff: None };
    let mut op = PartOp::with_weight_cutoff(N_QUBITS, n_partitions, MAX_WEIGHT as usize);
    op.add(&to_monomial::<W>(&observable()), 1.0).unwrap();

    let t0 = Instant::now();
    for g in gates {
        let gen: Monomial<W> = to_monomial(&g.gen);
        op.apply_rotation::<PauliAlgebra>(&gen, &g.angle, &cutoff).unwrap();
    }
    let wall_s = t0.elapsed().as_secs_f64();
    let (scan_s, absorb_s) = op.phase_seconds();
    Run { wall_s, terms: op.len(), expectation: op.expectation::<PauliAlgebra>(fock), scan_s, absorb_s }
}

#[test]
#[ignore = "measurement, not an assertion; run with --ignored --release"]
fn partition_scaling_ising_trotter_6x6() {
    let fock = vec![0u64];
    let steps: usize = std::env::var("PROPAQ_AB_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);
    let gates = trotter_circuit(steps, 0.1);

    println!();
    println!("Ising-Trotter {NX}x{NY} ({N_QUBITS} qubits), {steps} steps, weight cutoff {MAX_WEIGHT}");
    println!("each row gives both engines the same worker budget");
    println!(
        "{:>8}  {:>9}  {:>10}  {:>10}  {:>8}  {:>9}  {:>9}  {:>8}  {:>8}",
        "threads", "terms", "soa (s)", "new (s)", "new/soa", "soa spdup", "new spdup", "scan (s)",
        "absorb(s)"
    );

    let mut soa_serial = 0.0f64;
    let mut new_serial = 0.0f64;

    for threads in [1usize, 2, 4, 8, 16, 32, 64] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
        // The SoA engine reads its worker count from the installed pool; the
        // partitioned engine also runs inside it, with one partition per worker.
        let (soa, new) = pool.install(|| {
            let soa = run_soa(&gates, &fock);
            let new = run_partitioned(&gates, &fock, threads);
            (soa, new)
        });

        assert_eq!(
            new.terms, soa.terms,
            "{threads} threads: term counts diverged under a structural cutoff \
             (new={}, soa={})",
            new.terms, soa.terms
        );
        assert!(
            (new.expectation - soa.expectation).abs() <= 1e-9 * soa.expectation.abs().max(1.0),
            "{threads} threads: expectation diverged: new={} soa={}",
            new.expectation,
            soa.expectation
        );

        if threads == 1 {
            soa_serial = soa.wall_s;
            new_serial = new.wall_s;
        }

        println!(
            "{:>8}  {:>9}  {:>10.4}  {:>10.4}  {:>7.2}x  {:>8.2}x  {:>8.2}x  {:>8.4}  {:>8.4}",
            threads,
            soa.terms,
            soa.wall_s,
            new.wall_s,
            soa.wall_s / new.wall_s.max(1e-12),
            soa_serial / soa.wall_s.max(1e-12),
            new_serial / new.wall_s.max(1e-12),
            new.scan_s,
            new.absorb_s,
        );
    }
    println!();
    println!("new/soa is the engine speedup at equal resources.");
    println!("the two rightmost columns are each engine's own parallel scaling");
    println!("against its own single-threaded time.");
}
