///
/// A/B timing: the monoprop-shaped engine against the SoA engine, on an
/// Ising-Trotter circuit shaped like the one the benchmark suite runs.
///
/// Ignored by default because it is a measurement, not an assertion. Run with:
///
/// ```text
/// cargo test --release -p propaq-pauli --test monoprop_engine_ab -- --ignored --nocapture
/// ```
///
/// The circuit is built here rather than loaded from the Python benchmark IR, so
/// this stays a pure Rust A/B. It has the same shape as `ising_trotter`: a ZZ
/// coupling on every lattice edge plus a transverse X field on every site, one
/// such layer per Trotter step.
///
/// Truncation uses a weight cutoff rather than a coefficient cutoff, on purpose.
/// A weight cutoff is structural, so it converts exactly and both engines keep
/// identical term sets, which is what makes the wall times comparable. A
/// coefficient cutoff would let the two engines carry different numbers of terms
/// and the timing would stop meaning anything.
///
use std::time::Instant;

use propaq_core::monomial::Monomial;
use propaq_core::operator::{EmitCutoff, Operator};
use propaq_core::partitioned::PartitionedOperator;
use propaq_core::soa::{kernels, SoaTermSum};
use propaq_core::truncators::ResolvedConfig;
use propaq_pauli::algebra::{planes_of, to_monomial, PauliAlgebra};
use propaq_pauli::string::{PauliBasis, PauliString};

/// 6x6 lattice, matching the benchmark's ising_trotter size.
const NX: usize = 6;
const NY: usize = 6;
const N_QUBITS: usize = NX * NY;
/// 72 interleaved bits, so two storage words.
const W: usize = 2;
/// The old two-plane form rounds 36 up to one 64-bit word per plane.
const STRIDE: usize = 1;
/// Weight cutoff, chosen to keep the run bounded at the larger step counts.
const MAX_WEIGHT: u32 = 6;

/// u8 addresses 255 positions, which covers this width's 128 bits.
type NewOp = Operator<f64, u8, W>;
type PartOp = PartitionedOperator<f64, u8, W>;

/// A ZZ coupling or an X field, as an (x_plane, z_plane) pair.
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

/// Nearest-neighbour edges of an NX by NY grid, in row-major site order.
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

/// One Trotter step: ZZ on every edge, then RX on every site.
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

/// The observable: a single Z on the last site, matching the benchmark.
fn observable() -> PauliString {
    pauli_from_masks(0, 1 << (N_QUBITS - 1))
}

struct Run {
    wall_s: f64,
    terms: usize,
    expectation: f64,
    key_bytes: usize,
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
        key_bytes: terms.sparse_key_bytes(),
    }
}

fn run_new(gates: &[Gate], fock: &[u64]) -> Run {
    let cutoff = EmitCutoff { max_weight: Some(MAX_WEIGHT), min_coeff: None };
    let mut op = NewOp::with_weight_cutoff(N_QUBITS, MAX_WEIGHT as usize);
    op.add(&to_monomial::<W>(&observable()), 1.0).unwrap();

    let t0 = Instant::now();
    for g in gates {
        let gen: Monomial<W> = to_monomial(&g.gen);
        op.apply_rotation::<PauliAlgebra>(&gen, &g.angle, &cutoff).unwrap();
    }
    let wall_s = t0.elapsed().as_secs_f64();
    Run {
        wall_s,
        terms: op.len(),
        expectation: op.expectation::<PauliAlgebra>(fock),
        key_bytes: op.key_bytes(),
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
    Run {
        wall_s,
        terms: op.len(),
        expectation: op.expectation::<PauliAlgebra>(fock),
        key_bytes: op.key_bytes(),
    }
}

#[test]
#[ignore = "measurement, not an assertion; run with --ignored --release"]
fn ab_ising_trotter_6x6() {
    let fock = vec![0u64];
    // The SoA kernels go parallel above 512 terms via rayon's global pool, and
    // the new engine is single-threaded. Set RAYON_NUM_THREADS=1 to compare
    // architectures rather than thread counts.
    let soa_threads = rayon::current_num_threads();
    let max_step: usize = std::env::var("PROPAQ_AB_MAX_STEP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(7);

    println!();
    println!("Ising-Trotter {NX}x{NY} ({N_QUBITS} qubits), weight cutoff {MAX_WEIGHT}");
    println!("soa engine threads = {soa_threads}, new engine threads = 1");
    println!(
        "{:>5}  {:>9}  {:>10}  {:>10}  {:>8}  {:>11}  {:>11}",
        "steps", "terms", "soa (s)", "new (s)", "speedup", "soa B/term", "new B/term"
    );

    for steps in [1usize, 3, 5, 7, 9].into_iter().filter(|&s| s <= max_step) {
        let gates = trotter_circuit(steps, 0.1);
        let soa = run_soa(&gates, &fock);
        let new = run_new(&gates, &fock);

        // A weight cutoff is structural, so the two engines must keep the same
        // terms. If they do not, the timing below is not comparable and the
        // divergence is the real result.
        assert_eq!(
            new.terms, soa.terms,
            "steps {steps}: term counts diverged under a structural cutoff \
             (new={}, soa={}), so the timings are not comparable",
            new.terms, soa.terms
        );
        assert!(
            (new.expectation - soa.expectation).abs() <= 1e-9 * soa.expectation.abs().max(1.0),
            "steps {steps}: expectation diverged: new={} soa={}",
            new.expectation,
            soa.expectation
        );

        println!(
            "{:>5}  {:>9}  {:>10.4}  {:>10.4}  {:>7.2}x  {:>11.1}  {:>11.1}",
            steps,
            soa.terms,
            soa.wall_s,
            new.wall_s,
            soa.wall_s / new.wall_s.max(1e-12),
            soa.key_bytes as f64 / soa.terms.max(1) as f64,
            new.key_bytes as f64 / new.terms.max(1) as f64,
        );
    }
    println!();
    println!("expectation agreement is asserted, so the wall times above compare");
    println!("identical computations rather than different truncations.");
}
