///
/// What the deferred Clifford frame is worth, as a function of how Clifford-rich
/// the circuit is.
///
/// Ignored by default. Run with:
///
/// ```text
/// cargo test --release -p propaq-pauli --test monoprop_clifford_ab \
///     -- --ignored --nocapture
/// ```
///
/// The circuit is Ising-Trotter with a tunable number of single-qubit Clifford
/// rotations added per step, so the sweep runs from the benchmark's own regime
/// (no Cliffords at all, where the frame must be free) up to Clifford-dominated.
///
/// The two arms are the same engine with `set_defer_cliffords` on and off, not
/// two different engines. With it off a Clifford rotation branches like any
/// other gate, which is what this engine would do without the frame; an eager
/// in-place key rewrite is not the comparison, because that would invalidate the
/// hash index, the partition assignment, and every inverted-index column, and is
/// deliberately not implemented here.
///
/// Term counts are reported rather than asserted equal. They legitimately differ:
/// the branching path leaves a source row with a `cos(pi/2)` coefficient of about
/// 6e-17, and an append-only store never reclaims it. Expectation values are
/// asserted equal, since those near-zero rows must not move the answer.
///
use std::time::Instant;

use propaq_core::monomial::Monomial;
use propaq_core::operator::EmitCutoff;
use propaq_core::partitioned::PartitionedOperator;
use propaq_pauli::algebra::{to_monomial, PauliAlgebra};
use propaq_pauli::string::PauliString;

const NX: usize = 6;
const NY: usize = 6;
const N_QUBITS: usize = NX * NY;
const W: usize = 2;
const MAX_WEIGHT: u32 = 6;

type Op = PartitionedOperator<f64, u8, W>;

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

/// One Trotter step, plus `clifford_layers` layers of single-qubit Cliffords.
///
/// The Cliffords are quarter turns about Z, so they are exactly the case a
/// per-qubit frame represents.
fn circuit(steps: usize, dt: f64, clifford_layers: usize) -> Vec<Gate> {
    let edges = lattice_edges();
    let mut gates = Vec::new();
    for _ in 0..steps {
        for &(a, b) in &edges {
            gates.push(Gate { gen: pauli_from_masks(0, (1 << a) | (1 << b)), angle: dt });
        }
        for q in 0..N_QUBITS {
            gates.push(Gate { gen: pauli_from_masks(1 << q, 0), angle: 0.5 * dt });
        }
        for _ in 0..clifford_layers {
            for q in 0..N_QUBITS {
                gates.push(Gate {
                    gen: pauli_from_masks(0, 1 << q),
                    angle: std::f64::consts::FRAC_PI_2,
                });
            }
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
}

fn run(gates: &[Gate], fock: &[u64], partitions: usize, defer: bool) -> Run {
    let cutoff = EmitCutoff { max_weight: Some(MAX_WEIGHT), min_coeff: None };
    let mut op = Op::with_weight_cutoff(N_QUBITS, partitions, MAX_WEIGHT as usize);
    op.set_defer_cliffords(defer);
    op.add(&to_monomial::<W>(&observable()), 1.0).unwrap();

    let t0 = Instant::now();
    for g in gates {
        let gen: Monomial<W> = to_monomial(&g.gen);
        op.apply_rotation::<PauliAlgebra>(&gen, &g.angle, &cutoff).unwrap();
    }
    let wall_s = t0.elapsed().as_secs_f64();
    Run { wall_s, terms: op.len(), expectation: op.expectation::<PauliAlgebra>(fock) }
}

#[test]
#[ignore = "measurement, not an assertion; run with --ignored --release"]
fn clifford_frame_ab() {
    let fock = vec![0u64];
    let steps: usize = std::env::var("PROPAQ_AB_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    let partitions: usize = std::env::var("PROPAQ_AB_PARTITIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(rayon::current_num_threads());

    println!();
    println!("Ising-Trotter {NX}x{NY}, {steps} steps, weight cutoff {MAX_WEIGHT}, {partitions} partitions");
    println!("clifford layers are extra single-qubit quarter turns per Trotter step");
    println!(
        "{:>8}  {:>7}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "cliff/st", "gates", "off (s)", "frame (s)", "speedup", "off terms", "frame terms"
    );

    for clifford_layers in [0usize, 1, 2, 4, 8] {
        let gates = circuit(steps, 0.1, clifford_layers);
        let off = run(&gates, &fock, partitions, false);
        let on = run(&gates, &fock, partitions, true);

        assert!(
            (on.expectation - off.expectation).abs() <= 1e-8 * off.expectation.abs().max(1.0),
            "{clifford_layers} clifford layers: expectation diverged: frame={} off={}",
            on.expectation,
            off.expectation
        );

        println!(
            "{:>8}  {:>7}  {:>10.4}  {:>10.4}  {:>7.2}x  {:>10}  {:>10}",
            clifford_layers,
            gates.len(),
            off.wall_s,
            on.wall_s,
            off.wall_s / on.wall_s.max(1e-12),
            off.terms,
            on.terms,
        );
    }
    println!();
    println!("expectation values are asserted equal across both arms.");
    println!("term counts differ because the branching path leaves cos(pi/2) rows");
    println!("of about 6e-17 that an append-only store never reclaims.");
}
