///
/// Whole-circuit differential: the monoprop-shaped engine (`Operator` plus
/// `PauliAlgebra`) against the SoA engine it is intended to replace.
///
/// The SoA engine is the oracle. Both are driven from the same seeded circuit
/// and compared on the full key-to-coefficient map and the expectation value.
///
/// Truncation is off in the correctness tests. The two engines truncate with
/// different semantics by design (the new one predicts a child's magnitude
/// before creating it, the old one drops terms after accumulation), so with a
/// cutoff active they are *expected* to diverge. That divergence is measured in
/// its own test rather than asserted away.
///
use std::collections::HashMap;

use propaq_core::coeff::CoeffRepr;

use propaq_core::monomial::Monomial;
use propaq_core::operator::{EmitCutoff, Operator};
use propaq_core::partitioned::PartitionedOperator;
use propaq_core::soa::{kernels, SoaBasis, SoaTermSum};
use propaq_pauli::algebra::{from_monomial, planes_of, to_monomial, PauliAlgebra};
use propaq_pauli::string::{PauliBasis, PauliString};

/// 40 qubits: 80 interleaved bits, so two storage words, and one word in the
/// old two-plane form. Wide enough that the two layouts disagree if the bit
/// convention is wrong, small enough to run undecimated with truncation off.
const N_QUBITS: usize = 40;
const W: usize = 2;
const STRIDE: usize = 1;

type NewOp = Operator<f64, u16, W>;

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Qubits drawn from two narrow windows straddling the 32-qubit boundary, so
/// generators actually anticommute with live terms and the circuit branches,
/// while still producing multiword monomials.
fn active_qubit(rng: &mut Rng) -> usize {
    let k = rng.below(12) as usize;
    if k < 6 {
        k
    } else {
        30 + (k - 6)
    }
}

/// A random Pauli over the active windows.
fn random_pauli(rng: &mut Rng, max_weight: usize) -> PauliString {
    let mut xw = vec![0u64; STRIDE];
    let mut zw = vec![0u64; STRIDE];
    for _ in 0..1 + rng.below(max_weight as u64) {
        let q = active_qubit(rng);
        match rng.below(3) {
            0 => xw[q / 64] |= 1u64 << (q % 64),
            1 => zw[q / 64] |= 1u64 << (q % 64),
            _ => {
                xw[q / 64] |= 1u64 << (q % 64);
                zw[q / 64] |= 1u64 << (q % 64);
            }
        }
    }
    let x = propaq_core::bitset::Bitset::from_words(xw);
    let z = propaq_core::bitset::Bitset::from_words(zw);
    let weight = (&x | &z).count_ones();
    PauliString { x, z, n_qubits: N_QUBITS, weight }
}

/// One gate of the shared circuit.
struct Gate {
    gen: PauliString,
    angle: f64,
}

/// Builds a circuit once, so both engines see byte-identical input.
fn build_circuit(seed: u64, n_gates: usize) -> (Vec<PauliString>, Vec<Gate>, Vec<u64>) {
    let mut rng = Rng(seed);
    let observable: Vec<PauliString> = (0..3).map(|_| random_pauli(&mut rng, 2)).collect();
    let gates: Vec<Gate> = (0..n_gates)
        .map(|_| Gate { gen: random_pauli(&mut rng, 3), angle: 0.1 + rng.unit() })
        .collect();
    let fock: Vec<u64> = (0..STRIDE).map(|_| rng.next_u64()).collect();
    (observable, gates, fock)
}

/// Runs the circuit on the SoA engine, returning term map and expectation.
fn run_soa(observable: &[PauliString], gates: &[Gate], fock: &[u64]) -> (HashMap<PauliString, f64>, f64) {
    let mut terms = SoaTermSum::<f64>::new(N_QUBITS, STRIDE);
    for p in observable {
        let (x, z) = planes_of(p, STRIDE);
        terms.push([&x, &z], 1.0);
    }
    kernels::merge::<PauliBasis, f64>(&mut terms);

    for g in gates {
        let (gx, gz) = planes_of(&g.gen, STRIDE);
        // Mirror the real propagator: a Clifford angle takes the in-place
        // conjugation path. On the generic path the cosine branch is not
        // exactly zero (cos(pi/2) is 6e-17), so it would leave ghost terms that
        // are an artifact of the driver rather than a difference between
        // engines.
        let clifford = <f64 as CoeffRepr>::is_clifford_param(&g.angle, 1e-9);
        kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &g.angle, clifford);
        kernels::merge::<PauliBasis, f64>(&mut terms);
    }

    let mut buf = vec![0u64; 2 * STRIDE];
    let map = (0..terms.len())
        .map(|i| {
            let key = PauliBasis::term_from_planes(terms.decode_row(i, &mut buf), N_QUBITS);
            (key, *terms.coeff(i))
        })
        .collect();
    let exp = kernels::expectation::<PauliBasis, f64>(&terms, fock);
    (map, exp)
}

/// Runs the same circuit on the new engine.
fn run_new(
    observable: &[PauliString],
    gates: &[Gate],
    fock: &[u64],
    cutoff: &EmitCutoff,
) -> (HashMap<PauliString, f64>, f64) {
    let mut op = NewOp::new(N_QUBITS);
    for p in observable {
        op.add(&to_monomial::<W>(p), 1.0).unwrap();
    }
    for g in gates {
        let gen: Monomial<W> = to_monomial(&g.gen);
        op.apply_rotation::<PauliAlgebra>(&gen, &g.angle, cutoff).unwrap();
    }
    let map = op.iter().map(|(k, c)| (from_monomial::<W>(&k, N_QUBITS), *c)).collect();
    let exp = op.expectation::<PauliAlgebra>(fock);
    (map, exp)
}

/// Compares two term maps, ignoring keys whose coefficient is exactly zero.
///
/// The append-only store keeps a row whose coefficient cancelled to zero, where
/// the SoA engine's merge also keeps it, but a term that never got created in
/// one engine and exists with coefficient zero in the other is not a real
/// disagreement.
fn assert_maps_agree(got: &HashMap<PauliString, f64>, want: &HashMap<PauliString, f64>, ctx: &str) {
    let nonzero = |m: &HashMap<PauliString, f64>| -> HashMap<PauliString, f64> {
        m.iter().filter(|(_, &v)| v != 0.0).map(|(k, &v)| (k.clone(), v)).collect()
    };
    let (g, w) = (nonzero(got), nonzero(want));
    assert_eq!(g.len(), w.len(), "{ctx}: live term count diverged (new={} soa={})", g.len(), w.len());
    for (key, &wv) in &w {
        let gv = g
            .get(key)
            .unwrap_or_else(|| panic!("{ctx}: key present in the SoA result is missing from the new engine"));
        assert!(
            (gv - wv).abs() <= 1e-9 * wv.abs().max(1.0),
            "{ctx}: coefficient diverged: new={gv} soa={wv}"
        );
    }
}

#[test]
fn engines_agree_on_a_single_rotation() {
    let (obs, gates, fock) = build_circuit(0x243F_6A88_85A3_08D3, 1);
    let (want, want_exp) = run_soa(&obs, &gates, &fock);
    let (got, got_exp) = run_new(&obs, &gates, &fock, &EmitCutoff::none());
    assert_maps_agree(&got, &want, "single rotation");
    assert!((got_exp - want_exp).abs() <= 1e-9 * want_exp.abs().max(1.0));
}

#[test]
fn engines_agree_on_randomized_circuits_with_truncation_off() {
    for (trial, seed) in [0x9E37_79B9_7F4A_7C15u64, 0x2545_F491_4F6C_DD1D, 0x853C_49E6_748F_EA9B]
        .into_iter()
        .enumerate()
    {
        let (obs, gates, fock) = build_circuit(seed, 24);
        let (want, want_exp) = run_soa(&obs, &gates, &fock);
        let (got, got_exp) = run_new(&obs, &gates, &fock, &EmitCutoff::none());

        let live = want.values().filter(|&&v| v != 0.0).count();
        assert!(live > 30, "trial {trial}: only {live} terms; the circuit did not branch enough");

        assert_maps_agree(&got, &want, &format!("trial {trial}"));
        assert!(
            (got_exp - want_exp).abs() <= 1e-9 * want_exp.abs().max(1.0),
            "trial {trial}: expectation diverged: new={got_exp} soa={want_exp}"
        );
    }
}

#[test]
fn engines_agree_when_a_child_repeatedly_lands_on_an_existing_term() {
    // A short generator set over few qubits forces heavy collision between
    // emitted children and live rows, which is the path where dedup on insert
    // and merge-after-the-fact could most easily disagree.
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let observable: Vec<PauliString> = (0..2).map(|_| random_pauli(&mut rng, 1)).collect();
    let gates: Vec<Gate> = (0..30)
        .map(|_| {
            let q = rng.below(3) as usize;
            let mut xw = vec![0u64; STRIDE];
            let mut zw = vec![0u64; STRIDE];
            if rng.below(2) == 0 {
                xw[0] |= 1u64 << q;
            } else {
                zw[0] |= 1u64 << q;
            }
            let x = propaq_core::bitset::Bitset::from_words(xw);
            let z = propaq_core::bitset::Bitset::from_words(zw);
            let weight = (&x | &z).count_ones();
            Gate {
                gen: PauliString { x, z, n_qubits: N_QUBITS, weight },
                angle: 0.1 + rng.unit(),
            }
        })
        .collect();
    let fock = vec![0b101u64];

    let (want, want_exp) = run_soa(&observable, &gates, &fock);
    let (got, got_exp) = run_new(&observable, &gates, &fock, &EmitCutoff::none());
    assert_maps_agree(&got, &want, "collision-heavy");
    assert!((got_exp - want_exp).abs() <= 1e-9 * want_exp.abs().max(1.0));
}

#[test]
fn a_weight_cutoff_converts_exactly_because_it_is_structural() {
    // Weight depends only on the key, so emit-time and post-hoc application
    // must agree exactly. This is the cutoff that ports cleanly.
    let (obs, gates, fock) = build_circuit(0x13198A2E_03707344, 20);
    let max_weight = 4u32;

    let mut terms = SoaTermSum::<f64>::new(N_QUBITS, STRIDE);
    for p in &obs {
        let (x, z) = planes_of(p, STRIDE);
        terms.push([&x, &z], 1.0);
    }
    let cfg = propaq_core::truncators::ResolvedConfig {
        weight: Some(max_weight),
        ..Default::default()
    };
    for g in &gates {
        let (gx, gz) = planes_of(&g.gen, STRIDE);
        kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &g.angle, false);
        kernels::merge_and_truncate::<PauliBasis, f64>(&mut terms, Some(&cfg));
    }
    let mut buf = vec![0u64; 2 * STRIDE];
    let want: HashMap<PauliString, f64> = (0..terms.len())
        .map(|i| (PauliBasis::term_from_planes(terms.decode_row(i, &mut buf), N_QUBITS), *terms.coeff(i)))
        .collect();

    let cutoff = EmitCutoff { max_weight: Some(max_weight), min_coeff: None };
    let (got, _) = run_new(&obs, &gates, &fock, &cutoff);

    assert!(!want.is_empty(), "the truncated circuit produced no terms");
    assert_maps_agree(&got, &want, "weight cutoff");
}

#[test]
fn a_coefficient_cutoff_diverges_and_the_divergence_is_reported() {
    // The engines truncate on magnitude with different semantics: the new one
    // predicts the child's magnitude from its parent and declines to create it,
    // the old one creates it and drops it after accumulation. They differ
    // exactly on cancellation. This test records the size of that difference
    // rather than asserting the two agree.
    let (obs, gates, fock) = build_circuit(0xA409_3822_299F_31D0, 24);
    let atol = 1e-6;

    let mut terms = SoaTermSum::<f64>::new(N_QUBITS, STRIDE);
    for p in &obs {
        let (x, z) = planes_of(p, STRIDE);
        terms.push([&x, &z], 1.0);
    }
    let cfg = propaq_core::truncators::ResolvedConfig {
        coefficient: Some(atol),
        ..Default::default()
    };
    for g in &gates {
        let (gx, gz) = planes_of(&g.gen, STRIDE);
        kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &g.angle, false);
        kernels::merge_and_truncate::<PauliBasis, f64>(&mut terms, Some(&cfg));
    }
    let soa_terms = terms.len();
    let soa_exp = kernels::expectation::<PauliBasis, f64>(&terms, &fock);

    let cutoff = EmitCutoff { max_weight: None, min_coeff: Some(atol) };
    let (got, got_exp) = run_new(&obs, &gates, &fock, &cutoff);
    let new_terms = got.values().filter(|&&v| v.abs() >= atol).count();

    println!("coefficient cutoff {atol:e}");
    println!("  soa terms = {soa_terms}, expectation = {soa_exp:.12e}");
    println!("  new terms = {new_terms}, expectation = {got_exp:.12e}");
    println!("  relative expectation difference = {:.3e}",
             (got_exp - soa_exp).abs() / soa_exp.abs().max(1e-30));

    // The predictive bound is conservative, so it should never drop a term the
    // post-accumulation bound keeps. Anything else means the bound is wrong,
    // not merely different.
    assert!(
        new_terms >= soa_terms,
        "the predictive cutoff dropped more than the post-hoc one ({new_terms} < {soa_terms}), \
         which means it is unsound rather than conservative"
    );
}

/// Runs a circuit on the partitioned engine, which defers single-qubit
/// Cliffords into a frame rather than applying them.
fn run_framed(
    observable: &[PauliString],
    gates: &[Gate],
    fock: &[u64],
    n_partitions: usize,
) -> (HashMap<PauliString, f64>, f64) {
    let mut op = PartitionedOperator::<f64, u16, W>::new(N_QUBITS, n_partitions);
    for p in observable {
        op.add(&to_monomial::<W>(p), 1.0).unwrap();
    }
    for g in gates {
        let gen: Monomial<W> = to_monomial(&g.gen);
        op.apply_rotation::<PauliAlgebra>(&gen, &g.angle, &EmitCutoff::none()).unwrap();
    }
    let map = op
        .iter()
        .map(|(k, sign, c)| (from_monomial::<W>(&k, N_QUBITS), sign * *c))
        .collect();
    let exp = op.expectation::<PauliAlgebra>(fock);
    (map, exp)
}

/// A circuit mixing single-qubit Clifford rotations with generic ones.
fn build_clifford_circuit(seed: u64, n_gates: usize) -> (Vec<PauliString>, Vec<Gate>, Vec<u64>) {
    let mut rng = Rng(seed);
    let observable: Vec<PauliString> = (0..3).map(|_| random_pauli(&mut rng, 2)).collect();
    let gates: Vec<Gate> = (0..n_gates)
        .map(|k| {
            if k % 2 == 0 {
                // Single-qubit Clifford: quarter turn about one qubit's X or Z.
                let q = active_qubit(&mut rng);
                let mut xw = vec![0u64; STRIDE];
                let mut zw = vec![0u64; STRIDE];
                if rng.below(2) == 0 {
                    xw[q / 64] |= 1u64 << (q % 64);
                } else {
                    zw[q / 64] |= 1u64 << (q % 64);
                }
                let x = propaq_core::bitset::Bitset::from_words(xw);
                let z = propaq_core::bitset::Bitset::from_words(zw);
                let weight = (&x | &z).count_ones();
                Gate {
                    gen: PauliString { x, z, n_qubits: N_QUBITS, weight },
                    angle: std::f64::consts::FRAC_PI_2,
                }
            } else {
                Gate { gen: random_pauli(&mut rng, 3), angle: 0.1 + rng.unit() }
            }
        })
        .collect();
    let fock: Vec<u64> = (0..STRIDE).map(|_| rng.next_u64()).collect();
    (observable, gates, fock)
}

#[test]
fn a_deferred_clifford_frame_matches_applying_cliffords_eagerly() {
    // The SoA engine applies every Clifford to every term as it goes. The
    // partitioned engine absorbs single-qubit Cliffords into a frame and never
    // touches a stored key. The two must agree exactly, which is the assertion
    // that catches a sign error or a composition-order error in the frame.
    for (trial, seed) in [0x243F_6A88_85A3_08D3u64, 0x13198A2E_03707344, 0xA409_3822_299F_31D0]
        .into_iter()
        .enumerate()
    {
        let (obs, gates, fock) = build_clifford_circuit(seed, 20);
        let (want, want_exp) = run_soa(&obs, &gates, &fock);

        for &parts in &[1usize, 4, 8] {
            let (got, got_exp) = run_framed(&obs, &gates, &fock, parts);
            assert_maps_agree(&got, &want, &format!("trial {trial}, {parts} partitions"));
            assert!(
                (got_exp - want_exp).abs() <= 1e-9 * want_exp.abs().max(1.0),
                "trial {trial}, {parts} partitions: expectation diverged: framed={got_exp} soa={want_exp}"
            );
        }
    }
}

#[test]
fn a_clifford_only_circuit_never_grows_the_store() {
    // Every gate is a single-qubit Clifford, so all of them land in the frame
    // and not one term is created or rewritten.
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let obs: Vec<PauliString> = (0..4).map(|_| random_pauli(&mut rng, 2)).collect();
    let mut op = PartitionedOperator::<f64, u16, W>::new(N_QUBITS, 4);
    for p in &obs {
        op.add(&to_monomial::<W>(p), 1.0).unwrap();
    }
    let before = op.len();
    assert!(op.frame_is_identity(), "the frame starts empty");

    for _ in 0..40 {
        let q = active_qubit(&mut rng);
        let mut xw = vec![0u64; STRIDE];
        let zw = vec![0u64; STRIDE];
        xw[q / 64] |= 1u64 << (q % 64);
        let x = propaq_core::bitset::Bitset::from_words(xw);
        let z = propaq_core::bitset::Bitset::from_words(zw);
        let weight = (&x | &z).count_ones();
        let gen = PauliString { x, z, n_qubits: N_QUBITS, weight };
        let added = op
            .apply_rotation::<PauliAlgebra>(
                &to_monomial::<W>(&gen),
                &std::f64::consts::FRAC_PI_2,
                &EmitCutoff::none(),
            )
            .unwrap();
        assert_eq!(added, 0, "a deferred Clifford must create no terms");
    }
    assert_eq!(op.len(), before, "a Clifford-only circuit must not grow the store");
    assert!(!op.frame_is_identity(), "the frame must have absorbed the gates");
}
