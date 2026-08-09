///
/// Accuracy regression check: propagates the same random circuit through the
/// same observable using SoaTermSum<f64> and SoaTermSum<f32>, and compares
/// the resulting coefficients and expectation value.
///
use propaq_core::bitset::Bitset;
use propaq_core::soa::{kernels, SoaBasis, SoaTermSum};
use propaq_pauli::string::{PauliBasis, PauliString};

fn make_pauli(x: u64, z: u64, n: usize) -> PauliString {
    let xb = Bitset::from_le_bytes(&x.to_le_bytes());
    let zb = Bitset::from_le_bytes(&z.to_le_bytes());
    let weight = (&xb | &zb).count_ones();
    PauliString { x: xb, z: zb, n_qubits: n, weight }
}

fn planes_of(term: &PauliString, n_qubits: usize, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    let mut gz = vec![0u64; stride];
    PauliBasis::term_into_planes(term, n_qubits, [&mut gx, &mut gz]);
    (gx, gz)
}

/// Tiny deterministic xorshift64* PRNG so the test is reproducible without
/// pulling in the `rand` crate as a new dependency.
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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// A random single- or two-qubit Pauli generator, mimicking the gate set a
/// real circuit would apply (as opposed to a fully dense random bitmask).
fn random_generator(rng: &mut Rng, n_qubits: usize) -> (u64, u64) {
    if rng.next_f64() < 0.7 {
        let q = rng.below(n_qubits as u64);
        match rng.below(3) {
            0 => (1u64 << q, 0),         // X
            1 => (1u64 << q, 1u64 << q), // Y
            _ => (0, 1u64 << q),         // Z
        }
    } else {
        let q1 = rng.below(n_qubits as u64);
        let mut q2 = rng.below(n_qubits as u64);
        while q2 == q1 {
            q2 = rng.below(n_qubits as u64);
        }
        if rng.below(2) == 0 {
            (1u64 << q1 | 1u64 << q2, 0) // XX
        } else {
            (0, 1u64 << q1 | 1u64 << q2) // ZZ
        }
    }
}

#[test]
fn f32_vs_f64_accuracy_under_random_propagation() {
    let n_qubits = 10usize;
    let n_gates = 150usize;
    let stride = PauliBasis::stride_words(n_qubits);

    // Seed observable: sum of single-qubit Z strings.
    let mut seed = SoaTermSum::<f64>::new(n_qubits, stride);
    for q in 0..n_qubits {
        let term = make_pauli(0, 1u64 << q, n_qubits);
        let (gx, gz) = planes_of(&term, n_qubits, stride);
        seed.push([&gx, &gz], 1.0);
    }

    let mut terms64 = seed.copy();
    let mut terms32: SoaTermSum<f32> = seed.map_coeffs(|c| *c as f32);

    let mut rng = Rng(0x9E3779B97F4A7C15);
    for _ in 0..n_gates {
        let (gx_bits, gz_bits) = random_generator(&mut rng, n_qubits);
        let angle = (rng.next_f64() * 2.0 - 1.0) * std::f64::consts::PI;

        let gen_term = make_pauli(gx_bits, gz_bits, n_qubits);
        let (gx, gz) = planes_of(&gen_term, n_qubits, stride);

        kernels::apply_rotation::<PauliBasis, f64>(&mut terms64, [&gx, &gz], &angle, false);
        kernels::merge::<PauliBasis, f64>(&mut terms64);

        kernels::apply_rotation::<PauliBasis, f32>(&mut terms32, [&gx, &gz], &angle, false);
        kernels::merge::<PauliBasis, f32>(&mut terms32);
    }

    assert_eq!(terms64.len(), terms32.len(), "term count diverged between f64 and f32 runs");
    let n = terms64.len();

    let norm64: f64 = (0..n).map(|i| terms64.coeff(i).powi(2)).sum();
    let norm32: f64 = (0..n).map(|i| (*terms32.coeff(i) as f64).powi(2)).sum();
    let norm_rel_err = (norm64 - norm32).abs() / norm64.max(1e-300);

    let fock = vec![0u64; stride];
    let exp64: f64 = (0..n)
        .map(|i| terms64.coeff(i) * PauliBasis::trace_sparse(terms64.row_positions(i), terms64.plane_span(), n_qubits, &fock))
        .sum();
    let exp32: f64 = (0..n)
        .map(|i| (*terms32.coeff(i) as f64)
            * PauliBasis::trace_sparse(terms32.row_positions(i), terms32.plane_span(), n_qubits, &fock))
        .sum();
    let exp_rel_err = (exp64 - exp32).abs() / exp64.abs().max(1e-12);

    let mut max_term_rel_err = 0.0f64;
    let mut sum_term_rel_err = 0.0f64;
    let mut worst: Vec<(f64, f64)> = Vec::new(); // (rel_err, |c64|)
    for i in 0..n {
        let c64 = *terms64.coeff(i);
        let c32 = *terms32.coeff(i) as f64;
        let rel = (c64 - c32).abs() / c64.abs().max(1e-9);
        max_term_rel_err = max_term_rel_err.max(rel);
        sum_term_rel_err += rel;
        if rel > 0.05 {
            worst.push((rel, c64.abs()));
        }
    }
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("terms with rel_err > 0.05: {} out of {n}", worst.len());
    for (rel, mag) in worst.iter().take(10) {
        println!("  rel_err = {rel:.3e}  |c64| = {mag:.3e}");
    }

    println!("n_terms after {n_gates} gates on {n_qubits} qubits = {n}");
    println!("norm_squared:  f64 = {norm64:.6e}  f32 = {norm32:.6e}  rel_err = {norm_rel_err:.3e}");
    println!("expectation:   f64 = {exp64:.6e}   f32 = {exp32:.6e}   rel_err = {exp_rel_err:.3e}");
    println!("per-term coeff rel_err: max = {max_term_rel_err:.3e}  mean = {:.3e}", sum_term_rel_err / n as f64);

    assert!(norm_rel_err < 1e-3, "norm_squared relative error too large: {norm_rel_err:.3e}");
    assert!(exp_rel_err < 1e-2, "expectation value relative error too large: {exp_rel_err:.3e}");
}
