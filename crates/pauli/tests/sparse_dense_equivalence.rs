///
/// Equivalence checks for the sparse Pauli backend against the retained
/// word-plane algebra it replaced.
///
/// Two layers:
///
///   * the `SoaBasis` sparse methods against their word-plane counterparts, on
///     randomized single and multiword rows;
///   * whole randomized circuits run through the real kernels under both
///     kernel layouts, where the `Dense` layout decodes every row and calls the
///     word-plane methods, i.e. acts as a retained dense oracle for the sparse
///     kernels.
///
/// The layout is process-global, so every test here serializes on `LAYOUT_LOCK`;
/// this is its own test binary, so nothing else can observe the flip.
///
use std::collections::HashMap;
use std::sync::Mutex;

use propaq_core::soa::sparse::encode_planes_into;
use propaq_core::soa::{kernels, set_kernel_layout, KernelLayout, Position, SoaBasis, SoaTermSum};
use propaq_pauli::string::PauliBasis;

static LAYOUT_LOCK: Mutex<()> = Mutex::new(());

const N_QUBITS: usize = 96;

/// Tiny deterministic xorshift64 PRNG, so the tests are reproducible without a
/// new dependency.
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

fn stride() -> usize {
    PauliBasis::stride_words(N_QUBITS)
}

/// Support is drawn from two narrow windows that straddle the stride-word
/// boundary: wide enough to produce genuinely multiword keys, narrow enough
/// that generators actually anticommute and the circuit branches.
fn active_qubit(rng: &mut Rng) -> usize {
    let k = rng.below(16) as usize;
    if k < 8 { k } else { 64 + (k - 8) }
}

/// A random term over the active windows, touching both planes.
fn random_planes(rng: &mut Rng, max_weight: usize) -> (Vec<u64>, Vec<u64>) {
    let s = stride();
    let mut x = vec![0u64; s];
    let mut z = vec![0u64; s];
    let weight = 1 + rng.below(max_weight as u64) as usize;
    for _ in 0..weight {
        let q = active_qubit(rng);
        match rng.below(3) {
            0 => x[q / 64] |= 1u64 << (q % 64),
            1 => z[q / 64] |= 1u64 << (q % 64),
            _ => {
                x[q / 64] |= 1u64 << (q % 64);
                z[q / 64] |= 1u64 << (q % 64);
            }
        }
    }
    (x, z)
}

/// A random term spread over the whole register, for the basis-method
/// differential checks (which do not need the circuit to branch).
fn random_wide_planes(rng: &mut Rng, max_weight: usize) -> (Vec<u64>, Vec<u64>) {
    let s = stride();
    let mut x = vec![0u64; s];
    let mut z = vec![0u64; s];
    let weight = 1 + rng.below(max_weight as u64) as usize;
    for _ in 0..weight {
        let q = rng.below(N_QUBITS as u64) as usize;
        match rng.below(3) {
            0 => x[q / 64] |= 1u64 << (q % 64),
            1 => z[q / 64] |= 1u64 << (q % 64),
            _ => {
                x[q / 64] |= 1u64 << (q % 64);
                z[q / 64] |= 1u64 << (q % 64);
            }
        }
    }
    (x, z)
}

fn sparse_row(planes: [&[u64]; 2]) -> Vec<Position> {
    let mut row = Vec::new();
    encode_planes_into(planes, stride() * 64, &mut row);
    row
}

#[test]
fn sparse_basis_methods_match_the_word_plane_methods() {
    let mut rng = Rng(0x243F6A8885A308D3);
    let plane_span = stride() * 64;
    let fock: Vec<u64> = (0..stride()).map(|_| rng.next_u64()).collect();

    for _ in 0..2000 {
        let (tx, tz) = random_wide_planes(&mut rng, 6);
        let (gx, gz) = random_wide_planes(&mut rng, 4);
        let term = [&tx[..], &tz[..]];
        let gen = [&gx[..], &gz[..]];
        let term_row = sparse_row(term);
        let gen_row = sparse_row(gen);

        assert_eq!(
            PauliBasis::weight_sparse(&term_row, plane_span, N_QUBITS),
            PauliBasis::weight(term, N_QUBITS),
            "weight diverged"
        );
        assert_eq!(
            PauliBasis::trace_sparse(&term_row, plane_span, N_QUBITS, &fock),
            PauliBasis::trace(term, N_QUBITS, &fock),
            "trace diverged"
        );
        assert_eq!(
            PauliBasis::commutes_sparse(&term_row, &gen_row, plane_span),
            PauliBasis::commutes(term, gen),
            "commutation diverged"
        );

        let mut got = Vec::new();
        let got_phase = PauliBasis::product_sparse(&term_row, &gen_row, plane_span, &mut got);
        let mut wx = vec![0u64; stride()];
        let mut wz = vec![0u64; stride()];
        let want_phase = PauliBasis::product(term, gen, [&mut wx, &mut wz]);
        assert_eq!(got_phase, want_phase, "product phase diverged");
        assert_eq!(got, sparse_row([&wx, &wz]), "product key diverged");
    }
}

#[test]
fn a_diagonal_row_traces_the_same_as_its_word_planes() {
    // `random_planes` almost never produces a Z-only term, so the trace check
    // above mostly exercises the early return. Cover the parity branch too.
    let mut rng = Rng(0x13198A2E03707344);
    let plane_span = stride() * 64;
    let fock: Vec<u64> = (0..stride()).map(|_| rng.next_u64()).collect();
    for _ in 0..500 {
        let s = stride();
        let x = vec![0u64; s];
        let mut z = vec![0u64; s];
        for _ in 0..1 + rng.below(8) {
            let q = rng.below(N_QUBITS as u64) as usize;
            z[q / 64] |= 1u64 << (q % 64);
        }
        let term = [&x[..], &z[..]];
        let row = sparse_row(term);
        assert_eq!(
            PauliBasis::trace_sparse(&row, plane_span, N_QUBITS, &fock),
            PauliBasis::trace(term, N_QUBITS, &fock),
        );
    }
}

/// Every live term's key (as decoded word planes) mapped to its coefficient.
fn term_values(terms: &SoaTermSum<f64>) -> HashMap<(Vec<u64>, Vec<u64>), f64> {
    let mut buf = vec![0u64; 2 * terms.stride];
    (0..terms.len())
        .map(|i| {
            let planes = terms.decode_row(i, &mut buf);
            ((planes[0].to_vec(), planes[1].to_vec()), *terms.coeff(i))
        })
        .collect()
}

/// Runs one fixed pseudo-random circuit under the currently selected kernel
/// layout, returning the final term set and its expectation value.
fn run_circuit(seed: u64) -> (HashMap<(Vec<u64>, Vec<u64>), f64>, f64, usize) {
    let s = stride();
    let mut rng = Rng(seed);
    let mut terms = SoaTermSum::<f64>::new(N_QUBITS, s);

    // Seed observable: a couple of Z strings, deliberately pushed twice so the
    // very first merge has a duplicate to fold.
    for _ in 0..3 {
        let (x, z) = random_planes(&mut rng, 3);
        terms.push([&x, &z], 1.0);
        terms.push([&x, &z], 0.5);
    }

    let fock: Vec<u64> = (0..s).map(|_| rng.next_u64()).collect();

    for step in 0..60u32 {
        let (gx, gz) = if rng.unit() < 0.5 {
            // Single-qubit generator: drives the `local_word` fast paths.
            let q = active_qubit(&mut rng);
            let mut x = vec![0u64; s];
            let mut z = vec![0u64; s];
            match rng.below(3) {
                0 => x[q / 64] |= 1u64 << (q % 64),
                1 => z[q / 64] |= 1u64 << (q % 64),
                _ => {
                    x[q / 64] |= 1u64 << (q % 64);
                    z[q / 64] |= 1u64 << (q % 64);
                }
            }
            (x, z)
        } else {
            random_planes(&mut rng, 4)
        };

        match step % 4 {
            // Non-Clifford rotation: appends a branch per anticommuting row.
            0 | 1 => {
                let angle = 0.1 + rng.unit();
                kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &angle, false);
            }
            // Clifford rotation applied in place.
            2 => {
                let angle = std::f64::consts::FRAC_PI_2;
                kernels::apply_rotation::<PauliBasis, f64>(&mut terms, [&gx, &gz], &angle, true);
            }
            // Fused Clifford conjugation over one stride-word.
            _ => {
                let q = active_qubit(&mut rng);
                let (word, bit) = (q / 64, (q % 64) as u32);
                let gen_word = [1u64 << bit, 0u64];
                let rotations = vec![
                    (gen_word, std::f64::consts::FRAC_PI_2),
                    ([0u64, 1u64 << bit], std::f64::consts::FRAC_PI_2),
                ];
                if let Some(op) = kernels::build_fused_clifford::<PauliBasis, f64>(
                    word,
                    [bit, bit],
                    1,
                    &rotations,
                    1e-9,
                ) {
                    kernels::apply_clifford_op::<PauliBasis, f64>(&mut terms, &op);
                }
            }
        }

        kernels::merge::<PauliBasis, f64>(&mut terms);

        // Weight truncation, to force compaction holes partway through.
        if step % 11 == 10 {
            let cfg = propaq_core::truncators::ResolvedConfig {
                weight: Some(8),
                ..Default::default()
            };
            kernels::truncate::<PauliBasis, f64>(&mut terms, &cfg);
        }
    }

    kernels::merge::<PauliBasis, f64>(&mut terms);
    let expectation = kernels::expectation::<PauliBasis, f64>(&terms, &fock);
    (term_values(&terms), expectation, terms.len())
}

#[test]
fn sparse_kernels_match_the_dense_oracle_on_randomized_circuits() {
    let _guard = LAYOUT_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for seed in [0x9E3779B97F4A7C15u64, 0x2545F4914F6CDD1D, 0x853C49E6748FEA9B] {
        set_kernel_layout(KernelLayout::Dense);
        let (want, want_exp, want_len) = run_circuit(seed);
        set_kernel_layout(KernelLayout::Sparse);
        let (got, got_exp, got_len) = run_circuit(seed);
        set_kernel_layout(KernelLayout::Sparse);

        assert!(want_len > 20, "seed {seed:#x}: only {want_len} terms; circuit did not branch enough");
        assert_eq!(got_len, want_len, "seed {seed:#x}: final term count diverged");
        assert_eq!(got.len(), want.len(), "seed {seed:#x}: live key set size diverged");
        for (key, &wanted) in &want {
            let have = got
                .get(key)
                .unwrap_or_else(|| panic!("seed {seed:#x}: key missing from the sparse run"));
            assert!(
                (have - wanted).abs() <= 1e-12 * wanted.abs().max(1.0),
                "seed {seed:#x}: coefficient diverged: sparse={have} dense={wanted}"
            );
        }
        assert!(
            (got_exp - want_exp).abs() <= 1e-10 * want_exp.abs().max(1.0),
            "seed {seed:#x}: expectation diverged: sparse={got_exp} dense={want_exp}"
        );
    }
}

/// The point of the rewrite: on a wide register, a low-weight key costs its own
/// set bits rather than a full-width row in each plane.
#[test]
fn key_storage_stays_proportional_to_set_bits() {
    let wide_qubits = 4096;
    let s = PauliBasis::stride_words(wide_qubits);
    let mut terms = SoaTermSum::<f64>::new(wide_qubits, s);
    let mut x = vec![0u64; s];
    let z = vec![0u64; s];
    for i in 0..4096usize {
        x[i % s] = 1;
        terms.push([&x, &z], 1.0);
        x[i % s] = 0;
    }
    let dense_bytes = 2 * s * terms.len() * std::mem::size_of::<u64>();
    assert!(
        terms.sparse_key_bytes() * 8 < dense_bytes,
        "sparse keys ({} bytes) did not beat the dense equivalent ({dense_bytes} bytes)",
        terms.sparse_key_bytes()
    );
}
