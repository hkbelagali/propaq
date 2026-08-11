use super::*;

fn mon(bits: u64, n_modes: usize) -> MajoranaMonomial {
    let modes = Bitset::from_le_bytes(&bits.to_le_bytes());
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
    MajoranaMonomial {
        modes,
        n_modes,
        is_number_preserving: true,
        weight,
        p,
    }
}

fn mon_bits(bits: Vec<u64>, n_modes: usize) -> MajoranaMonomial {
    let modes = Bitset::from_words(bits);
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
    MajoranaMonomial {
        modes,
        n_modes,
        is_number_preserving: true,
        weight,
        p,
    }
}

fn fock(bits: u64) -> Bitset {
    Bitset::from_le_bytes(&bits.to_le_bytes())
}

#[test]
fn hermiticity_exp_all_residues() {
    for (len, expected) in [
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 0),
        (5, 0),
        (6, 1),
        (7, 1),
        (8, 0),
    ] {
        assert_eq!(hermiticity_exp(len), expected, "hermiticity_exp({len})");
    }
}

#[test]
fn parity_disjoint_no_inversions() {
    let a = Bitset::from_le_bytes(&[0b0011]);
    let b = Bitset::from_le_bytes(&[0b1100]);
    assert!(!resorting_parity(a.as_words(), b.as_words()));
}

#[test]
fn parity_single_inversion() {
    let a = Bitset::from_le_bytes(&[0b0010]);
    let b = Bitset::from_le_bytes(&[0b0001]);
    assert!(resorting_parity(a.as_words(), b.as_words()));
}

#[test]
fn parity_two_inversions_even() {
    let a = Bitset::from_le_bytes(&[0b1100]);
    let b = Bitset::from_le_bytes(&[0b0011]);
    assert!(!resorting_parity(a.as_words(), b.as_words()));
}

#[test]
fn parity_empty_b_is_false() {
    let a = Bitset::from_le_bytes(&[0xFF]);
    let b = Bitset::zero();
    assert!(!resorting_parity(a.as_words(), b.as_words()));
}

#[test]
fn weight_identity() {
    assert_eq!(mon(0, 8).weight, 0);
}

#[test]
fn weight_single_gamma() {
    assert_eq!(mon(0b01, 8).weight, 1);
}

#[test]
fn weight_number_operator() {
    assert_eq!(mon(0b11, 8).weight, 1);
}

#[test]
fn weight_four_x_modes() {
    assert_eq!(mon(0b0101_0101, 8).weight, 4);
}

#[test]
fn weight_large_n_modes() {
    assert_eq!(mon(0b01, 128).weight, 1);
}

#[test]
fn weight_multi_word_mode() {
    let m = mon_bits(vec![0u64, 1u64], 128);
    assert_eq!(m.weight, 33);
}

#[test]
fn trace_identity_any_fock() {
    let m = mon(0, 8);
    assert_eq!(m.trace_fock_state_impl(&fock(0)), 1.0);
    assert_eq!(m.trace_fock_state_impl(&fock(0b1111)), 1.0);
}

#[test]
fn trace_unpaired_mode_is_zero() {
    let m = mon(0b01, 8);
    assert_eq!(m.trace_fock_state_impl(&fock(0)), 0.0);
    assert_eq!(m.trace_fock_state_impl(&fock(1)), 0.0);
}

#[test]
fn trace_site0_empty_fock() {
    assert_eq!(mon(0b11, 8).trace_fock_state_impl(&fock(0)), -1.0);
}

#[test]
fn trace_site0_occupied_fock() {
    assert_eq!(mon(0b11, 8).trace_fock_state_impl(&fock(1)), 1.0);
}

#[test]
fn trace_two_sites_all_combinations() {
    let m = mon(0b1111, 8);
    assert_eq!(m.trace_fock_state_impl(&fock(0b00)), -1.0);
    assert_eq!(m.trace_fock_state_impl(&fock(0b01)), 1.0);
    assert_eq!(m.trace_fock_state_impl(&fock(0b10)), 1.0);
    assert_eq!(m.trace_fock_state_impl(&fock(0b11)), -1.0);
}

fn assert_weight_and_p_correct(result: &MajoranaMonomial) {
    let expected_weight = MajoranaMonomial::compute_weight_for(&result.modes, result.n_modes);
    assert_eq!(
        result.weight, expected_weight,
        "weight mismatch for modes={:?}",
        result.modes
    );
    let (_, expected_p) = MajoranaMonomial::weight_and_p_for(&result.modes, result.n_modes);
    assert_eq!(
        result.p, expected_p,
        "p drifted for modes={:?}",
        result.modes
    );
}

#[test]
fn matmul_identity_on_left() {
    let identity = mon(0, 8);
    let m = mon(0b0011, 8);
    let (phase, result) = identity.matmul_internal(&m);
    assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert_eq!(result.modes, m.modes);
    assert_weight_and_p_correct(&result);
}

#[test]
fn matmul_identity_on_right() {
    let m = mon(0b0011, 8);
    let identity = mon(0, 8);
    let (phase, result) = m.matmul_internal(&identity);
    assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert_eq!(result.modes, m.modes);
    assert_weight_and_p_correct(&result);
}

#[test]
fn matmul_self_is_identity() {
    let m = mon(0b0111, 8);
    let (phase, result) = m.matmul_internal(&m);
    assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert!(result.modes.is_zero());
    assert_weight_and_p_correct(&result);
}

#[test]
fn matmul_disjoint_phase_is_minus_one() {
    let a = mon(0b0011, 8);
    let b = mon(0b1100, 8);
    let (phase, result) = a.matmul_internal(&b);
    assert!((phase - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    assert_eq!(result.modes.count_ones(), 4);
    assert_weight_and_p_correct(&result);
}

#[test]
fn commutes_with_itself() {
    let m = mon(0b0011, 8);
    assert!(m.commutes_with_impl(&m));
}

#[test]
fn commutes_disjoint_even_lengths() {
    let a = mon(0b0011, 8);
    let b = mon(0b1100, 8);
    assert!(a.commutes_with_impl(&b));
}

#[test]
fn anticommutes_single_overlap_even_lengths() {
    let a = mon(0b0011, 8);
    let b = mon(0b0110, 8);
    assert!(!a.commutes_with_impl(&b));
}

#[test]
fn commutes_single_modes_disjoint() {
    let a = mon(0b0001, 8);
    let b = mon(0b0010, 8);
    assert!(!a.commutes_with_impl(&b));
}

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn random_bitset(rng: &mut Rng, n_modes: usize) -> Bitset {
    let n_words = n_modes.div_ceil(64);
    let mut words: Vec<u64> = (0..n_words).map(|_| rng.next_u64()).collect();
    let rem = n_modes % 64;
    if rem != 0 {
        let mask = (1u64 << rem) - 1;
        *words.last_mut().unwrap() &= mask;
    }
    Bitset::from_words(words)
}

fn random_mon(rng: &mut Rng, n_modes: usize) -> MajoranaMonomial {
    let modes = random_bitset(rng, n_modes);
    let (weight, p) = MajoranaMonomial::weight_and_p_for(&modes, n_modes);
    MajoranaMonomial {
        modes,
        n_modes,
        is_number_preserving: true,
        weight,
        p,
    }
}

#[test]
fn weight_matches_reference_exhaustive_small() {

    for n_qubits in 1usize..=6 {
        let n_modes = 2 * n_qubits;
        let space = 1u64 << n_modes;
        let stride = (space / 37).max(1);
        for a_bits in 0..space {
            let a = mon(a_bits, n_modes);
            let mut b_bits = 0u64;
            while b_bits < space {
                let b = mon(b_bits, n_modes);
                let (_, result) = a.matmul_internal(&b);
                assert_weight_and_p_correct(&result);
                b_bits += stride;
            }
        }
    }
}

#[test]
fn weight_matches_reference_randomized_multiword() {
    let mut rng = Rng(0xC0FFEE_D15EA5E5);
    for &n_qubits in &[30usize, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200] {
        let n_modes = 2 * n_qubits;
        for _ in 0..300 {
            let a = random_mon(&mut rng, n_modes);
            let b = random_mon(&mut rng, n_modes);
            let (_, result) = a.matmul_internal(&b);
            assert_weight_and_p_correct(&result);
        }
    }
}

#[test]
fn weight_and_p_no_drift_over_chained_updates() {
    // Simulates a term being multiplied by 200 successive gate
    // generators in sequence, checking after every step that neither
    // the incrementally-tracked weight nor the cached `p` has drifted
    // from a full from-scratch recomputation.
    let mut rng = Rng(0x1234_5678_9ABC_DEF0);
    for &n_qubits in &[8usize, 33, 65, 128] {
        let n_modes = 2 * n_qubits;
        let mut term = random_mon(&mut rng, n_modes);
        for _ in 0..200 {
            let generator = random_mon(&mut rng, n_modes);
            let (_, next) = generator.matmul_internal(&term);
            assert_weight_and_p_correct(&next);
            term = next;
        }
    }
}

// Section: `MajoranaBasis` (columnar) vs `MajoranaMonomial` (owned) cross-checks, the seam most
// at risk since `weight`/`product` depend on the cached `p` plane tracking `modes`.

fn planes_of(m: &MajoranaMonomial, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut g0 = vec![0u64; stride];
    let mut g1 = vec![0u64; stride];
    MajoranaBasis::term_into_planes(m, m.n_modes, [&mut g0, &mut g1]);
    (g0, g1)
}

fn assert_majorana_basis_matches(a: &MajoranaMonomial, b: &MajoranaMonomial, stride: usize) {
    let (a0, a1) = planes_of(a, stride);
    let (b0, b1) = planes_of(b, stride);
    let a_planes = [a0.as_slice(), a1.as_slice()];
    let b_planes = [b0.as_slice(), b1.as_slice()];
    let ctx = || format!("a.modes={a0:?} b.modes={b0:?}");

    assert_eq!(
        MajoranaBasis::commutes(a_planes, b_planes),
        a.commutes_with_impl(b),
        "commutes mismatch for {}",
        ctx(),
    );
    assert_eq!(
        MajoranaBasis::weight(a_planes, a.n_modes),
        a.weight,
        "weight mismatch for {}",
        ctx()
    );

    // gen=a, term=b => a @ b, matching `a.matmul_internal(b)`.
    let (expected_phase, expected_result) = a.matmul_internal(b);
    let mut out0 = vec![0u64; stride];
    let mut out1 = vec![0u64; stride];
    let phase = MajoranaBasis::product(b_planes, a_planes, [&mut out0, &mut out1]);
    assert!(
        (phase - expected_phase).norm() < 1e-10,
        "phase mismatch for {}",
        ctx()
    );
    let result = MajoranaBasis::term_from_planes([&out0, &out1], a.n_modes);
    assert_eq!(
        result.modes,
        expected_result.modes,
        "product modes mismatch for {}",
        ctx()
    );
    assert_eq!(
        result.p,
        expected_result.p,
        "product p mismatch for {}",
        ctx()
    );
    assert_eq!(
        result.weight,
        expected_result.weight,
        "product weight mismatch for {}",
        ctx()
    );

    for fock_bits in 0u64..16 {
        let fock_words = [fock_bits];
        assert_eq!(
            MajoranaBasis::trace(a_planes, a.n_modes, &fock_words),
            a.trace_fock_state_impl(&fock(fock_bits)),
            "trace mismatch for {} fock={fock_bits}",
            ctx(),
        );
    }

    assert_eq!(
        MajoranaBasis::key_eq(a_planes, b_planes),
        *a == *b,
        "key_eq mismatch for {}",
        ctx()
    );
    if MajoranaBasis::key_eq(a_planes, b_planes) {
        assert_eq!(
            MajoranaBasis::key_hash(a_planes),
            MajoranaBasis::key_hash(b_planes),
            "key_eq monomials must key_hash equally for {}",
            ctx(),
        );
    }
}

#[test]
fn majorana_basis_matches_aos_exhaustive_small() {
    for n_qubits in 1usize..=4 {
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        let space = 1u64 << n_modes;
        for a_bits in 0..space {
            let a = mon(a_bits, n_modes);
            for b_bits in 0..space {
                let b = mon(b_bits, n_modes);
                assert_majorana_basis_matches(&a, &b, stride);
            }
        }
    }
}

#[test]
fn majorana_basis_matches_aos_randomized_multiword() {
    let mut rng = Rng(0xFEED_FACE_C0FF_EE00);
    for &n_qubits in &[30usize, 33, 64, 100, 128] {
        let n_modes = 2 * n_qubits;
        let stride = MajoranaBasis::stride_words(n_modes);
        for _ in 0..100 {
            let a = random_mon(&mut rng, n_modes);
            let b = random_mon(&mut rng, n_modes);
            assert_majorana_basis_matches(&a, &b, stride);
        }
    }
}

#[test]
fn majorana_basis_key_eq_and_hash_ignore_p_plane() {
    let stride = 1;
    let a = mon(0b0101, 8);
    let (a0, a1) = planes_of(&a, stride);
    let mut a1_garbage = a1.clone();
    a1_garbage[0] ^= 0xDEAD_BEEF;
    assert!(MajoranaBasis::key_eq([&a0, &a1], [&a0, &a1_garbage]));
    assert_eq!(
        MajoranaBasis::key_hash([&a0, &a1]),
        MajoranaBasis::key_hash([&a0, &a1_garbage]),
    );
    let c = mon(0b1111, 8);
    let (c0, c1) = planes_of(&c, stride);
    assert!(!MajoranaBasis::key_eq([&a0, &a1], [&c0, &c1]));
}
