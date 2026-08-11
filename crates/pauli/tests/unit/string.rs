use super::*;
fn pauli(x: u64, z: u64, n: usize) -> PauliString {
    let xb = Bitset::from_le_bytes(&x.to_le_bytes());
    let zb = Bitset::from_le_bytes(&z.to_le_bytes());
    let weight = (&xb | &zb).count_ones();
    PauliString {
        x: xb,
        z: zb,
        n_qubits: n,
        weight,
    }
}

fn fock(bits: u64) -> Bitset {
    Bitset::from_le_bytes(&bits.to_le_bytes())
}

#[test]
fn identity_weight_zero() {
    assert_eq!(pauli(0, 0, 4).weight, 0);
}

#[test]
fn single_x_weight_one() {
    assert_eq!(pauli(0b01, 0, 4).weight, 1);
}

#[test]
fn single_z_weight_one() {
    assert_eq!(pauli(0, 0b01, 4).weight, 1);
}

#[test]
fn single_y_weight_one() {
    assert_eq!(pauli(0b01, 0b01, 4).weight, 1);
}

#[test]
fn identity_commutes_with_everything() {
    let id = pauli(0, 0, 4);
    let x = pauli(0b01, 0, 4);
    assert!(id.commutes_with_impl(&x));
    assert!(x.commutes_with_impl(&id));
}

#[test]
fn x_commutes_with_itself() {
    let x = pauli(0b01, 0, 4);
    assert!(x.commutes_with_impl(&x));
}

#[test]
fn x_anticommutes_z_same_qubit() {
    let x = pauli(0b01, 0, 4);
    let z = pauli(0, 0b01, 4);
    assert!(!x.commutes_with_impl(&z));
}

#[test]
fn x0_commutes_z1_different_qubits() {
    let x0 = pauli(0b01, 0, 4);
    let z1 = pauli(0, 0b10, 4);
    assert!(x0.commutes_with_impl(&z1));
}

#[test]
fn matmul_x_times_x_is_identity() {
    let x = pauli(0b01, 0, 4);
    let (phase, result) = x.matmul_impl(&x);
    assert!((phase - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    assert_eq!(result.weight, 0);
}

#[test]
fn matmul_x_times_z_gives_y_with_phase() {
    let x = pauli(0b01, 0, 4);
    let z = pauli(0, 0b01, 4);
    let (phase, result) = x.matmul_impl(&z);
    assert!((phase - Complex64::new(0.0, -1.0)).norm() < 1e-10);
    assert_eq!(result.weight, 1);
}

#[test]
fn trace_identity_is_one() {
    assert_eq!(pauli(0, 0, 4).trace_fock_state_impl(&fock(0)), 1.0);
}

#[test]
fn trace_x_is_zero() {
    assert_eq!(pauli(0b01, 0, 4).trace_fock_state_impl(&fock(0)), 0.0);
}

#[test]
fn trace_z0_empty_state() {
    assert_eq!(pauli(0, 0b01, 4).trace_fock_state_impl(&fock(0b00)), 1.0);
}

#[test]
fn trace_z0_occupied_state() {
    assert_eq!(pauli(0, 0b01, 4).trace_fock_state_impl(&fock(0b01)), -1.0);
}

#[test]
fn trace_zz_all_combinations() {
    let zz = pauli(0, 0b11, 4);
    assert_eq!(zz.trace_fock_state_impl(&fock(0b00)), 1.0);
    assert_eq!(zz.trace_fock_state_impl(&fock(0b01)), -1.0);
    assert_eq!(zz.trace_fock_state_impl(&fock(0b10)), -1.0);
    assert_eq!(zz.trace_fock_state_impl(&fock(0b11)), 1.0);
}

fn planes_of(p: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    let mut gz = vec![0u64; stride];
    PauliBasis::term_into_planes(p, p.n_qubits, [&mut gx, &mut gz]);
    (gx, gz)
}

fn assert_basis_matches(a: &PauliString, b: &PauliString) {
    let stride = 1;
    let (ax, az) = planes_of(a, stride);
    let (bx, bz) = planes_of(b, stride);
    let a_planes = [ax.as_slice(), az.as_slice()];
    let b_planes = [bx.as_slice(), bz.as_slice()];
    let ctx = || format!("a=(x={ax:?},z={az:?}) b=(x={bx:?},z={bz:?})");

    assert_eq!(
        PauliBasis::commutes(a_planes, b_planes),
        a.commutes_with_impl(b),
        "commutes mismatch for {}",
        ctx(),
    );
    assert_eq!(
        PauliBasis::weight(a_planes, a.n_qubits),
        a.weight,
        "weight mismatch for {}",
        ctx()
    );

    let (expected_phase, expected_result) = a.matmul_impl(b);
    let mut out_x = vec![0u64; stride];
    let mut out_z = vec![0u64; stride];
    let phase = PauliBasis::product(b_planes, a_planes, [&mut out_x, &mut out_z]);
    assert!(
        (phase - expected_phase).norm() < 1e-10,
        "phase mismatch for {}",
        ctx()
    );
    let result = PauliBasis::term_from_planes([&out_x, &out_z], a.n_qubits);
    assert_eq!(
        result.x,
        expected_result.x,
        "product x mismatch for {}",
        ctx()
    );
    assert_eq!(
        result.z,
        expected_result.z,
        "product z mismatch for {}",
        ctx()
    );

    for fock_bits in 0u64..16 {
        let fock_words = [fock_bits];
        assert_eq!(
            PauliBasis::trace(a_planes, a.n_qubits, &fock_words),
            a.trace_fock_state_impl(&fock(fock_bits)),
            "trace mismatch for {} fock={fock_bits}",
            ctx(),
        );
    }

    assert_eq!(
        PauliBasis::key_eq(a_planes, b_planes),
        *a == *b,
        "key_eq mismatch for {}",
        ctx()
    );
    if PauliBasis::key_eq(a_planes, b_planes) {
        assert_eq!(
            PauliBasis::key_hash(a_planes),
            PauliBasis::key_hash(b_planes),
            "key_eq strings must key_hash equally for {}",
            ctx(),
        );
    }
}

#[test]
fn pauli_basis_matches_aos_exhaustive_4_qubit() {
    for xa in 0u64..16 {
        for za in 0u64..16 {
            let a = pauli(xa, za, 4);
            for xb in 0u64..16 {
                for zb in 0u64..16 {
                    let b = pauli(xb, zb, 4);
                    assert_basis_matches(&a, &b);
                }
            }
        }
    }
}

#[test]
fn local_word_identifies_single_nonzero_word() {
    // gen confined to word 1 of a 3-word stride.
    let gen_x = vec![0u64, 0b100, 0u64];
    let gen_z = vec![0u64, 0u64, 0u64];
    assert_eq!(PauliBasis::local_word([&gen_x, &gen_z]), Some(1));

    // gen spanning two words -> not local.
    let gen_x2 = vec![1u64, 1u64, 0u64];
    let gen_z2 = vec![0u64, 0u64, 0u64];
    assert_eq!(PauliBasis::local_word([&gen_x2, &gen_z2]), None);

    // all-zero gen -> no nonzero word at all.
    let gen_x3 = vec![0u64, 0u64, 0u64];
    let gen_z3 = vec![0u64, 0u64, 0u64];
    assert_eq!(PauliBasis::local_word([&gen_x3, &gen_z3]), None);
}


#[test]
fn commutes_at_word_and_product_at_word_hand_checked() {
    let gen_word = [0u64, 1u64]; // (x=0, z=1) = Z
    let term_word = [1u64, 0u64]; // (x=1, z=0) = X
    assert!(!PauliBasis::commutes_at_word(term_word, gen_word));
    let (out_word, phase) = PauliBasis::product_at_word(term_word, gen_word);
    assert_eq!(out_word, [1u64, 1u64]); // X XOR Z (bitwise) = Y's (x=1,z=1) representation
    let term_x = [term_word[0]];
    let term_z = [term_word[1]];
    let gen_x = [gen_word[0]];
    let gen_z = [gen_word[1]];
    let mut out_x = [0u64];
    let mut out_z = [0u64];
    let generic_phase = PauliBasis::product(
        [&term_x, &term_z],
        [&gen_x, &gen_z],
        [&mut out_x, &mut out_z],
    );
    assert_eq!(phase, generic_phase);
    assert_eq!(out_word, [out_x[0], out_z[0]]);
}

#[test]
fn local_word_fast_path_matches_generic_across_random_multi_word_cases() {
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for &stride in &[1usize, 2, 3, 5] {
        for _trial in 0..300 {
            let bit_pos = (next() as usize) % (stride * 64);
            let word = bit_pos / 64;
            let bit = bit_pos % 64;
            let kind = next() % 3; // 0=X, 1=Z, 2=Y
            let mut gen_x = vec![0u64; stride];
            let mut gen_z = vec![0u64; stride];
            match kind {
                0 => gen_x[word] = 1u64 << bit,
                1 => gen_z[word] = 1u64 << bit,
                _ => {
                    gen_x[word] = 1u64 << bit;
                    gen_z[word] = 1u64 << bit;
                }
            }
            let gen = [gen_x.as_slice(), gen_z.as_slice()];

            assert_eq!(
                PauliBasis::local_word(gen),
                Some(word),
                "stride={stride} trial={_trial}: local_word should identify the single nonzero word"
            );

            let mut term_x = vec![0u64; stride];
            let mut term_z = vec![0u64; stride];
            for w in 0..stride {
                term_x[w] = next();
                term_z[w] = next();
            }
            let term = [term_x.as_slice(), term_z.as_slice()];

            let generic_commutes = PauliBasis::commutes(term, gen);
            let fast_commutes = PauliBasis::commutes_at_word(
                [term_x[word], term_z[word]],
                [gen_x[word], gen_z[word]],
            );
            assert_eq!(
                generic_commutes, fast_commutes,
                "stride={stride} trial={_trial}: commutes mismatch (term_x={term_x:?} term_z={term_z:?} gen_x={gen_x:?} gen_z={gen_z:?})"
            );

            if !generic_commutes {
                let mut out_x = vec![0u64; stride];
                let mut out_z = vec![0u64; stride];
                let generic_phase = PauliBasis::product(term, gen, [&mut out_x, &mut out_z]);
                let (fast_out_word, fast_phase) = PauliBasis::product_at_word(
                    [term_x[word], term_z[word]],
                    [gen_x[word], gen_z[word]],
                );
                assert_eq!(
                    fast_phase, generic_phase,
                    "stride={stride} trial={_trial}: phase mismatch (term_x={term_x:?} term_z={term_z:?} gen_x={gen_x:?} gen_z={gen_z:?})"
                );
                assert_eq!(
                    out_x[word], fast_out_word[0],
                    "stride={stride} trial={_trial}: out x-word mismatch"
                );
                assert_eq!(
                    out_z[word], fast_out_word[1],
                    "stride={stride} trial={_trial}: out z-word mismatch"
                );
                for w in 0..stride {
                    if w != word {
                        assert_eq!(
                            out_x[w], term_x[w],
                            "stride={stride} trial={_trial}: word {w} x should be untouched"
                        );
                        assert_eq!(
                            out_z[w], term_z[w],
                            "stride={stride} trial={_trial}: word {w} z should be untouched"
                        );
                    }
                }
            }
        }
    }
}
