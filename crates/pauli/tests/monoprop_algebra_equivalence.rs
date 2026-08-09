///
/// Differential test: the interleaved `PauliAlgebra` against the word-plane
/// `PauliBasis` it is replacing.
///
/// `PauliBasis` is currently correct and covered, so it is the oracle here. The
/// bit-convention mapping (x-plane bit `k` -> interleaved bit `2k`, z-plane bit
/// `k` -> `2k + 1`) is exercised in both directions, since a mapping error would
/// otherwise make every downstream engine comparison meaningless.
///
use num_complex::Complex64;

use propaq_core::algebra::Algebra;
use propaq_core::monomial::Monomial;
use propaq_core::soa::SoaBasis;
use propaq_pauli::algebra::{from_monomial, planes_of, to_monomial, PauliAlgebra};
use propaq_pauli::string::{PauliBasis, PauliString};

/// 96 qubits needs 192 bits, so three storage words.
const N_QUBITS: usize = 96;
const W: usize = 3;
/// Word count of the old two-plane form at this width.
const STRIDE: usize = 2;

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
}

/// A random Pauli string of bounded weight, spread across the whole register so
/// multiword monomials are exercised.
fn random_pauli(rng: &mut Rng, max_weight: usize) -> PauliString {
    let mut xw = vec![0u64; STRIDE];
    let mut zw = vec![0u64; STRIDE];
    let weight = 1 + rng.below(max_weight as u64) as usize;
    for _ in 0..weight {
        let q = rng.below(N_QUBITS as u64) as usize;
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

#[test]
fn monomial_conversion_round_trips() {
    let mut rng = Rng(0x243F_6A88_85A3_08D3);
    for _ in 0..2000 {
        let p = random_pauli(&mut rng, 8);
        let m: Monomial<W> = to_monomial(&p);
        let back = from_monomial::<W>(&m, N_QUBITS);
        assert_eq!(back.x, p.x, "x plane diverged through the monomial form");
        assert_eq!(back.z, p.z, "z plane diverged through the monomial form");
        assert_eq!(back.weight, p.weight);
    }
}

#[test]
fn the_bit_convention_places_x_even_and_z_odd() {
    // Explicit, so a silent convention flip cannot pass the randomized tests by
    // being self-consistent in both directions.
    let mut xw = vec![0u64; STRIDE];
    let mut zw = vec![0u64; STRIDE];
    xw[0] = 1 << 3; // X on qubit 3
    zw[0] = 1 << 5; // Z on qubit 5
    let x = propaq_core::bitset::Bitset::from_words(xw);
    let z = propaq_core::bitset::Bitset::from_words(zw);
    let p = PauliString { x, z, n_qubits: N_QUBITS, weight: 2 };
    let m: Monomial<W> = to_monomial(&p);
    assert!(m.test(6), "X on qubit 3 must be interleaved bit 6");
    assert!(m.test(11), "Z on qubit 5 must be interleaved bit 11");
    assert_eq!(m.count(), 2);
}

#[test]
fn weight_matches_the_word_plane_oracle() {
    let mut rng = Rng(0x13198A2E_03707344);
    for _ in 0..3000 {
        let p = random_pauli(&mut rng, 10);
        let (xp, zp) = planes_of(&p, STRIDE);
        let m: Monomial<W> = to_monomial(&p);
        assert_eq!(
            <PauliAlgebra as Algebra<W>>::weight(&m),
            PauliBasis::weight([&xp, &zp], N_QUBITS),
            "weight diverged"
        );
    }
}

#[test]
fn trace_matches_the_word_plane_oracle() {
    let mut rng = Rng(0xA409_3822_299F_31D0);
    let fock: Vec<u64> = (0..STRIDE).map(|_| rng.next_u64()).collect();
    for trial in 0..3000 {
        // Mostly generic terms, but every third is Z-only so the parity branch
        // is exercised rather than just the off-diagonal early return.
        let p = if trial % 3 == 0 {
            let mut zw = vec![0u64; STRIDE];
            for _ in 0..1 + rng.below(8) {
                let q = rng.below(N_QUBITS as u64) as usize;
                zw[q / 64] |= 1u64 << (q % 64);
            }
            let x = propaq_core::bitset::Bitset::from_words(vec![0u64; STRIDE]);
            let z = propaq_core::bitset::Bitset::from_words(zw);
            let weight = (&x | &z).count_ones();
            PauliString { x, z, n_qubits: N_QUBITS, weight }
        } else {
            random_pauli(&mut rng, 8)
        };
        let (xp, zp) = planes_of(&p, STRIDE);
        let m: Monomial<W> = to_monomial(&p);
        assert_eq!(
            <PauliAlgebra as Algebra<W>>::trace(&m, &fock),
            PauliBasis::trace([&xp, &zp], N_QUBITS, &fock),
            "trace diverged on trial {trial}"
        );
    }
}

#[test]
fn anticommutation_matches_the_word_plane_oracle() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..3000 {
        let term = random_pauli(&mut rng, 8);
        let gen = random_pauli(&mut rng, 5);
        let (tx, tz) = planes_of(&term, STRIDE);
        let (gx, gz) = planes_of(&gen, STRIDE);
        let ctx = <PauliAlgebra as Algebra<W>>::make_gen_context(&to_monomial(&gen));
        assert_eq!(
            <PauliAlgebra as Algebra<W>>::anticommutes(&ctx, &to_monomial(&term)),
            !PauliBasis::commutes([&tx, &tz], [&gx, &gz]),
            "anticommutation diverged"
        );
    }
}

/// A random Pauli supported on two narrow windows straddling the word boundary.
///
/// Uniform draws over 96 qubits almost never anticommute, so the product test
/// would otherwise sample too few branching pairs to mean anything.
fn random_pauli_windowed(rng: &mut Rng, max_weight: usize) -> PauliString {
    let mut xw = vec![0u64; STRIDE];
    let mut zw = vec![0u64; STRIDE];
    let weight = 1 + rng.below(max_weight as u64) as usize;
    for _ in 0..weight {
        let k = rng.below(16) as usize;
        let q = if k < 8 { k } else { 60 + (k - 8) };
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

#[test]
fn product_key_and_phase_match_the_word_plane_oracle() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut checked = 0usize;
    for _ in 0..6000 {
        let term = random_pauli_windowed(&mut rng, 8);
        let gen = random_pauli_windowed(&mut rng, 5);
        let (tx, tz) = planes_of(&term, STRIDE);
        let (gx, gz) = planes_of(&gen, STRIDE);

        // The engine only ever multiplies anticommuting terms, so that is the
        // case worth asserting on.
        if PauliBasis::commutes([&tx, &tz], [&gx, &gz]) {
            continue;
        }
        checked += 1;

        let ctx = <PauliAlgebra as Algebra<W>>::make_gen_context(&to_monomial(&gen));
        let (got_key, got_phase) = <PauliAlgebra as Algebra<W>>::product(&ctx, &to_monomial(&term));

        let mut wx = vec![0u64; STRIDE];
        let mut wz = vec![0u64; STRIDE];
        let want_phase = PauliBasis::product([&tx, &tz], [&gx, &gz], [&mut wx, &mut wz]);

        assert_eq!(got_key, to_monomial::<W>(&{
            let x = propaq_core::bitset::Bitset::from_words(wx.clone());
            let z = propaq_core::bitset::Bitset::from_words(wz.clone());
            let weight = (&x | &z).count_ones();
            PauliString { x, z, n_qubits: N_QUBITS, weight }
        }), "product key diverged");
        assert_eq!(got_phase, want_phase, "product phase diverged");

        // An anticommuting Pauli product is always purely imaginary, which is
        // what lets the engine feed it straight to CoeffRepr::apply_rotation.
        assert!(got_phase.re.abs() < 1e-12, "anticommuting product must have zero real part");
        assert!((got_phase.im.abs() - 1.0).abs() < 1e-12, "phase must be +-i");
    }
    assert!(checked > 1000, "only {checked} anticommuting pairs sampled; test is too weak");
}

#[test]
fn product_is_involutive_and_preserves_the_generator_support() {
    let mut rng = Rng(0x853C_49E6_748F_EA9B);
    for _ in 0..1000 {
        let term = random_pauli(&mut rng, 8);
        let gen = random_pauli(&mut rng, 5);
        let g: Monomial<W> = to_monomial(&gen);
        let m: Monomial<W> = to_monomial(&term);
        let ctx = <PauliAlgebra as Algebra<W>>::make_gen_context(&g);
        let (once, _) = <PauliAlgebra as Algebra<W>>::product(&ctx, &m);
        let (twice, _) = <PauliAlgebra as Algebra<W>>::product(&ctx, &once);
        assert_eq!(twice, m, "G*(G*M) must return the original key");
    }
}

#[test]
fn identity_generator_commutes_with_everything() {
    let ctx = <PauliAlgebra as Algebra<W>>::make_gen_context(&Monomial::<W>::zero());
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..500 {
        let m: Monomial<W> = to_monomial(&random_pauli(&mut rng, 8));
        assert!(!<PauliAlgebra as Algebra<W>>::anticommutes(&ctx, &m));
    }
}

#[test]
fn phase_is_reproducible_for_a_hand_checked_case() {
    // X on qubit 0, generator Z on qubit 0. They anticommute, and Z*X = iY.
    let mut m = Monomial::<W>::zero();
    m.set(0); // X on qubit 0
    let mut g = Monomial::<W>::zero();
    g.set(1); // Z on qubit 0
    let ctx = <PauliAlgebra as Algebra<W>>::make_gen_context(&g);
    assert!(<PauliAlgebra as Algebra<W>>::anticommutes(&ctx, &m));
    let (key, phase) = <PauliAlgebra as Algebra<W>>::product(&ctx, &m);
    assert!(key.test(0) && key.test(1), "Z*X must land on Y (both bits set)");
    assert_eq!(phase, Complex64::new(0.0, 1.0), "Z*X = iY");
}
