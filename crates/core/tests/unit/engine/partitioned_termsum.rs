use super::*;
use crate::termsum::EmitCutoff;
use num_complex::Complex64;

const W: usize = 1;

/// The same minimal algebra the single-partition tests use.
struct TestAlgebra;

impl Basis<W> for TestAlgebra {
    const KIND: crate::basis::BasisKind = crate::basis::BasisKind::Pauli;

    type GenContext = BasisString<W>;

    fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> Self::GenContext {
        assert_eq!(sign, 1.0, "the test algebra carries no generator sign");
        *gen
    }
    fn generator(ctx: &Self::GenContext) -> &BasisString<W> {
        ctx
    }
    fn anticommutes(ctx: &Self::GenContext, mono: &BasisString<W>) -> bool {
        mono.parity_and(ctx)
    }

    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W> {
        ctx
    }
    fn product(ctx: &Self::GenContext, mono: &BasisString<W>) -> (BasisString<W>, Complex64) {
        (*mono ^ *ctx, Complex64::new(0.0, 1.0))
    }
    fn weight(mono: &BasisString<W>, _n_units: usize) -> u32 {
        mono.count() as u32
    }
    fn trace(mono: &BasisString<W>, _n_units: usize, fock: &[u64]) -> f64 {
        let f = fock.first().copied().unwrap_or(0);
        if mono.words()[0] & f == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

type Part = PartitionedTermSum<f64, u16, W>;
type Single = TermSum<f64, u16, W>;

fn mono(bits: &[usize]) -> BasisString<W> {
    BasisString::from_positions(bits.iter().copied())
}

fn values<I: Iterator<Item = (BasisString<W>, f64)>>(it: I) -> std::collections::HashMap<u64, f64> {
    it.filter(|(_, c)| *c != 0.0)
        .map(|(k, c)| (k.words()[0], c))
        .collect()
}

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


fn run_both(
    seed: u64,
    n_partitions: usize,
    n_gates: usize,
) -> (
    std::collections::HashMap<u64, f64>,
    std::collections::HashMap<u64, f64>,
) {
    run_both_with(seed, n_partitions, n_gates, &EmitCutoff::none())
}


fn run_both_with(
    seed: u64,
    n_partitions: usize,
    n_gates: usize,
    cutoff: &EmitCutoff,
) -> (
    std::collections::HashMap<u64, f64>,
    std::collections::HashMap<u64, f64>,
) {
    run_both_inner(seed, n_partitions, n_gates, cutoff)
}


fn run_both_inner(
    seed: u64,
    n_partitions: usize,
    n_gates: usize,
    cutoff: &EmitCutoff,
) -> (
    std::collections::HashMap<u64, f64>,
    std::collections::HashMap<u64, f64>,
) {
    let mut rng = Rng(seed);
    let seeds: Vec<(BasisString<W>, f64)> = (0..4)
        .map(|_| (mono(&[rng.below(6) as usize]), 1.0 + rng.unit()))
        .collect();
    let gates: Vec<(BasisString<W>, f64)> = (0..n_gates)
        .map(|_| {

            let a = rng.below(6) as usize;
            let b = (a + 1 + rng.below(5) as usize) % 6;
            (mono(&[a, b]), 0.1 + rng.unit())
        })
        .collect();

    let mut single = Single::new(8);
    for (k, c) in &seeds {
        single.add(k, *c).unwrap();
    }
    for (g, angle) in &gates {
        single
            .apply_rotation::<TestAlgebra>(g, angle, cutoff)
            .unwrap();
    }

    let mut part = Part::new(8, n_partitions);
    for (k, c) in &seeds {
        part.add(k, *c).unwrap();
    }
    for (g, angle) in &gates {
        part.apply_rotation::<TestAlgebra>(g, angle, cutoff)
            .unwrap();
    }

    (
        values(single.iter().map(|(k, c)| (k, *c))),
        values(
            part.iter::<TestAlgebra>()
                .map(|(k, sign, c)| (k, sign * *c)),
        ),
    )
}

#[test]
fn one_partition_matches_the_single_partition_engine() {
    let (want, got) = run_both(0x9E37_79B9_7F4A_7C15, 1, 20);
    assert_eq!(got, want);
}

#[test]
fn the_pair_rescue_is_independent_of_partition_count() {

    let cutoff = EmitCutoff {
        min_coeff: Some(0.1),
        ..Default::default()
    };
    for &s in &[1usize, 2, 3, 5, 8] {
        let (want, got) = run_both_with(0x1234_5678_9ABC_DEF1, s, 40, &cutoff);
        assert_eq!(got.len(), want.len(), "{s} partitions: term count diverged");
        for (key, wv) in &want {
            let gv = got
                .get(key)
                .unwrap_or_else(|| panic!("{s} partitions: key {key} missing"));
            assert!(
                (gv - wv).abs() <= 1e-9 * wv.abs().max(1.0),
                "{s} partitions: key {key} diverged: got {gv} want {wv}"
            );
        }
    }
}

#[test]
fn reclaim_drops_decayed_terms_and_leaves_the_store_usable() {

    let cutoff = EmitCutoff {
        min_coeff: Some(0.1),
        ..Default::default()
    };
    let mut op = PartitionedTermSum::<f64, u8, W>::new(8, 4);
    for k in 0..24u64 {
        op.add(&mono(&[(k % 6) as usize, ((k / 6) + 1) as usize]), 1.0)
            .unwrap();
    }
    let before = op.len();
    assert!(
        before > 4,
        "need enough terms to spread over four partitions"
    );

    op.scale_by_weight::<TestAlgebra>(|_| 1e-6);
    let dropped = op.reclaim::<TestAlgebra>(&cutoff).unwrap();
    assert_eq!(dropped, before, "every term was below the cutoff");
    assert_eq!(op.len(), 0);

    op.add(&mono(&[0]), 1.0).unwrap();
    op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none())
        .unwrap();
    assert_eq!(op.len(), 2, "the rebuilt store must still branch");
}

#[test]
fn reclaim_keeps_what_the_cutoff_still_admits() {
    let cutoff = EmitCutoff {
        min_coeff: Some(0.1),
        ..Default::default()
    };
    let mut op = PartitionedTermSum::<f64, u8, W>::new(8, 4);
    op.add(&mono(&[0]), 1.0).unwrap();
    op.add(&mono(&[1]), 1e-9).unwrap();
    op.add(&mono(&[2]), 0.5).unwrap();
    let dropped = op.reclaim::<TestAlgebra>(&cutoff).unwrap();
    assert_eq!(dropped, 1, "only the term under the cutoff goes");
    assert_eq!(op.len(), 2);
}

#[test]
fn reclaim_without_a_cutoff_is_a_no_op() {
    let mut op = PartitionedTermSum::<f64, u8, W>::new(8, 4);
    op.add(&mono(&[0]), 1e-30).unwrap();
    assert_eq!(op.reclaim::<TestAlgebra>(&EmitCutoff::none()).unwrap(), 0);
    assert_eq!(
        op.len(),
        1,
        "nothing may be dropped when nothing was asked for"
    );
}

#[test]
fn partition_count_does_not_change_the_result() {
    for &s in &[1usize, 2, 3, 4, 8, 16] {
        let (want, got) = run_both(0x2545_F491_4F6C_DD1D, s, 24);
        assert_eq!(got.len(), want.len(), "{s} partitions: term count diverged");
        for (key, wv) in &want {
            let gv = got
                .get(key)
                .unwrap_or_else(|| panic!("{s} partitions: key {key} missing"));
            assert!(
                (gv - wv).abs() <= 1e-9 * wv.abs().max(1.0),
                "{s} partitions: key {key} diverged: got {gv} want {wv}"
            );
        }
    }
}

#[test]
fn a_term_lives_only_in_the_partition_that_owns_its_key() {
    let mut part = Part::new(8, 4);
    let mut rng = Rng(0x853C_49E6_748F_EA9B);
    for _ in 0..64 {
        part.add(&mono(&[rng.below(6) as usize]), 1.0).unwrap();
    }
    for _ in 0..12 {
        let a = rng.below(6) as usize;
        let b = (a + 1) % 6;
        part.apply_rotation::<TestAlgebra>(&mono(&[a, b]), &0.3, &EmitCutoff::none())
            .unwrap();
    }
    for (idx, p) in part.partitions.iter().enumerate() {
        for (key, _) in p.iter() {
            assert_eq!(
                partition_of(&key, 4),
                idx,
                "a key was stored outside its owning partition"
            );
        }
    }
}

#[test]
fn expectation_agrees_across_partition_counts() {
    let fock = [0b101u64];
    let mut baseline = None;
    for &s in &[1usize, 2, 5, 8] {
        let mut part = Part::new(8, s);
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for _ in 0..16 {
            part.add(&mono(&[rng.below(6) as usize]), 1.0 + rng.unit())
                .unwrap();
        }
        for _ in 0..10 {
            let a = rng.below(6) as usize;
            let b = (a + 1) % 6;
            part.apply_rotation::<TestAlgebra>(&mono(&[a, b]), &0.3, &EmitCutoff::none())
                .unwrap();
        }
        let got = part.expectation::<TestAlgebra>(&fock);
        match baseline {
            None => baseline = Some(got),
            Some(want) => assert!(
                (got - want).abs() < 1e-9,
                "{s} partitions: expectation {got} vs {want}"
            ),
        }
    }
}

#[test]
fn a_phase_only_rotation_scales_every_partition_without_appending() {
    let mut part = Part::new(8, 4);
    for q in 0..6usize {
        part.add(&mono(&[q]), 1.0).unwrap();
    }
    let before = part.len();
    let added = part
        .apply_rotation::<TestAlgebra>(&mono(&[0, 1]), &std::f64::consts::PI, &EmitCutoff::none())
        .unwrap();
    assert_eq!(added, 0);
    assert_eq!(
        part.len(),
        before,
        "a phase-only rotation must not grow the store"
    );
}

#[test]
fn an_empty_operator_is_a_no_op() {
    let mut part = Part::new(8, 4);
    assert_eq!(
        part.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none())
            .unwrap(),
        0
    );
    assert!(part.is_empty());
}

struct QubitLocalNoise {
    unit: usize,
    factor: f64,
}

impl crate::term_kernel::NoiseKernel for QubitLocalNoise {
    fn factor(&self, term: crate::term_kernel::TermView<'_>) -> f64 {
        if term.words[self.unit / 64] >> (self.unit % 64) & 1 != 0 {
            self.factor
        } else {
            1.0
        }
    }
}

struct SupportOnly {
    allowed: u64,
}

impl crate::term_kernel::TruncationKernel for SupportOnly {
    fn keep(&self, term: crate::term_kernel::TermView<'_>, _coeff_magnitude: f64) -> bool {
        term.words[0] & !self.allowed == 0
    }

    fn keep_batch(
        &self,
        _basis_kind: crate::basis::BasisKind,
        words: &[u64],
        stride: usize,
        weights: &[u32],
        _n_units: usize,
        _coeff_magnitudes: &[f64],
        out: &mut [u8],
    ) -> bool {
        for i in 0..weights.len() {
            out[i] = u8::from(words[i * stride] & !self.allowed == 0);
        }
        true
    }
}

#[test]
fn scale_by_key_gives_the_same_answer_at_every_partition_count() {
    let keys: Vec<_> = (0..24usize).map(|q| mono(&[q % 6, (q + 2) % 6])).collect();
    let mut baseline = None;
    for s in [1usize, 2, 4, 8] {
        let mut part = Part::new(8, s);
        for (i, key) in keys.iter().enumerate() {
            part.add(key, 1.0 + i as f64).unwrap();
        }
        part.scale_by_key::<TestAlgebra>(&QubitLocalNoise {
            unit: 3,
            factor: 0.125,
        });
        let got = values(part.iter::<TestAlgebra>().map(|(k, sign, c)| (k, sign * c)));
        match baseline {
            None => baseline = Some(got),
            Some(ref want) => assert_eq!(&got, want, "{s} partitions disagreed"),
        }
    }
}

#[test]
fn scale_by_key_matches_the_single_partition_operator() {
    let keys: Vec<_> = (0..10usize).map(|q| mono(&[q % 5])).collect();
    let mut single = Single::new(8);
    let mut part = Part::new(8, 4);
    for (i, key) in keys.iter().enumerate() {
        single.add(key, 1.0 + i as f64).unwrap();
        part.add(key, 1.0 + i as f64).unwrap();
    }
    let kernel = QubitLocalNoise {
        unit: 0,
        factor: 0.25,
    };
    single.scale_by_key::<TestAlgebra>(&kernel);
    part.scale_by_key::<TestAlgebra>(&kernel);
    assert_eq!(
        values(part.iter::<TestAlgebra>().map(|(k, sign, c)| (k, sign * c))),
        values(single.iter().map(|(k, c)| (k, *c))),
    );
}

#[test]
fn a_term_predicate_reclaims_the_terms_outside_its_mask() {
    let cutoff = EmitCutoff {
        term: Some(std::sync::Arc::new(SupportOnly {
            allowed: 0b0101_0101,
        })),
        ..Default::default()
    };
    let mut part = Part::new(8, 4);
    for q in 0..8usize {
        part.add(&mono(&[q]), 1.0).unwrap();
    }
    let dropped = part.reclaim::<TestAlgebra>(&cutoff).unwrap();
    assert_eq!(dropped, 4, "the four odd positions leave the mask");
    let kept = values(part.iter::<TestAlgebra>().map(|(k, sign, c)| (k, sign * c)));
    assert!(kept.keys().all(|k| k & !0b0101_0101u64 == 0));
}

#[test]
fn a_term_predicate_holds_across_a_whole_rotation_at_every_partition_count() {
    let cutoff = EmitCutoff {
        term: Some(std::sync::Arc::new(SupportOnly {
            allowed: 0b0011_1111,
        })),
        ..Default::default()
    };
    let mut baseline = None;
    for s in [1usize, 2, 4] {
        let mut part = Part::new(8, s);
        for q in 0..6usize {
            part.add(&mono(&[q]), 1.0 + q as f64).unwrap();
        }
        for a in 0..5usize {
            part.apply_rotation::<TestAlgebra>(&mono(&[a, a + 1]), &0.3, &cutoff)
                .unwrap();
        }
        // Nothing outside the mask may have been created along the way.
        let got = values(part.iter::<TestAlgebra>().map(|(k, sign, c)| (k, sign * c)));
        assert!(got.keys().all(|k| k & !0b0011_1111u64 == 0));
        match baseline {
            None => baseline = Some(got),
            Some(ref want) => assert_eq!(&got, want, "{s} partitions disagreed"),
        }
    }
}
