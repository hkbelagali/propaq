use super::*;
use crate::strings::BasisString;

const W: usize = 1;

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

type Op = TermSum<f64, u16, W>;

fn mono(bits: &[usize]) -> BasisString<W> {
    BasisString::from_positions(bits.iter().copied())
}

fn values(op: &Op) -> std::collections::HashMap<u64, f64> {
    op.iter().map(|(k, c)| (k.words()[0], *c)).collect()
}

#[test]
fn add_folds_a_duplicate_key_instead_of_appending() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1.0).unwrap();
    op.add(&mono(&[1]), 2.0).unwrap();
    op.add(&mono(&[0]), 3.0).unwrap();
    assert_eq!(op.len(), 2, "the duplicate must fold, not append");
    let v = values(&op);
    assert_eq!(v[&0b01], 4.0);
    assert_eq!(v[&0b10], 2.0);
}

#[test]
fn a_commuting_term_is_untouched_by_a_rotation() {
    let mut op = Op::new(8);
    op.add(&mono(&[1]), 5.0).unwrap();
    // Overlap with generator {0} is zero, so it commutes.
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[0]), &0.7, &EmitCutoff::none())
        .unwrap();
    assert_eq!(added, 0);
    assert_eq!(op.len(), 1);
    assert_eq!(*op.coeff(0), 5.0);
}

#[test]
fn an_anticommuting_term_splits_into_cos_and_sin_branches() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 2.0).unwrap();
    let angle = 0.3f64;
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[0]), &angle, &EmitCutoff::none())
        .unwrap();
    assert_eq!(added, 1);
    assert_eq!(op.len(), 2);
    let v = values(&op);
    assert!((v[&0b01] - 2.0 * angle.cos()).abs() < 1e-12, "cos branch");
    assert!(
        (v[&0b00] - -(2.0 * angle.sin())).abs() < 1e-12,
        "sin branch"
    );
}

fn paired_generator() -> BasisString<W> {
    mono(&[0, 1])
}

#[test]
fn a_child_landing_on_an_existing_row_accumulates_rather_than_appending() {
    let mut op = Op::new(8);
    // Both keys anticommute with {0,1}, and each maps onto the other.
    op.add(&mono(&[0]), 2.0).unwrap();
    op.add(&mono(&[1]), 3.0).unwrap();
    assert_eq!(op.len(), 2);
    let added = op
        .apply_rotation::<TestAlgebra>(&paired_generator(), &0.3, &EmitCutoff::none())
        .unwrap();
    assert_eq!(
        added, 0,
        "both children already exist, so nothing is appended"
    );
    assert_eq!(op.len(), 2);
}

#[test]
fn the_sine_branch_is_taken_against_the_pre_rotation_coefficient() {
    let angle = 0.4f64;
    let (c0, c1) = (2.0f64, 5.0f64);
    let mut op = Op::new(8);
    op.add(&mono(&[0]), c0).unwrap();
    op.add(&mono(&[1]), c1).unwrap();
    op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &EmitCutoff::none())
        .unwrap();

    let v = values(&op);
    let (sin_t, cos_t) = angle.sin_cos();
    let want_0 = c0 * cos_t - c1 * sin_t;
    let want_1 = c1 * cos_t - c0 * sin_t;
    assert!(
        (v[&0b01] - want_0).abs() < 1e-12,
        "got {}, want {want_0}",
        v[&0b01]
    );
    assert!(
        (v[&0b10] - want_1).abs() < 1e-12,
        "got {}, want {want_1}",
        v[&0b10]
    );
}

#[test]
fn a_phase_only_rotation_scales_without_appending() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 2.0).unwrap();
    op.add(&mono(&[1]), 3.0).unwrap();
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[0]), &std::f64::consts::PI, &EmitCutoff::none())
        .unwrap();
    assert_eq!(added, 0);
    assert_eq!(op.len(), 2, "a phase-only rotation must not grow the store");
    let v = values(&op);
    assert!(
        (v[&0b01] + 2.0).abs() < 1e-12,
        "anticommuting term scaled by cos(pi)"
    );
    assert_eq!(v[&0b10], 3.0, "commuting term untouched");
}

#[test]
fn a_weight_cutoff_suppresses_the_child_at_emit_time() {
    let mut op = Op::new(8);
    op.add(&mono(&[0, 1]), 1.0).unwrap();
    let cutoff = EmitCutoff {
        max_weight: Some(2),
        ..Default::default()
    };
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[2]), &0.3, &cutoff)
        .unwrap();
    assert_eq!(added, 0, "the over-weight child must never be created");
    assert_eq!(op.len(), 1);
}

#[test]
fn a_coefficient_cutoff_suppresses_a_small_child_at_emit_time() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1e-12).unwrap();
    let cutoff = EmitCutoff {
        min_coeff: Some(1e-6),
        ..Default::default()
    };
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &cutoff)
        .unwrap();
    assert_eq!(added, 0, "the tiny child must never be created");
    assert_eq!(op.len(), 1);
}

#[test]
fn a_pair_rescue_keeps_the_branch_its_partner_paid_for() {
    let angle = 0.3f64;
    let (big, small) = (1.0f64, 1e-9f64);
    let cutoff = EmitCutoff {
        min_coeff: Some(1e-6),
        ..Default::default()
    };
    let mut op = Op::new(8);
    op.add(&mono(&[0]), big).unwrap();
    op.add(&mono(&[1]), small).unwrap();
    op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &cutoff)
        .unwrap();

    let v = values(&op);
    let (sin_t, cos_t) = angle.sin_cos();
    assert_eq!(op.len(), 2, "a rescue must not create a row");
    assert!(
        (v[&0b01] - (big * cos_t - small * sin_t)).abs() < 1e-18,
        "the rejected branch is owed back to its partner, got {}",
        v[&0b01]
    );
    assert!(
        (v[&0b10] - (small * cos_t - big * sin_t)).abs() < 1e-15,
        "the cleared branch"
    );
}

#[test]
fn a_term_floor_suppresses_the_lossy_predicates_while_the_store_is_small() {
    let angle = 0.3f64;
    let biting = EmitCutoff {
        min_coeff: Some(1e-6),
        ..Default::default()
    };
    assert!(
        biting.at_size(1).min_coeff.is_some(),
        "no floor set, so nothing is suppressed"
    );

    let floored = EmitCutoff {
        min_coeff: Some(1e-6),
        min_terms: Some(1000),
        ..Default::default()
    };
    assert!(
        floored.at_size(10).min_coeff.is_none(),
        "below the floor the bound must be off"
    );
    assert!(
        floored.at_size(1000).min_coeff.is_some(),
        "at the floor the bound comes back"
    );

    let tiny = 1e-9f64;
    let gen = mono(&[0]);
    let mut with_floor = Op::new(8);
    with_floor.add(&mono(&[0]), tiny).unwrap();
    let effective = floored.at_size(with_floor.len());
    with_floor
        .apply_rotation::<TestAlgebra>(&gen, &angle, &effective)
        .unwrap();
    assert_eq!(
        with_floor.len(),
        2,
        "below the floor the tiny branch must survive"
    );

    let mut no_floor = Op::new(8);
    no_floor.add(&mono(&[0]), tiny).unwrap();
    let effective = biting.at_size(no_floor.len());
    no_floor
        .apply_rotation::<TestAlgebra>(&gen, &angle, &effective)
        .unwrap();
    assert_eq!(
        no_floor.len(),
        1,
        "without a floor the same branch must be refused"
    );
}

#[test]
fn a_term_floor_carries_through_a_lossless_copy() {
    let c = EmitCutoff {
        max_weight: Some(3),
        min_coeff: Some(1e-6),
        min_terms: Some(50),
        ..Default::default()
    };
    let l = c.lossless();
    assert_eq!(l.min_terms, Some(50));
    assert!(l.max_weight.is_none() && l.min_coeff.is_none() && l.native.is_none());
}

#[test]
fn a_pair_rescue_does_not_revive_a_branch_neither_half_earned() {
    let angle = 0.3f64;
    let cutoff = EmitCutoff {
        min_coeff: Some(1e-6),
        ..Default::default()
    };
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1e-9).unwrap();
    op.add(&mono(&[1]), 2e-9).unwrap();
    op.apply_rotation::<TestAlgebra>(&paired_generator(), &angle, &cutoff)
        .unwrap();

    let v = values(&op);
    let cos_t = angle.cos();
    assert_eq!(op.len(), 2);
    assert!((v[&0b01] - 1e-9 * cos_t).abs() < 1e-24);
    assert!((v[&0b10] - 2e-9 * cos_t).abs() < 1e-24);
}

#[test]
fn a_pair_rescue_needs_both_halves_in_the_store() {
    let cutoff = EmitCutoff {
        min_coeff: Some(1e-6),
        ..Default::default()
    };
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1e-9).unwrap();
    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &cutoff)
        .unwrap();
    assert_eq!(added, 0, "a lone tiny term still may not create a child");
    assert_eq!(op.len(), 1);
}

#[test]
fn repeated_rotations_keep_the_store_deduplicated() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1.0).unwrap();
    for _ in 0..12 {
        op.apply_rotation::<TestAlgebra>(&mono(&[0]), &0.3, &EmitCutoff::none())
            .unwrap();
    }

    assert_eq!(op.len(), 2, "dedup on insert must bound the orbit");
}

#[test]
fn expectation_sums_coefficient_times_trace() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 2.0).unwrap();
    op.add(&mono(&[1]), 3.0).unwrap();
    let got = op.expectation::<TestAlgebra>(&[0b01]);
    assert!((got - (-2.0 + 3.0)).abs() < 1e-12);
}

#[test]
fn rows_are_never_removed_so_indices_stay_stable() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1.0).unwrap();
    let key0 = op.key(0);
    for _ in 0..8 {
        op.apply_rotation::<TestAlgebra>(&mono(&[1]), &0.3, &EmitCutoff::none())
            .unwrap();
    }
    assert_eq!(op.key(0), key0, "row 0 must still hold its original key");
}

#[test]
fn u16_positions_are_wide_enough_for_this_width() {
    assert!(BasisString::<W>::num_bits() <= u16::MAX as usize);
}

struct QubitLocalNoise {
    unit: usize,
    factor: f64,
}

impl crate::term_kernel::NoiseKernel for QubitLocalNoise {
    fn factor(&self, term: crate::term_kernel::TermView<'_>) -> f64 {
        let touched = term.words[self.unit / 64] >> (self.unit % 64) & 1 != 0;
        if touched {
            self.factor
        } else {
            1.0
        }
    }
}

struct WeightOnlyNoise {
    damping: f64,
}

impl crate::term_kernel::NoiseKernel for WeightOnlyNoise {
    fn factor(&self, term: crate::term_kernel::TermView<'_>) -> f64 {
        (-self.damping * term.weight as f64).exp()
    }
}

struct SupportOnly {
    allowed: u64,
}

impl crate::term_kernel::TruncationKernel for SupportOnly {
    fn keep(&self, term: crate::term_kernel::TermView<'_>, _coeff_magnitude: f64) -> bool {
        term.words[0] & !self.allowed == 0
    }
}

#[test]
fn scale_by_key_damps_only_the_terms_touching_the_chosen_unit() {
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1.0).unwrap();
    op.add(&mono(&[1]), 1.0).unwrap();
    op.add(&mono(&[0, 1]), 1.0).unwrap();
    op.scale_by_key::<TestAlgebra>(&QubitLocalNoise {
        unit: 0,
        factor: 0.25,
    });
    let v = values(&op);
    assert_eq!(v[&0b01], 0.25);

    assert_eq!(v[&0b10], 1.0);
    assert_eq!(v[&0b11], 0.25);
}

#[test]
fn a_weight_only_kernel_matches_the_table_pass_bit_for_bit() {
    let damping = 0.37;
    let keys = [mono(&[0]), mono(&[1]), mono(&[0, 1]), mono(&[2, 3, 4])];
    let mut by_table = Op::new(8);
    let mut by_kernel = Op::new(8);
    for (i, key) in keys.iter().enumerate() {
        by_table.add(key, 1.0 + i as f64).unwrap();
        by_kernel.add(key, 1.0 + i as f64).unwrap();
    }
    by_table.scale_by_weight::<TestAlgebra>(|w| (-damping * w as f64).exp());
    by_kernel.scale_by_key::<TestAlgebra>(&WeightOnlyNoise { damping });
    assert_eq!(values(&by_table), values(&by_kernel));
}

#[test]
fn scale_by_key_spans_more_than_one_batch_chunk() {
    let n = crate::term_kernel::KERNEL_BATCH + 37;
    let mut op = Op::new(32);
    for i in 0..n {
        op.add(&BasisString::<W>::from_words([i as u64 + 1]), 1.0)
            .unwrap();
    }
    op.scale_by_key::<TestAlgebra>(&QubitLocalNoise {
        unit: 0,
        factor: 0.5,
    });
    for i in 0..n {
        let expected = if (i as u64 + 1) & 1 != 0 { 0.5 } else { 1.0 };
        assert_eq!(*op.coeff(i), expected, "row {i} scaled wrongly");
    }
}

#[test]
fn a_term_predicate_refuses_a_child_outside_its_support_mask() {
    let cutoff = EmitCutoff {
        term: Some(std::sync::Arc::new(SupportOnly { allowed: 0b0011 })),
        ..Default::default()
    };
    let mut op = Op::new(8);
    op.add(&mono(&[0]), 1.0).unwrap();

    let added = op
        .apply_rotation::<TestAlgebra>(&mono(&[2]), &0.3, &cutoff)
        .unwrap();
    assert_eq!(added, 0, "a child outside the mask must not be created");
    assert_eq!(op.len(), 1);

    let added = op
        .apply_rotation::<TestAlgebra>(&paired_generator(), &0.3, &cutoff)
        .unwrap();
    assert_eq!(added, 1);
}

#[test]
fn reclaim_by_kernel_agrees_with_the_per_term_predicate() {
    let kernel = SupportOnly { allowed: 0b0101 };
    let mut op = Op::new(8);
    for key in [mono(&[0]), mono(&[1]), mono(&[2]), mono(&[0, 2])] {
        op.add(&key, 1.0).unwrap();
    }
    let dropped = op.reclaim_by_kernel::<TestAlgebra>(&kernel).unwrap();
    assert_eq!(dropped, 1, "only {{1}} leaves the mask");
    let v = values(&op);
    assert_eq!(v.len(), 3);
    assert!(!v.contains_key(&0b10));
}
