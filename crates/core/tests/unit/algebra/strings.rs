use super::*;

#[test]
fn zero_is_empty() {
    let m = BasisString::<2>::zero();
    assert!(m.is_zero());
    assert_eq!(m.count(), 0);
    assert_eq!(m.positions().count(), 0);
}

#[test]
fn words_for_rounds_up_to_two_bits_per_unit() {
    assert_eq!(BasisString::<1>::words_for(0), 0);
    assert_eq!(BasisString::<1>::words_for(1), 1);
    assert_eq!(BasisString::<1>::words_for(32), 1);
    assert_eq!(BasisString::<1>::words_for(33), 2);
    assert_eq!(BasisString::<1>::words_for(36), 2);
    assert_eq!(BasisString::<1>::words_for(64), 2);
}

#[test]
fn set_test_clear_round_trip() {
    let mut m = BasisString::<2>::zero();
    for p in [0usize, 1, 63, 64, 127] {
        m.set(p);
    }
    for p in [0usize, 1, 63, 64, 127] {
        assert!(m.test(p), "bit {p} should be set");
    }
    assert!(!m.test(2));
    assert_eq!(m.count(), 5);
    m.clear(63);
    assert!(!m.test(63));
    assert_eq!(m.count(), 4);
}

#[test]
fn positions_are_ascending_and_complete() {
    let want = [0usize, 5, 63, 64, 65, 191];
    let m = BasisString::<3>::from_positions(want);
    let got: Vec<usize> = m.positions().collect();
    assert_eq!(got, want);
    assert!(got.windows(2).all(|w| w[0] < w[1]));
}

#[test]
fn xor_is_the_product_key_and_is_involutive() {
    let a = BasisString::<2>::from_positions([1usize, 5, 70]);
    let b = BasisString::<2>::from_positions([5usize, 9, 70]);
    let c = a ^ b;
    assert_eq!(c.positions().collect::<Vec<_>>(), vec![1, 9]);
    assert_eq!(c ^ b, a, "xor must be involutive in the second operand");
}

#[test]
fn count_and_and_parity_agree_with_a_naive_reference() {
    let a = BasisString::<2>::from_positions([1usize, 5, 9, 70]);
    let b = BasisString::<2>::from_positions([5usize, 9, 70, 100]);
    let naive = (a & b).count();
    assert_eq!(a.count_and(&b), naive);
    assert_eq!(a.parity_and(&b), naive % 2 == 1);
}

#[test]
fn support_counts_units_not_bits() {
    // Unit 0 has both bits set, unit 3 has one. Support is 2, count is 3.
    let m = BasisString::<1>::from_positions([0usize, 1, 6]);
    assert_eq!(m.count(), 3);
    assert_eq!(m.support(), 2);
}

#[test]
fn support_spans_word_boundaries() {
    // Unit 31 occupies bits 62 and 63; unit 32 occupies bits 64 and 65.
    let m = BasisString::<2>::from_positions([62usize, 63, 64]);
    assert_eq!(m.support(), 2);
}

#[test]
fn pair_swap_exchanges_each_units_two_bits() {
    let m = BasisString::<2>::from_positions([0usize, 3, 64]);
    let s = m.pair_swap();
    assert_eq!(s.positions().collect::<Vec<_>>(), vec![1, 2, 65]);
    assert_eq!(s.pair_swap(), m, "pair_swap must be its own inverse");
}

#[test]
fn pair_swap_preserves_support_and_count() {
    let m = BasisString::<3>::from_positions([0usize, 1, 5, 64, 130]);
    let s = m.pair_swap();
    assert_eq!(s.count(), m.count());
    assert_eq!(s.support(), m.support());
}

#[test]
fn equal_basis_strings_hash_equally_and_differ_otherwise() {
    let a = BasisString::<2>::from_positions([1usize, 70]);
    let b = BasisString::<2>::from_positions([1usize, 70]);
    let c = BasisString::<2>::from_positions([1usize, 71]);
    assert_eq!(a, b);
    assert_eq!(a.hash_value(), b.hash_value());
    assert_ne!(a, c);
    assert_ne!(a.hash_value(), c.hash_value());
}

#[test]
fn single_word_hash_path_is_still_injective_on_a_sample() {
    let mut seen = std::collections::HashSet::new();
    for i in 0..4096u64 {
        assert!(
            seen.insert(BasisString::<1>::from_words([i]).hash_value()),
            "collision at {i}"
        );
    }
}

#[test]
fn bit_ops_match_a_word_level_reference() {
    let a = BasisString::<2>::from_words([0xF0F0, 0x00FF]);
    let b = BasisString::<2>::from_words([0x0FF0, 0xFF00]);
    assert_eq!((a ^ b).words(), &[0xF0F0 ^ 0x0FF0, 0x00FF ^ 0xFF00]);
    assert_eq!((a & b).words(), &[0xF0F0 & 0x0FF0, 0x00FF & 0xFF00]);
    assert_eq!((a | b).words(), &[0xF0F0 | 0x0FF0, 0x00FF | 0xFF00]);
}

#[test]
#[allow(clippy::op_ref)]
fn reference_bit_ops_agree_with_by_value_ops() {
    let a = BasisString::<2>::from_positions([1usize, 70]);
    let b = BasisString::<2>::from_positions([2usize, 70]);
    assert_eq!(&a ^ &b, a ^ b);
    assert_eq!(&a & &b, a & b);
    assert_eq!(&a | &b, a | b);
}
