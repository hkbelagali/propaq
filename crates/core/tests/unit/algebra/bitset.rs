use super::*;

#[test]
fn zero_is_empty() {
    let b = Bitset::zero();
    assert!(b.is_zero());
    assert_eq!(b.count_ones(), 0);
}

#[test]
fn from_le_bytes_single_byte() {
    let b = Bitset::from_le_bytes(&[0b1010_1010]);
    assert_eq!(b.count_ones(), 4);
    assert_eq!(b.bit(0), 0);
    assert_eq!(b.bit(1), 1);
    assert_eq!(b.bit(7), 1);
}

#[test]
fn roundtrip_le_bytes() {
    let original = vec![0xABu8, 0xCD, 0xEF];
    let b = Bitset::from_le_bytes(&original);
    assert_eq!(b.to_le_bytes(), original);
}

#[test]
fn bit_access_out_of_range() {
    let b = Bitset::from_le_bytes(&[0xFF]);
    assert_eq!(b.bit(8), 0);
    assert_eq!(b.bit(1000), 0);
}

#[test]
fn shl_basic() {
    let b = Bitset::from_le_bytes(&[0b0011]);
    let s = b.shl(2);
    assert_eq!(s.bit(0), 0);
    assert_eq!(s.bit(1), 0);
    assert_eq!(s.bit(2), 1);
    assert_eq!(s.bit(3), 1);
}

#[test]
fn shl_by_zero_is_clone() {
    let b = Bitset::from_le_bytes(&[0b1010]);
    assert_eq!(b.shl(0), b);
}

#[test]
fn shl_crosses_word_boundary() {
    let b = Bitset::from_le_bytes(&[1]);
    let s = b.shl(64);
    assert_eq!(s.bit(0), 0);
    assert_eq!(s.bit(64), 1);
}

#[test]
fn bitwise_and_overlap() {
    let a = Bitset::from_le_bytes(&[0b1100]);
    let b = Bitset::from_le_bytes(&[0b1010]);
    let c = &a & &b;
    assert_eq!(c.count_ones(), 1);
    assert_eq!(c.bit(3), 1);
    assert_eq!(c.bit(2), 0);
}

#[test]
fn bitwise_and_disjoint_is_zero() {
    let a = Bitset::from_le_bytes(&[0b1100]);
    let b = Bitset::from_le_bytes(&[0b0011]);
    assert!((&a & &b).is_zero());
}

#[test]
fn bitwise_or_union() {
    let a = Bitset::from_le_bytes(&[0b1100]);
    let b = Bitset::from_le_bytes(&[0b0011]);
    let c = &a | &b;
    assert_eq!(c.count_ones(), 4);
    assert_eq!(c, Bitset::from_le_bytes(&[0b1111]));
}

#[test]
fn bitwise_xor_symmetric_difference() {
    let a = Bitset::from_le_bytes(&[0b1100]);
    let b = Bitset::from_le_bytes(&[0b1010]);
    let c = &a ^ &b;
    assert_eq!(c.bit(1), 1);
    assert_eq!(c.bit(2), 1);
    assert_eq!(c.bit(3), 0);
    assert_eq!(c.count_ones(), 2);
}

#[test]
fn xor_self_is_zero() {
    let a = Bitset::from_le_bytes(&[0b1010_1010, 0b1100_1100]);
    assert!((&a ^ &a).is_zero());
}

#[test]
fn equality_normalized() {
    let a = Bitset::from_le_bytes(&[0b1010]);
    let b = Bitset::from_le_bytes(&[0b1010]);
    let c = Bitset::from_le_bytes(&[0b0101]);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn all_ones_upto() {
    let b = Bitset::all_ones_upto(4);
    assert_eq!(b.count_ones(), 4);
    for i in 0..4 {
        assert_eq!(b.bit(i), 1, "bit {i} should be 1");
    }
    assert_eq!(b.bit(4), 0);
}

#[test]
fn all_ones_upto_zero_is_empty() {
    assert!(Bitset::all_ones_upto(0).is_zero());
}

#[test]
fn multiword_count_ones() {
    let b = Bitset::from_words(vec![u64::MAX, 1]);
    assert_eq!(b.count_ones(), 65);
}

#[test]
fn multiword_bit_access() {
    let b = Bitset::from_words(vec![0u64, 1u64 << 5]);
    assert_eq!(b.bit(64), 0);
    assert_eq!(b.bit(69), 1);
}

#[test]
fn ord_zero_less_than_nonzero() {
    let a = Bitset::zero();
    let b = Bitset::from_le_bytes(&[1]);
    assert!(a < b);
}

#[test]
fn ord_equal() {
    let a = Bitset::from_le_bytes(&[0b1010]);
    let b = Bitset::from_le_bytes(&[0b1010]);
    assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
}

#[test]
fn ord_higher_word_dominates() {
    let a = Bitset::from_words(vec![u64::MAX, 0]);
    let b = Bitset::from_words(vec![0u64, 1]);
    assert!(a < b);
}
