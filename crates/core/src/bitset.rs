///
/// A simple bitset implementation using a `SmallVec` of `u64` words. 
/// Pauli and Majorana strings can be compactly represented using 
/// this structure, and it supports basic bitwise operations and lexicographic ordering.
///

use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};
use std::cmp::Ordering;
use smallvec::SmallVec;

type Words = SmallVec<[u64; 4]>;

#[derive(Clone, Debug, Default)]
pub struct Bitset {
    words: Words,
}

impl Bitset {
    pub fn zero() -> Self {
        Self { words: Words::new() }
    }

    pub fn from_words(words: impl Into<Words>) -> Self {
        let mut b = Self { words: words.into() };
        b.normalize();
        b
    }

    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }
        let n_words = (bytes.len() + 7) / 8;
        let mut words = Words::with_capacity(n_words);
        let mut i = 0;
        while i < bytes.len() {
            let end = (i + 8).min(bytes.len());
            let mut buf = [0u8; 8];
            buf[..end - i].copy_from_slice(&bytes[i..end]);
            words.push(u64::from_le_bytes(buf));
            i += 8;
        }
        let mut b = Self { words };
        b.normalize();
        b
    }

    pub fn to_le_bytes(&self) -> Vec<u8> {
        if self.words.is_empty() {
            return vec![];
        }
        let mut bytes: Vec<u8> = self.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        bytes
    }

    pub fn is_zero(&self) -> bool {
        self.words.is_empty()
    }

    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn bit(&self, pos: usize) -> u64 {
        let wi = pos / 64;
        let bi = pos % 64;
        if wi >= self.words.len() { 0 } else { (self.words[wi] >> bi) & 1 }
    }

    /// Read-only access to the underlying words, used for optimised algorithms.
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Word at index `i`, zero if `i` is beyond this value's own (possibly
    /// `normalize()`-shortened) length — consistent with the canonical
    /// representation treating missing high words as zero. Used by
    /// batch-gather code that needs a fixed, system-wide word count
    /// regardless of any individual value's own length.
    pub fn word_at(&self, i: usize) -> u64 {
        self.words.get(i).copied().unwrap_or(0)
    }

    /// Bits set at positions 0 through n-1 (dense prefix mask).
    pub fn all_ones_upto(n: usize) -> Self {
        if n == 0 { return Self::zero(); }
        let n_words = (n + 63) / 64;
        let mut words: Words = Words::from_elem(!0u64, n_words);
        let rem = n % 64;
        if rem != 0 { *words.last_mut().unwrap() = (1u64 << rem) - 1; }
        Self::from_words(words)
    }

    /// Left-shift by `shift` bit positions.
    pub fn shl(&self, shift: usize) -> Self {
        if shift == 0 { return self.clone(); }
        let word_shift = shift / 64;
        let bit_shift  = shift % 64;
        let new_len = self.words.len() + word_shift + if bit_shift > 0 { 1 } else { 0 };
        let mut words: Words = Words::from_elem(0u64, new_len);
        for (i, &w) in self.words.iter().enumerate() {
            let lo = if bit_shift > 0 { w << bit_shift } else { w };
            let hi = if bit_shift > 0 { w >> (64 - bit_shift) } else { 0 };
            words[i + word_shift] |= lo;
            if hi != 0 && i + word_shift + 1 < words.len() {
                words[i + word_shift + 1] |= hi;
            }
        }
        Self::from_words(words)
    }

    fn normalize(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
}

impl BitAnd for &Bitset {
    type Output = Bitset;
    fn bitand(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().min(rhs.words.len());
        let mut words = Words::from_elem(0u64, len);
        for i in 0..len {
            words[i] = self.words[i] & rhs.words[i];
        }
        Bitset::from_words(words)
    }
}

impl BitXor for &Bitset {
    type Output = Bitset;
    fn bitxor(self, rhs: &Bitset) -> Bitset {
        let min_len = self.words.len().min(rhs.words.len());
        let max_len = self.words.len().max(rhs.words.len());
        let mut words = Words::from_elem(0u64, max_len);
        for i in 0..min_len {
            words[i] = self.words[i] ^ rhs.words[i];
        }
        // Tail beyond the shorter operand: XOR with 0 is a copy.
        let longer = if self.words.len() > rhs.words.len() { self } else { rhs };
        words[min_len..max_len].copy_from_slice(&longer.words[min_len..max_len]);
        Bitset::from_words(words)
    }
}

impl BitOr for &Bitset {
    type Output = Bitset;
    fn bitor(self, rhs: &Bitset) -> Bitset {
        let min_len = self.words.len().min(rhs.words.len());
        let max_len = self.words.len().max(rhs.words.len());
        let mut words = Words::from_elem(0u64, max_len);
        for i in 0..min_len {
            words[i] = self.words[i] | rhs.words[i];
        }
        // Tail beyond the shorter operand: OR with 0 is a copy.
        let longer = if self.words.len() > rhs.words.len() { self } else { rhs };
        words[min_len..max_len].copy_from_slice(&longer.words[min_len..max_len]);
        Bitset::from_words(words)
    }
}

impl PartialEq for Bitset {
    fn eq(&self, other: &Self) -> bool {
        self.words == other.words
    }
}

impl Eq for Bitset {}

impl Hash for Bitset {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words.hash(state);
    }
}

/// Lexicographic ordering on words from most-significant to least-significant.
impl PartialOrd for Bitset {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bitset {
    fn cmp(&self, other: &Self) -> Ordering {
        let len = self.words.len().max(other.words.len());
        for i in (0..len).rev() {
            let a = self.words.get(i).copied().unwrap_or(0);
            let b = other.words.get(i).copied().unwrap_or(0);
            match a.cmp(&b) {
                Ordering::Equal => {}
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
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
        for i in 0..4 { assert_eq!(b.bit(i), 1, "bit {i} should be 1"); }
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
}
