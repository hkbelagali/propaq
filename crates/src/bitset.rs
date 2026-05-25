//! This module defines a flexible length bitset, represented as a vector of 64-bit words. It supports basic bitwise operations, counting set bits, and shifting.
//! This is needed to store an arbitrary number of Majorana modes
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};
use smallvec::SmallVec;

// 4 inline words = 256 Majorana modes (128 qubits) without heap allocation.
// Larger systems spill to heap automatically.
type Words = SmallVec<[u64; 4]>;

#[derive(Clone, Debug, Default)]
pub struct Bitset {
    words: Words,
}

impl Bitset {
    /// Generate an empty bitset
    pub fn zero() -> Self {
        Self { words: Words::new() }
    }

    /// Build the bitset from a vector of 64-bit words, removing any trailing zeros.
    pub fn from_words(words: impl Into<Words>) -> Self {
        let mut b = Self { words: words.into() };
        b.normalize();
        b
    }

    /// Build the bitset from a little-endian byte array
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

    /// Write the bitset to a little-endian byte array
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

    /// Check if a bitset is empty (all bits zero) 
    pub fn is_zero(&self) -> bool {
        self.words.is_empty()
    }

    /// Count the number of set bits in the bitset
    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    /// Get the value of the bit at position 'pos'
    pub fn bit(&self, pos: usize) -> u64 {
        let wi = pos / 64;
        let bi = pos % 64;
        if wi >= self.words.len() { 0 } else { (self.words[wi] >> bi) & 1 }
    }

    /// Returns the position of the lowest set bit, or `usize::MAX` if zero.
    pub fn trailing_zeros(&self) -> usize {
        for (i, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return i * 64 + w.trailing_zeros() as usize;
            }
        }
        usize::MAX
    }

    /// Clear the bit at position 'pos' 
    pub fn clear_bit(&mut self, pos: usize) {
        let wi = pos / 64;
        let bi = pos % 64;
        if wi < self.words.len() {
            self.words[wi] &= !(1u64 << bi);
        }
        self.normalize();
    }

    /// Count set bits at positions strictly above `pos`.
    pub fn count_ones_above(&self, pos: usize) -> u64 {
        let wi = pos / 64;
        let bi = pos % 64;
        let mut count = 0u64;
        if wi < self.words.len() {
            if bi < 63 {
                count += (self.words[wi] >> (bi + 1)).count_ones() as u64;
            }
            for i in (wi + 1)..self.words.len() {
                count += self.words[i].count_ones() as u64;
            }
        }
        count
    }

    /// Shift the bitset right by "shift" bit positions, padding with zeros
    pub fn shr(&self, shift: usize) -> Self {
        if shift == 0 {
            return self.clone();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        if word_shift >= self.words.len() {
            return Self::zero();
        }
        let mut words = Words::with_capacity(self.words.len() - word_shift);
        for i in word_shift..self.words.len() {
            let lo = if bit_shift > 0 { self.words[i] >> bit_shift } else { self.words[i] };
            let hi = if bit_shift > 0 && i + 1 < self.words.len() {
                self.words[i + 1] << (64 - bit_shift)
            } else {
                0
            };
            words.push(lo | hi);
        }
        Self::from_words(words)
    }

    /// Get the word at index i
    fn word(&self, i: usize) -> u64 {
        self.words.get(i).copied().unwrap_or(0)
    }

    /// Bits set at positions 0, 2, 4, … strictly below n (even indices only).
    pub fn even_mask_upto(n: usize) -> Self {
        if n == 0 { return Self::zero(); }
        let n_words = (n + 63) / 64;
        let mut words = Words::with_capacity(n_words);
        for w in 0..n_words {
            let base = w * 64;
            let bits_in_word = n.saturating_sub(base).min(64);
            let full_even = 0x5555_5555_5555_5555u64;
            let word = if bits_in_word == 64 {
                full_even
            } else {
                full_even & ((1u64 << bits_in_word).wrapping_sub(1))
            };
            words.push(word);
        }
        Self::from_words(words)
    }

    /// Bits set at positions 0 through n-1 (dense prefix mask).
    pub fn all_ones_upto(n: usize) -> Self {
        if n == 0 { return Self::zero(); }
        let n_words = (n + 63) / 64;
        let mut words: Words = std::iter::repeat(!0u64).take(n_words).collect();
        let rem = n % 64;
        if rem != 0 { *words.last_mut().unwrap() = (1u64 << rem) - 1; }
        Self::from_words(words)
    }

    /// Left-shift by `shift` bit positions (mirrors the existing `shr`).
    pub fn shl(&self, shift: usize) -> Self {
        if shift == 0 { return self.clone(); }
        let word_shift = shift / 64;
        let bit_shift  = shift % 64;
        let new_len = self.words.len() + word_shift + if bit_shift > 0 { 1 } else { 0 };
        let mut words: Words = std::iter::repeat(0u64).take(new_len).collect();
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

/// Implement logical bitwise operations for references to Bitset
impl BitAnd for &Bitset {
    type Output = Bitset;
    fn bitand(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().min(rhs.words.len());
        let words: Words = (0..len).map(|i| self.words[i] & rhs.words[i]).collect();
        Bitset::from_words(words)
    }
}

impl BitXor for &Bitset {
    type Output = Bitset;
    fn bitxor(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().max(rhs.words.len());
        let words: Words = (0..len).map(|i| self.word(i) ^ rhs.word(i)).collect();
        Bitset::from_words(words)
    }
}

impl BitOr for &Bitset {
    type Output = Bitset;
    fn bitor(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().max(rhs.words.len());
        let words: Words = (0..len).map(|i| self.word(i) | rhs.word(i)).collect();
        Bitset::from_words(words)
    }
}

/// Implement equality and hashing
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_empty() {
        let b = Bitset::zero();
        assert!(b.is_zero());
        assert_eq!(b.count_ones(), 0);
        assert_eq!(b.trailing_zeros(), usize::MAX);
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
    fn trailing_zeros_single_bit() {
        let b = Bitset::from_le_bytes(&[0b0010_0000]);
        assert_eq!(b.trailing_zeros(), 5);
    }

    #[test]
    fn trailing_zeros_lowest_bit() {
        let b = Bitset::from_le_bytes(&[0b1111_0000]);
        assert_eq!(b.trailing_zeros(), 4);
    }

    #[test]
    fn count_ones_above_dense_low_nibble() {
        let b = Bitset::from_le_bytes(&[0b1111]);
        assert_eq!(b.count_ones_above(0), 3);
        assert_eq!(b.count_ones_above(1), 2);
        assert_eq!(b.count_ones_above(2), 1);
        assert_eq!(b.count_ones_above(3), 0);
        assert_eq!(b.count_ones_above(4), 0);
    }

    #[test]
    fn count_ones_above_last_bit_in_word() {
        let b = Bitset::from_words(vec![1u64 << 63, 1]);
        assert_eq!(b.count_ones_above(63), 1); // bit 64 is above 63
        assert_eq!(b.count_ones_above(64), 0);
    }

    #[test]
    fn shr_basic() {
        let b = Bitset::from_le_bytes(&[0b1100]);
        let s = b.shr(2);
        assert_eq!(s.bit(0), 1);
        assert_eq!(s.bit(1), 1);
        assert_eq!(s.bit(2), 0);
    }

    #[test]
    fn shr_by_zero_is_clone() {
        let b = Bitset::from_le_bytes(&[0b1010]);
        assert_eq!(b.shr(0), b);
    }

    #[test]
    fn shr_past_end_is_zero() {
        let b = Bitset::from_le_bytes(&[0b1111]);
        assert!(b.shr(100).is_zero());
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
    fn shl_shr_roundtrip() {
        let b = Bitset::from_le_bytes(&[0b0011]);
        assert_eq!(b.shl(3).shr(3), b);
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
        // a = bits {2,3}, b = bits {1,3}; XOR = {1,2}
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
    fn even_mask_upto() {
        let b = Bitset::even_mask_upto(8);
        for i in [0usize, 2, 4, 6] { assert_eq!(b.bit(i), 1, "bit {i} should be 1"); }
        for i in [1usize, 3, 5, 7] { assert_eq!(b.bit(i), 0, "bit {i} should be 0"); }
    }

    #[test]
    fn clear_bit_and_normalize() {
        let mut b = Bitset::from_le_bytes(&[0b1111]);
        b.clear_bit(2);
        assert_eq!(b.bit(2), 0);
        assert_eq!(b.count_ones(), 3);
        b.clear_bit(0);
        b.clear_bit(1);
        b.clear_bit(3);
        assert!(b.is_zero());
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
    fn multiword_trailing_zeros_second_word() {
        let b = Bitset::from_words(vec![0u64, 0b100u64]);
        assert_eq!(b.trailing_zeros(), 66);
    }
}
