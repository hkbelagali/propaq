//! This module defines a flexible length bitset, represented as a vector of 64-bit words. It supports basic bitwise operations, counting set bits, and shifting. 
//! This is needed to store an arbitrary number of Majorana modes
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};

#[derive(Clone, Debug, Default)]
pub struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    /// Generate an empty bitset
    pub fn zero() -> Self {
        Self { words: vec![] }
    }

    /// Build the bitset from a vector of 64-bit words, removing any trailing zeros.
    pub fn from_words(words: Vec<u64>) -> Self {
        let mut b = Self { words };
        b.normalize();
        b
    }

    /// Build the bitset from a little-endian byte array
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }
        let n_words = (bytes.len() + 7) / 8;
        let mut words = Vec::with_capacity(n_words);
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
        let mut words = Vec::with_capacity(self.words.len() - word_shift);
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
        let mut words = Vec::with_capacity(n_words);
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
        let mut words = vec![!0u64; n_words];
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
        let mut words = vec![0u64; new_len];
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
        let words: Vec<u64> = (0..len).map(|i| self.words[i] & rhs.words[i]).collect();
        Bitset::from_words(words)
    }
}

impl BitXor for &Bitset {
    type Output = Bitset;
    fn bitxor(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().max(rhs.words.len());
        let words: Vec<u64> = (0..len).map(|i| self.word(i) ^ rhs.word(i)).collect();
        Bitset::from_words(words)
    }
}

impl BitOr for &Bitset {
    type Output = Bitset;
    fn bitor(self, rhs: &Bitset) -> Bitset {
        let len = self.words.len().max(rhs.words.len());
        let words: Vec<u64> = (0..len).map(|i| self.word(i) | rhs.word(i)).collect();
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
