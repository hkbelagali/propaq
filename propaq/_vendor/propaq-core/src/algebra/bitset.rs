//! A simple bitset implementation using a `SmallVec` of `u64` words.
//!
//! This structure is used for the symplectic representation of
//! basis strings.
//!

use smallvec::SmallVec;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};

type Words = SmallVec<[u64; 4]>;

#[derive(Clone, Debug, Default)]
pub struct Bitset {
    words: Words,
}

impl Bitset {
    pub fn zero() -> Self {
        Self {
            words: Words::new(),
        }
    }

    pub fn from_words(words: impl Into<Words>) -> Self {
        let mut b = Self {
            words: words.into(),
        };
        b.normalize();
        b
    }

    /// Same as `from_words`, but copies directly from a borrowed slice
    /// into an inline `SmallVec`.
    pub fn from_slice(words: &[u64]) -> Self {
        let mut b = Self {
            words: Words::from_slice(words),
        };
        b.normalize();
        b
    }

    /// Create a bitset from a little-endian byte slice.
    pub fn from_le_bytes(bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::zero();
        }
        let n_words = bytes.len().div_ceil(8);
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

    /// Write a bitset to a little-endian byte vector.
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

    /// Get Bitset[pos].
    pub fn bit(&self, pos: usize) -> u64 {
        let wi = pos / 64;
        let bi = pos % 64;
        if wi >= self.words.len() {
            0
        } else {
            (self.words[wi] >> bi) & 1
        }
    }

    /// Read-only access to the underlying words
    pub fn as_words(&self) -> &[u64] {
        &self.words
    }

    /// Bits set at positions 0 through n-1.
    pub fn all_ones_upto(n: usize) -> Self {
        if n == 0 {
            return Self::zero();
        }
        let n_words = n.div_ceil(64);
        let mut words: Words = Words::from_elem(!0u64, n_words);
        let rem = n % 64;
        if rem != 0 {
            *words.last_mut().unwrap() = (1u64 << rem) - 1;
        }
        Self::from_words(words)
    }

    /// Left-shift by `shift` bit positions.
    pub fn shl(&self, shift: usize) -> Self {
        if shift == 0 {
            return self.clone();
        }
        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let new_len = self.words.len() + word_shift + if bit_shift > 0 { 1 } else { 0 };
        let mut words: Words = Words::from_elem(0u64, new_len);
        for (i, &w) in self.words.iter().enumerate() {
            let lo = if bit_shift > 0 { w << bit_shift } else { w };
            let hi = if bit_shift > 0 {
                w >> (64 - bit_shift)
            } else {
                0
            };
            words[i + word_shift] |= lo;
            if hi != 0 && i + word_shift + 1 < words.len() {
                words[i + word_shift + 1] |= hi;
            }
        }
        Self::from_words(words)
    }

    /// Trim trailing zero words from the bitset.
    fn normalize(&mut self) {
        while self.words.last() == Some(&0) {
            self.words.pop();
        }
    }
}

/// Bitwise AND for two bitsets
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

/// Bitwise XOR for two bitsets
impl BitXor for &Bitset {
    type Output = Bitset;
    fn bitxor(self, rhs: &Bitset) -> Bitset {
        let min_len = self.words.len().min(rhs.words.len());
        let max_len = self.words.len().max(rhs.words.len());
        let mut words = Words::from_elem(0u64, max_len);
        for i in 0..min_len {
            words[i] = self.words[i] ^ rhs.words[i];
        }
        // Tail beyond the shorter operand, XOR with 0 is a copy.
        let longer = if self.words.len() > rhs.words.len() {
            self
        } else {
            rhs
        };
        words[min_len..max_len].copy_from_slice(&longer.words[min_len..max_len]);
        Bitset::from_words(words)
    }
}

/// Bitwise OR for two bitsets
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
        let longer = if self.words.len() > rhs.words.len() {
            self
        } else {
            rhs
        };
        words[min_len..max_len].copy_from_slice(&longer.words[min_len..max_len]);
        Bitset::from_words(words)
    }
}

/// Equality and hashing for bitsets
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
#[path = "../../tests/unit/algebra/bitset.rs"]
mod tests;
