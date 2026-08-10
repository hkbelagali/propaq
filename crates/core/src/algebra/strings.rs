//!
//! Generic string representation for a basis operator belonging to
//! some algebra, such as a Pauli string or Majorana monomial.
//! We store the operators in their symplectic representations.
//!
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};

/// Number of bits in one storage word.
const WORD_BITS: usize = 64;

/// One basis operator, as `W` words of interleaved unit pairs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct BasisString<const W: usize> {
    words: [u64; W],
}

impl<const W: usize> Default for BasisString<W> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const W: usize> BasisString<W> {
    /// The identity operator, with no bit set.
    #[inline]
    pub const fn zero() -> Self {
        BasisString { words: [0u64; W] }
    }

    /// Builds a basis string from raw storage words.
    #[inline]
    pub const fn from_words(words: [u64; W]) -> Self {
        BasisString { words }
    }

    /// The raw storage words.
    #[inline]
    pub const fn words(&self) -> &[u64; W] {
        &self.words
    }

    /// Number of storage words, known at compile time.
    #[inline]
    pub const fn num_words() -> usize {
        W
    }

    /// Number of addressable bit positions, known at compile time.
    #[inline]
    pub const fn num_bits() -> usize {
        W * WORD_BITS
    }

    /// The smallest `W` that can hold `n_units` units at two bits each.
    #[inline]
    pub const fn words_for(n_units: usize) -> usize {
        (2 * n_units).div_ceil(WORD_BITS)
    }

    /// True if bit `pos` is set.
    #[inline]
    pub fn test(&self, pos: usize) -> bool {
        debug_assert!(pos < Self::num_bits(), "bit position out of range");
        (self.words[pos / WORD_BITS] >> (pos % WORD_BITS)) & 1 != 0
    }

    /// Sets bit `pos`.
    #[inline]
    pub fn set(&mut self, pos: usize) {
        debug_assert!(pos < Self::num_bits(), "bit position out of range");
        self.words[pos / WORD_BITS] |= 1u64 << (pos % WORD_BITS);
    }

    /// Clears bit `pos`.
    #[inline]
    pub fn clear(&mut self, pos: usize) {
        debug_assert!(pos < Self::num_bits(), "bit position out of range");
        self.words[pos / WORD_BITS] &= !(1u64 << (pos % WORD_BITS));
    }

    /// True if no bit is set.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Number of set bits.
    #[inline]
    pub fn count(&self) -> usize {
        let mut n = 0u32;
        for i in 0..W {
            n += self.words[i].count_ones();
        }
        n as usize
    }

    /// Number of bits set in both basis strings, without materializing the AND.
    #[inline]
    pub fn count_and(&self, other: &Self) -> usize {
        let mut n = 0u32;
        for i in 0..W {
            n += (self.words[i] & other.words[i]).count_ones();
        }
        n as usize
    }

    #[inline]
    pub fn parity_and(&self, other: &Self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc ^= self.words[i] & other.words[i];
        }
        acc.count_ones() % 2 == 1
    }

    /// Number of units with at least one of their two bits set.
    /// This is the Pauli weight of the string/JW-mapped monomial.
    #[inline]
    pub fn support(&self) -> usize {
        let mut n = 0u32;
        for i in 0..W {
            let w = self.words[i];
            // Fold each unit's two bits onto its even bit, then count those.
            n += ((w | (w >> 1)) & Self::EVEN_MASK).count_ones();
        }
        n as usize
    }

    /// Mask selecting the even bit of every unit pair.
    const EVEN_MASK: u64 = 0x5555_5555_5555_5555;

    /// Swaps the two bits of every unit pair.
    #[inline]
    pub fn pair_swap(&self) -> Self {
        let mut out = [0u64; W];
        for (i, slot) in out.iter_mut().enumerate().take(W) {
            let w = self.words[i];
            *slot = ((w & Self::EVEN_MASK) << 1) | ((w >> 1) & Self::EVEN_MASK);
        }
        BasisString { words: out }
    }

    /// Ascending positions of the set bits.
    #[inline]
    pub fn positions(&self) -> Positions<'_, W> {
        Positions {
            words: &self.words,
            word: 0,
            rest: if W > 0 { self.words[0] } else { 0 },
        }
    }

    /// Rebuilds a basis string from ascending set-bit positions.
    pub fn from_positions(positions: impl IntoIterator<Item = usize>) -> Self {
        let mut m = Self::zero();
        for p in positions {
            m.set(p);
        }
        m
    }

    /// A 64-bit hash of the whole basis string.
    #[inline]
    pub fn hash_value(&self) -> u64 {
        if W == 1 {
            return splitmix64(self.words[0]);
        }
        let mut h = rustc_hash::FxHasher::default();
        self.words.hash(&mut h);
        h.finish()
    }
}

/// Finalizer from splitmix64, used to avalanche a single storage word.
#[inline]
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl<const W: usize> Hash for BasisString<W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words.hash(state);
    }
}

/// Iterator over a basis string's ascending set-bit positions.
pub struct Positions<'a, const W: usize> {
    words: &'a [u64; W],
    word: usize,
    rest: u64,
}

impl<const W: usize> Iterator for Positions<'_, W> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.rest != 0 {
                let bit = self.rest.trailing_zeros() as usize;
                self.rest &= self.rest - 1;
                return Some(self.word * WORD_BITS + bit);
            }
            self.word += 1;
            if self.word >= W {
                return None;
            }
            self.rest = self.words[self.word];
        }
    }
}

macro_rules! impl_bitop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const W: usize> $trait for BasisString<W> {
            type Output = BasisString<W>;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                let mut out = [0u64; W];
                for i in 0..W {
                    out[i] = self.words[i] $op rhs.words[i];
                }
                BasisString { words: out }
            }
        }
        impl<const W: usize> $trait<&BasisString<W>> for &BasisString<W> {
            type Output = BasisString<W>;
            #[inline]
            fn $method(self, rhs: &BasisString<W>) -> Self::Output {
                let mut out = [0u64; W];
                for i in 0..W {
                    out[i] = self.words[i] $op rhs.words[i];
                }
                BasisString { words: out }
            }
        }
    };
}

impl_bitop!(BitXor, bitxor, ^);
impl_bitop!(BitAnd, bitand, &);
impl_bitop!(BitOr, bitor, |);

#[cfg(test)]
#[path = "../../tests/unit/algebra/strings.rs"]
mod tests;
