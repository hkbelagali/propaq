///
/// A single basis operator as one compile-time-sized interleaved bitset.
///
/// Two bits per unit: bit `2k` and bit `2k + 1` describe unit `k`. Read as a
/// Majorana product, bit `2k` is the even mode and bit `2k + 1` the odd mode of
/// site `k`. Read as a Pauli string, the pair is the Jordan-Wigner image, with
/// `2k` the X component and `2k + 1` the Z component of qubit `k`.
///
/// This replaces the two separate word planes the SoA store used. One bitset
/// means a product is a single XOR over a fixed-length array whose length the
/// compiler knows, rather than two dynamically bounded loops.
///
/// `W` is the number of `u64` words, not the unit count: Rust cannot compute an
/// array length from a const parameter on stable, so callers derive
/// `W = ceil(2 * n_units / 64)` and select the instantiation. Bits at or above
/// `2 * n_units` are never set by any operation here, so the unused tail of the
/// final word stays clear without an explicit sanitize step.
///
use std::hash::{Hash, Hasher};
use std::ops::{BitAnd, BitOr, BitXor};

/// Number of bits in one storage word.
const WORD_BITS: usize = 64;

/// One basis operator, as `W` words of interleaved unit pairs.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Monomial<const W: usize> {
    words: [u64; W],
}

impl<const W: usize> Default for Monomial<W> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const W: usize> Monomial<W> {
    /// The identity operator, with no bit set.
    #[inline]
    pub const fn zero() -> Self {
        Monomial { words: [0u64; W] }
    }

    /// Builds a monomial from raw storage words.
    #[inline]
    pub const fn from_words(words: [u64; W]) -> Self {
        Monomial { words }
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

    /// Number of bits set in both monomials, without materializing the AND.
    #[inline]
    pub fn count_and(&self, other: &Self) -> usize {
        let mut n = 0u32;
        for i in 0..W {
            n += (self.words[i] & other.words[i]).count_ones();
        }
        n as usize
    }

    /// Parity of [`Monomial::count_and`], which is the value the rotation sign
    /// needs and is cheaper than the count when only the low bit matters.
    #[inline]
    pub fn parity_and(&self, other: &Self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc ^= self.words[i] & other.words[i];
        }
        acc.count_ones() % 2 == 1
    }

    /// Number of units with at least one of their two bits set.
    ///
    /// This is the Pauli weight, and the count of sites a Majorana monomial
    /// touches.
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
    ///
    /// For Pauli this is the `J(G)` map that turns a generator into its
    /// anticommutation fold columns, since X and Z exchange roles there.
    #[inline]
    pub fn pair_swap(&self) -> Self {
        let mut out = [0u64; W];
        for i in 0..W {
            let w = self.words[i];
            out[i] = ((w & Self::EVEN_MASK) << 1) | ((w >> 1) & Self::EVEN_MASK);
        }
        Monomial { words: out }
    }

    /// Ascending positions of the set bits.
    #[inline]
    pub fn positions(&self) -> Positions<'_, W> {
        Positions { words: &self.words, word: 0, rest: if W > 0 { self.words[0] } else { 0 } }
    }

    /// Rebuilds a monomial from ascending set-bit positions.
    pub fn from_positions(positions: impl IntoIterator<Item = usize>) -> Self {
        let mut m = Self::zero();
        for p in positions {
            m.set(p);
        }
        m
    }

    /// A 64-bit hash of the whole monomial.
    ///
    /// Single-word monomials skip the generic hasher and mix the one word
    /// directly, which is the common case at benchmark widths.
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

impl<const W: usize> Hash for Monomial<W> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.words.hash(state);
    }
}

/// Iterator over a monomial's ascending set-bit positions.
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
        impl<const W: usize> $trait for Monomial<W> {
            type Output = Monomial<W>;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                let mut out = [0u64; W];
                for i in 0..W {
                    out[i] = self.words[i] $op rhs.words[i];
                }
                Monomial { words: out }
            }
        }
        impl<const W: usize> $trait<&Monomial<W>> for &Monomial<W> {
            type Output = Monomial<W>;
            #[inline]
            fn $method(self, rhs: &Monomial<W>) -> Self::Output {
                let mut out = [0u64; W];
                for i in 0..W {
                    out[i] = self.words[i] $op rhs.words[i];
                }
                Monomial { words: out }
            }
        }
    };
}

impl_bitop!(BitXor, bitxor, ^);
impl_bitop!(BitAnd, bitand, &);
impl_bitop!(BitOr, bitor, |);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_empty() {
        let m = Monomial::<2>::zero();
        assert!(m.is_zero());
        assert_eq!(m.count(), 0);
        assert_eq!(m.positions().count(), 0);
    }

    #[test]
    fn words_for_rounds_up_to_two_bits_per_unit() {
        assert_eq!(Monomial::<1>::words_for(0), 0);
        assert_eq!(Monomial::<1>::words_for(1), 1);
        assert_eq!(Monomial::<1>::words_for(32), 1);
        assert_eq!(Monomial::<1>::words_for(33), 2);
        assert_eq!(Monomial::<1>::words_for(36), 2);
        assert_eq!(Monomial::<1>::words_for(64), 2);
    }

    #[test]
    fn set_test_clear_round_trip() {
        let mut m = Monomial::<2>::zero();
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
        let m = Monomial::<3>::from_positions(want);
        let got: Vec<usize> = m.positions().collect();
        assert_eq!(got, want);
        assert!(got.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn xor_is_the_product_key_and_is_involutive() {
        let a = Monomial::<2>::from_positions([1usize, 5, 70]);
        let b = Monomial::<2>::from_positions([5usize, 9, 70]);
        let c = a ^ b;
        assert_eq!(c.positions().collect::<Vec<_>>(), vec![1, 9]);
        assert_eq!(c ^ b, a, "xor must be involutive in the second operand");
    }

    #[test]
    fn count_and_and_parity_agree_with_a_naive_reference() {
        let a = Monomial::<2>::from_positions([1usize, 5, 9, 70]);
        let b = Monomial::<2>::from_positions([5usize, 9, 70, 100]);
        let naive = (a & b).count();
        assert_eq!(a.count_and(&b), naive);
        assert_eq!(a.parity_and(&b), naive % 2 == 1);
    }

    #[test]
    fn support_counts_units_not_bits() {
        // Unit 0 has both bits set, unit 3 has one. Support is 2, count is 3.
        let m = Monomial::<1>::from_positions([0usize, 1, 6]);
        assert_eq!(m.count(), 3);
        assert_eq!(m.support(), 2);
    }

    #[test]
    fn support_spans_word_boundaries() {
        // Unit 31 occupies bits 62 and 63; unit 32 occupies bits 64 and 65.
        let m = Monomial::<2>::from_positions([62usize, 63, 64]);
        assert_eq!(m.support(), 2);
    }

    #[test]
    fn pair_swap_exchanges_each_units_two_bits() {
        let m = Monomial::<2>::from_positions([0usize, 3, 64]);
        let s = m.pair_swap();
        assert_eq!(s.positions().collect::<Vec<_>>(), vec![1, 2, 65]);
        assert_eq!(s.pair_swap(), m, "pair_swap must be its own inverse");
    }

    #[test]
    fn pair_swap_preserves_support_and_count() {
        let m = Monomial::<3>::from_positions([0usize, 1, 5, 64, 130]);
        let s = m.pair_swap();
        assert_eq!(s.count(), m.count());
        assert_eq!(s.support(), m.support());
    }

    #[test]
    fn equal_monomials_hash_equally_and_differ_otherwise() {
        let a = Monomial::<2>::from_positions([1usize, 70]);
        let b = Monomial::<2>::from_positions([1usize, 70]);
        let c = Monomial::<2>::from_positions([1usize, 71]);
        assert_eq!(a, b);
        assert_eq!(a.hash_value(), b.hash_value());
        assert_ne!(a, c);
        assert_ne!(a.hash_value(), c.hash_value());
    }

    #[test]
    fn single_word_hash_path_is_still_injective_on_a_sample() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..4096u64 {
            assert!(seen.insert(Monomial::<1>::from_words([i]).hash_value()), "collision at {i}");
        }
    }

    #[test]
    fn bit_ops_match_a_word_level_reference() {
        let a = Monomial::<2>::from_words([0xF0F0, 0x00FF]);
        let b = Monomial::<2>::from_words([0x0FF0, 0xFF00]);
        assert_eq!((a ^ b).words(), &[0xF0F0 ^ 0x0FF0, 0x00FF ^ 0xFF00]);
        assert_eq!((a & b).words(), &[0xF0F0 & 0x0FF0, 0x00FF & 0xFF00]);
        assert_eq!((a | b).words(), &[0xF0F0 | 0x0FF0, 0x00FF | 0xFF00]);
    }

    #[test]
    fn reference_bit_ops_agree_with_by_value_ops() {
        let a = Monomial::<2>::from_positions([1usize, 70]);
        let b = Monomial::<2>::from_positions([2usize, 70]);
        assert_eq!(&a ^ &b, a ^ b);
        assert_eq!(&a & &b, a & b);
        assert_eq!(&a | &b, a | b);
    }
}
