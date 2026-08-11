//! 
//! Implements the Majorana basis and its algebra as an impl of Basis
//! 

use num_complex::Complex64;

use propaq_core::basis::{Basis, BasisKind};
use propaq_core::bitset::Bitset;
use propaq_core::strings::BasisString;

use crate::monomial::{hermiticity_exp, resorting_parity, MajoranaMonomial};

/// Gathers the even-indexed bits of `x` into its low half.
#[inline]
fn gather_even(mut x: u64) -> u64 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    (x | (x >> 16)) & 0x0000_0000_ffff_ffff
}

/// Site-indexed planes: which sites carry $\gamma_{2k}$, and which $\gamma_{2k+1}$.
fn site_planes<const W: usize>(m: &BasisString<W>) -> ([u64; W], [u64; W]) {
    let (mut even, mut odd) = ([0u64; W], [0u64; W]);
    for (w, &word) in m.words().iter().enumerate() {
        let half = 32 * (w % 2);
        even[w / 2] |= gather_even(word) << half;
        odd[w / 2] |= gather_even(word >> 1) << half;
    }
    (even, odd)
}

/// `dst ^= src << shift`, over a fixed-width little-endian bit array.
fn xor_shifted<const W: usize>(dst: &mut [u64; W], src: &[u64; W], shift: usize) {
    let (words, bits) = (shift / 64, shift % 64);
    for i in (words..W).rev() {
        let mut v = src[i - words] << bits;
        if bits > 0 && i > words {
            v |= src[i - words - 1] >> (64 - bits);
        }
        dst[i] ^= v;
    }
}

/// Clears every bit at or above `n`.
fn mask_to<const W: usize>(v: &mut [u64; W], n: usize) {
    for (i, w) in v.iter_mut().enumerate() {
        let low = i * 64;
        *w &= if low >= n {
            0
        } else if n - low >= 64 {
            u64::MAX
        } else {
            (1u64 << (n - low)) - 1
        };
    }
}

fn count<const W: usize>(v: &[u64; W]) -> u32 {
    v.iter().map(|w| w.count_ones()).sum()
}

/// The Jordan-Wigner qubit weight of a Majorana basis string over `n_sites` sites.
fn jw_weight<const W: usize>(m: &BasisString<W>, n_sites: usize) -> u32 {
    if n_sites == 0 {
        return 0;
    }
    let (x_bits, y_bits) = site_planes(m);
    let mut single = [0u64; W];
    let mut occupied = [0u64; W];
    for i in 0..W {
        single[i] = x_bits[i] ^ y_bits[i];
        occupied[i] = x_bits[i] | y_bits[i];
    }

    // p = prefix XOR of `single`, by doubling shifts, masked to the live sites.
    let mut p = single;
    let mut shift = 1usize;
    while shift < n_sites {
        let src = p;
        xor_shifted(&mut p, &src, shift);
        mask_to(&mut p, n_sites);
        shift <<= 1;
    }

    // An odd number of unpaired sites flips the string over the whole register.
    let odd_single = count(&single) & 1 == 1;
    let mut weight = 0u32;
    for i in 0..W {
        let low = i * 64;
        let live = if low >= n_sites {
            0
        } else if n_sites - low >= 64 {
            u64::MAX
        } else {
            (1u64 << (n_sites - low)) - 1
        };
        let string = if odd_single { p[i] ^ live } else { p[i] };
        weight += (single[i] | (occupied[i] ^ string)).count_ones();
    }
    weight
}

/// Per-gate precomputation for a Majorana rotation.
pub struct MajoranaGenContext<const W: usize> {
    gen: BasisString<W>,
    /// `popcount(gen)`, the generator's Majorana length, constant across the gate.
    gen_len: usize,
    /// True when `gen_len` is odd, which is what makes the anticommutation fold
    /// need a per-row parity correction.
    gen_len_odd: bool,
    /// Overall sign carried by the generator, from a Clifford frame conjugation.
    sign: f64,
}

/// The Majorana basis.
pub struct MajoranaAlgebra;

impl<const W: usize> Basis<W> for MajoranaAlgebra {
    const KIND: BasisKind = BasisKind::Majorana;

    type GenContext = MajoranaGenContext<W>;

    #[inline]
    fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> Self::GenContext {
        let gen_len = gen.count();
        MajoranaGenContext {
            gen: *gen,
            gen_len,
            gen_len_odd: gen_len & 1 == 1,
            sign,
        }
    }

    #[inline]
    fn generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.gen
    }

    #[inline]
    fn anticommutes(ctx: &Self::GenContext, string: &BasisString<W>) -> bool {
        // Two Majorana products commute iff `|M||G| + |M & G|` is even.
        let overlap = string.count_and(&ctx.gen);
        let cross = if ctx.gen_len_odd { string.count() } else { 0 };
        (cross + overlap) & 1 == 1
    }

    #[inline]
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.gen
    }

    #[inline]
    fn fold_needs_odd_correction(ctx: &Self::GenContext) -> bool {
        ctx.gen_len_odd
    }

    #[inline]
    fn product(ctx: &Self::GenContext, string: &BasisString<W>) -> (BasisString<W>, Complex64) {
        let out = *string ^ ctx.gen;
        let r_a = hermiticity_exp(ctx.gen_len);
        let r_b = hermiticity_exp(string.count());
        let r_c = hermiticity_exp(out.count());
        let swaps = resorting_parity(ctx.gen.words(), string.words()) as i32;
        let phase = match (r_a + r_b - r_c + 2 * swaps).rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        (out, phase * ctx.sign)
    }

    #[inline]
    fn weight(string: &BasisString<W>, n_units: usize) -> u32 {
        jw_weight(string, n_units)
    }

    fn trace(string: &BasisString<W>, n_units: usize, fock: &[u64]) -> f64 {
        let mut paired = 0i32;
        let mut product = 1i32;
        for k in 0..n_units {
            let (low, high) = (string.test(2 * k), string.test(2 * k + 1));
            if low != high {
                return 0.0;
            }
            if low {
                let n_k = (fock.get(k / 64).copied().unwrap_or(0) >> (k % 64)) & 1;
                product *= 2 * n_k as i32 - 1;
                paired += 1;
            }
        }
        let phase = if (paired / 2) % 2 == 0 { 1 } else { -1 };
        (phase * product) as f64
    }
}

/// Converts a `MajoranaMonomial` into the interleaved basis-string form.
pub fn to_basis_string<const W: usize>(term: &MajoranaMonomial) -> BasisString<W> {
    let mut m = BasisString::zero();
    for (i, &word) in term.modes.as_words().iter().enumerate() {
        if i < W {
            let mut w = word;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                m.set(i * 64 + b);
                w &= w - 1;
            }
        }
    }
    m
}

/// Rebuilds a `MajoranaMonomial` from the interleaved basis-string form.
pub fn from_basis_string<const W: usize>(
    string: &BasisString<W>,
    n_modes: usize,
) -> MajoranaMonomial {
    let mut words = vec![0u64; n_modes.div_ceil(64).max(1)];
    for pos in string.positions() {
        if pos < n_modes {
            words[pos / 64] |= 1u64 << (pos % 64);
        }
    }
    MajoranaMonomial::from_modes(Bitset::from_words(words), n_modes)
}
