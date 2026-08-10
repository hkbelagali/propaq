///
/// Majorana algebra over the interleaved [`Monomial`] representation.
///
/// Bit convention: bit `i` of the monomial is Majorana mode `gamma_i`, so
/// fermionic site `k` occupies bits `2k` and `2k + 1`. That is exactly what
/// `MajoranaMonomial::modes` already holds, and exactly what `Monomial`'s own
/// docstring specifies, so the mapping is the identity on bits and a product is
/// one XOR.
///
/// **`p` is not stored here.** `MajoranaMonomial` carries the Jordan-Wigner
/// Z-string parity plane alongside `modes`, but only as a cache:
/// `MajoranaBasis::key_hash` and `key_eq` read `modes` alone, and
/// `MajoranaMonomial::compute_weight_for` derives `p` from `modes` with a
/// log-depth prefix XOR. It is kept there to replace that scan with an XOR in
/// `weight_and_p_from_product`. Here it is recomputed inside [`Algebra::weight`],
/// which only runs when a structural cutoff is active and only for branches the
/// coefficient precheck already admitted, so the scan is off the hot path and
/// the key stays one bitset wide.
///
use num_complex::Complex64;

use propaq_core::algebra::Algebra;
use propaq_core::bitset::Bitset;
use propaq_core::monomial::Monomial;

use crate::monomial::{hermiticity_exp, resorting_parity, MajoranaMonomial};

/// Gathers the even-indexed bits of `x` into its low half.
///
/// The standard SWAR compress, used instead of BMI2's `PEXT` so the engine has
/// no per-call feature test and no scalar fallback to keep in step. Six shifts
/// against a `pext` latency of three is not worth two code paths here, since
/// this runs once per weight query rather than once per term.
#[inline]
fn gather_even(mut x: u64) -> u64 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    (x | (x >> 16)) & 0x0000_0000_ffff_ffff
}

/// Site-indexed planes: which sites carry `gamma_{2k}`, and which `gamma_{2k+1}`.
///
/// One monomial word spans 64 modes, so 32 sites, and two consecutive monomial
/// words fill one site word.
fn site_planes<const W: usize>(m: &Monomial<W>) -> ([u64; W], [u64; W]) {
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

/// The Jordan-Wigner qubit weight of a Majorana monomial over `n_sites` sites.
///
/// Ports `MajoranaMonomial::compute_weight_for` onto the fixed-width monomial:
/// compress to site planes, prefix-XOR the unpaired sites into the Z-string
/// parity `p`, then count the sites left non-identity.
fn jw_weight<const W: usize>(m: &Monomial<W>, n_sites: usize) -> u32 {
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
    gen: Monomial<W>,
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

impl<const W: usize> Algebra<W> for MajoranaAlgebra {
    type GenContext = MajoranaGenContext<W>;

    #[inline]
    fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext {
        let gen_len = gen.count();
        MajoranaGenContext { gen: *gen, gen_len, gen_len_odd: gen_len & 1 == 1, sign }
    }

    #[inline]
    fn generator(ctx: &Self::GenContext) -> &Monomial<W> {
        &ctx.gen
    }

    #[inline]
    fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool {
        // Two Majorana products commute iff `|M||G| + |M & G|` is even. With an
        // even-length generator the first term vanishes and this is the plain
        // overlap parity; with an odd one the term's own length enters, which is
        // the correction the inverted index needs `fold_needs_odd_correction`
        // for.
        let overlap = mono.count_and(&ctx.gen);
        let cross = if ctx.gen_len_odd { mono.count() } else { 0 };
        (cross + overlap) & 1 == 1
    }

    #[inline]
    fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W> {
        // Folding the generator's own columns yields the overlap parity, which
        // is the whole test for an even generator and all but the row-parity
        // correction for an odd one.
        &ctx.gen
    }

    #[inline]
    fn fold_needs_odd_correction(ctx: &Self::GenContext) -> bool {
        ctx.gen_len_odd
    }

    #[inline]
    fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64) {
        let out = *mono ^ ctx.gen;
        // Mirrors `MajoranaBasis::product`, which resorts the generator's modes
        // past the term's, in that order.
        let r_a = hermiticity_exp(ctx.gen_len);
        let r_b = hermiticity_exp(mono.count());
        let r_c = hermiticity_exp(out.count());
        let swaps = resorting_parity(ctx.gen.words(), mono.words()) as i32;
        let phase = match (r_a + r_b - r_c + 2 * swaps).rem_euclid(4) {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        (out, phase * ctx.sign)
    }

    #[inline]
    fn weight(mono: &Monomial<W>, n_units: usize) -> u32 {
        jw_weight(mono, n_units)
    }

    fn trace(mono: &Monomial<W>, n_units: usize, fock: &[u64]) -> f64 {
        // Ports `trace_fock_state_impl`: a site holding exactly one of its two
        // Majoranas is off-diagonal and kills the trace.
        let mut paired = 0i32;
        let mut product = 1i32;
        for k in 0..n_units {
            let (low, high) = (mono.test(2 * k), mono.test(2 * k + 1));
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

/// Converts a `MajoranaMonomial` into the interleaved monomial form.
pub fn to_monomial<const W: usize>(term: &MajoranaMonomial) -> Monomial<W> {
    let mut m = Monomial::zero();
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

/// Rebuilds a `MajoranaMonomial` from the interleaved monomial form.
pub fn from_monomial<const W: usize>(mono: &Monomial<W>, n_modes: usize) -> MajoranaMonomial {
    let mut words = vec![0u64; n_modes.div_ceil(64).max(1)];
    for pos in mono.positions() {
        if pos < n_modes {
            words[pos / 64] |= 1u64 << (pos % 64);
        }
    }
    MajoranaMonomial::from_modes(Bitset::from_words(words), n_modes)
}
