///
/// Pauli algebra over the interleaved [`Monomial`] representation.
///
/// Bit convention: unit `k` occupies bits `2k` (its X component) and `2k + 1`
/// (its Z component), so the four single-qubit Paulis are
/// `00 -> I, 01 -> X, 10 -> Z, 11 -> Y` read as `(bit 2k, bit 2k+1)`. This is
/// the same information the old two-plane form carried in `x` and `z`, folded
/// into one bitset so a product is a single XOR.
///
use num_complex::Complex64;

use propaq_core::algebra::Algebra;
use propaq_core::bitset::Bitset;
use propaq_core::monomial::Monomial;

use crate::string::PauliString;

/// Selects the X bit of every unit pair.
const X_MASK: u64 = 0x5555_5555_5555_5555;

/// Number of units carrying both an X and a Z component, which are the Y sites.
///
/// This is the `popcount(x & z)` term of the Pauli product phase.
#[inline]
fn count_y_sites<const W: usize>(m: &Monomial<W>) -> i32 {
    let mut n = 0u32;
    for &w in m.words() {
        n += (w & (w >> 1) & X_MASK).count_ones();
    }
    n as i32
}

/// Number of units where `a` has a Z component and `b` has an X component.
///
/// This is the `popcount(a.z & b.x)` cross term of the product phase.
#[inline]
fn count_z_and_x<const W: usize>(a: &Monomial<W>, b: &Monomial<W>) -> i32 {
    let mut n = 0u32;
    for i in 0..W {
        n += ((a.words()[i] >> 1) & b.words()[i] & X_MASK).count_ones();
    }
    n as i32
}

/// `i` raised to `p`, for `p` already reduced modulo 4.
#[inline]
fn i_pow(p: i32) -> Complex64 {
    match p {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        3 => Complex64::new(0.0, -1.0),
        _ => unreachable!("phase exponent must be reduced mod 4"),
    }
}

/// Per-gate precomputation for a Pauli rotation.
pub struct PauliGenContext<const W: usize> {
    gen: Monomial<W>,
    /// `J(G)`, the generator with each unit's X and Z bits exchanged.
    ///
    /// Anticommutation is `parity(|M & J(G)|)`, so folding the swap into the
    /// context turns the per-term test into one masked popcount.
    fold_gen: Monomial<W>,
    /// `popcount(gen.x & gen.z)`, constant across the gate.
    gen_y_sites: i32,
    /// Overall sign carried by the generator, from a Clifford frame conjugation.
    sign: f64,
}

/// The Pauli basis.
pub struct PauliAlgebra;

impl<const W: usize> Algebra<W> for PauliAlgebra {
    type GenContext = PauliGenContext<W>;

    #[inline]
    fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext {
        PauliGenContext {
            gen: *gen,
            fold_gen: gen.pair_swap(),
            gen_y_sites: count_y_sites(gen),
            sign,
        }
    }

    #[inline]
    fn generator(ctx: &Self::GenContext) -> &Monomial<W> {
        &ctx.gen
    }

    #[inline]
    fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool {
        // parity(|M.x & G.z|) + parity(|M.z & G.x|) collapses to a single
        // parity against the pair-swapped generator.
        mono.parity_and(&ctx.fold_gen)
    }

    #[inline]
    fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W> {
        &ctx.fold_gen
    }

    #[inline]
    fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64) {
        let out = *mono ^ ctx.gen;
        let p = (ctx.gen_y_sites + count_y_sites(mono) - count_y_sites(&out)
            + 2 * count_z_and_x(&ctx.gen, mono))
            .rem_euclid(4);
        (out, i_pow(p) * ctx.sign)
    }

    #[inline]
    fn weight(mono: &Monomial<W>, _n_units: usize) -> u32 {
        mono.support() as u32
    }

    #[inline]
    fn trace(mono: &Monomial<W>, _n_units: usize, fock: &[u64]) -> f64 {
        // Any X or Y component makes the term off-diagonal.
        for &w in mono.words() {
            if w & X_MASK != 0 {
                return 0.0;
            }
        }
        // Every remaining position is a Z component at an odd bit, so its unit
        // is just the position halved.
        let mut parity = 0u32;
        for pos in mono.positions() {
            let unit = pos / 2;
            parity ^= (fock.get(unit / 64).copied().unwrap_or(0) >> (unit % 64)) as u32 & 1;
        }
        if parity == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

/// Converts a `PauliString` into the interleaved monomial form.
///
/// Panics if the string is wider than `W` words can hold.
pub fn to_monomial<const W: usize>(term: &PauliString) -> Monomial<W> {
    let mut m = Monomial::zero();
    let xw = term.x.as_words();
    let zw = term.z.as_words();
    for q in 0..term.n_qubits {
        let (word, bit) = (q / 64, q % 64);
        if xw.get(word).copied().unwrap_or(0) >> bit & 1 != 0 {
            m.set(2 * q);
        }
        if zw.get(word).copied().unwrap_or(0) >> bit & 1 != 0 {
            m.set(2 * q + 1);
        }
    }
    m
}

/// Rebuilds a `PauliString` from the interleaved monomial form.
pub fn from_monomial<const W: usize>(mono: &Monomial<W>, n_qubits: usize) -> PauliString {
    let n_words = n_qubits.div_ceil(64).max(1);
    let mut xw = vec![0u64; n_words];
    let mut zw = vec![0u64; n_words];
    for pos in mono.positions() {
        let q = pos / 2;
        if q >= n_qubits {
            continue;
        }
        let target = if pos % 2 == 0 { &mut xw } else { &mut zw };
        target[q / 64] |= 1u64 << (q % 64);
    }
    let x = Bitset::from_words(xw);
    let z = Bitset::from_words(zw);
    let weight = (&x | &z).count_ones();
    PauliString { x, z, n_qubits, weight }
}

/// Writes a `PauliString`'s two word planes, as the old `TermBasis` form.
///
/// Used by the differential tests to drive both engines from one term.
pub fn planes_of(term: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut x = vec![0u64; stride];
    let mut z = vec![0u64; stride];
    let xw = term.x.as_words();
    let zw = term.z.as_words();
    x[..xw.len().min(stride)].copy_from_slice(&xw[..xw.len().min(stride)]);
    z[..zw.len().min(stride)].copy_from_slice(&zw[..zw.len().min(stride)]);
    (x, z)
}
