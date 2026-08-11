//! 
//! Implements the Pauli basis and its algebra as an impl of Basis
//! 

use num_complex::Complex64;

use propaq_core::basis::{Basis, BasisKind};
use propaq_core::bitset::Bitset;
use propaq_core::strings::BasisString;

use crate::string::PauliString;


const X_MASK: u64 = 0x5555_5555_5555_5555;

#[inline]
fn count_y_sites<const W: usize>(m: &BasisString<W>) -> i32 {
    let mut n = 0u32;
    for &w in m.words() {
        n += (w & (w >> 1) & X_MASK).count_ones();
    }
    n as i32
}

/// Number of units where `a` has a Z component and `b` has an X component.
#[inline]
fn count_z_and_x<const W: usize>(a: &BasisString<W>, b: &BasisString<W>) -> i32 {
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
    gen: BasisString<W>,
    fold_gen: BasisString<W>,

    gen_y_sites: i32,

    sign: f64,
}

/// The Pauli basis.
pub struct PauliAlgebra;

impl<const W: usize> Basis<W> for PauliAlgebra {
    const KIND: BasisKind = BasisKind::Pauli;

    type GenContext = PauliGenContext<W>;

    #[inline]
    fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> Self::GenContext {
        PauliGenContext {
            gen: *gen,
            fold_gen: gen.pair_swap(),
            gen_y_sites: count_y_sites(gen),
            sign,
        }
    }

    #[inline]
    fn generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.gen
    }

    #[inline]
    fn anticommutes(ctx: &Self::GenContext, string: &BasisString<W>) -> bool {

        string.parity_and(&ctx.fold_gen)
    }

    #[inline]
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W> {
        &ctx.fold_gen
    }

    #[inline]
    fn product(ctx: &Self::GenContext, string: &BasisString<W>) -> (BasisString<W>, Complex64) {
        let out = *string ^ ctx.gen;
        let p = (ctx.gen_y_sites + count_y_sites(string) - count_y_sites(&out)
            + 2 * count_z_and_x(&ctx.gen, string))
        .rem_euclid(4);
        (out, i_pow(p) * ctx.sign)
    }

    #[inline]
    fn weight(string: &BasisString<W>, _n_units: usize) -> u32 {
        string.support() as u32
    }

    #[inline]
    fn trace(string: &BasisString<W>, _n_units: usize, fock: &[u64]) -> f64 {
        // Any X or Y component makes the term off-diagonal.
        for &w in string.words() {
            if w & X_MASK != 0 {
                return 0.0;
            }
        }

        let mut parity = 0u32;
        for pos in string.positions() {
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

/// Converts a `PauliString` into the interleaved basis-string form.
pub fn to_basis_string<const W: usize>(term: &PauliString) -> BasisString<W> {
    let mut m = BasisString::zero();
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

/// Rebuilds a `PauliString` from the interleaved basis-string form.
pub fn from_basis_string<const W: usize>(string: &BasisString<W>, n_qubits: usize) -> PauliString {
    let n_words = n_qubits.div_ceil(64).max(1);
    let mut xw = vec![0u64; n_words];
    let mut zw = vec![0u64; n_words];
    for pos in string.positions() {
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
    PauliString {
        x,
        z,
        n_qubits,
        weight,
    }
}

/// Writes a `PauliString`'s two word planes
pub fn planes_of(term: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut x = vec![0u64; stride];
    let mut z = vec![0u64; stride];
    let xw = term.x.as_words();
    let zw = term.z.as_words();
    x[..xw.len().min(stride)].copy_from_slice(&xw[..xw.len().min(stride)]);
    z[..zw.len().min(stride)].copy_from_slice(&zw[..zw.len().min(stride)]);
    (x, z)
}
