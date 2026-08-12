//!
//! Trait for a basis-specific algebra, i.e. Pauli or Majorana.
//! The trait is defined over `BasisString<W>` bitset whose width
//! `W` is a compile-time constant.
//!

use num_complex::Complex64;

use crate::strings::BasisString;

/// Which algebra a basis string belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum BasisKind {
    Pauli = 0,
    Majorana = 1,
}

impl BasisKind {
    /// The ABI's encoding of this kind.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }
}

///
/// Trait for a basis-specific algebra
///
pub trait Basis<const W: usize>: Send + Sync + 'static {
    /// Which algebra this is, for models that read a term's raw words.
    const KIND: BasisKind;

    /// GenContext caches generator information, that way we can compute it once
    /// and reuse it for every term in storage.
    type GenContext: Send + Sync;

    /// Builds the per-gate context.
    fn make_gen_context(gen: &BasisString<W>) -> Self::GenContext {
        Self::make_signed_gen_context(gen, 1.0)
    }

    /// Builds a context for a generator carrying an overall sign. This matters
    /// if we have a negative sign in front of the generator with a symbolic
    /// parameter. Since we can't negate the symbolic angle, we fold the
    /// sign into the product phase.
    fn make_signed_gen_context(gen: &BasisString<W>, sign: f64) -> Self::GenContext;

    /// The generator this context was built from.
    fn generator(ctx: &Self::GenContext) -> &BasisString<W>;

    /// True if `mono` anticommutes with the context's generator, and therefore
    /// branches under a rotation about it.
    fn anticommutes(ctx: &Self::GenContext, mono: &BasisString<W>) -> bool;

    /// The string whose set positions inform what columns
    /// to XOR for determining anticommutation. This is
    /// the generator itself for the Majorana basis,
    /// but a pair-swapped version for the Pauli basis.
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W>;

    /// Whether or not to XOR in the row-parity for anticommutation.
    /// This is used for odd-weight Majorana generators.
    fn fold_needs_odd_correction(_ctx: &Self::GenContext) -> bool {
        false
    }

    /// The product `G * mono`: its key and its phase factor.
    fn product(ctx: &Self::GenContext, mono: &BasisString<W>) -> (BasisString<W>, Complex64);

    /// The term's weight, this is taken to be the normal Pauli
    /// weight for Pauli terms, and the Pauli weight of the JW
    /// image for Majorana terms
    fn weight(mono: &BasisString<W>, n_units: usize) -> u32;

    /// The term's diagonal expectation against a computational basis state,
    /// with `fock` holding one bit per unit.
    fn trace(mono: &BasisString<W>, n_units: usize, fock: &[u64]) -> f64;
}
