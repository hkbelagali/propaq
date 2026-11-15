//!
//! Trait for a basis-specific algebra, i.e. Pauli or Majorana.
//! The trait is defined over `BasisString<W>` bitset whose width
//! `W` is a compile-time constant.
//!
//! The Basis<W> trait was adopted from monoprop [1].
//!
//! [1]: https://github.com/Algorithmiq/monoprop
//!
//! Copyright and license from [1]:
//!
//! Copyright 2026 Algorithmiq
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//! <http://www.apache.org/licenses/LICENSE-2.0>
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

use num_complex::Complex64;

use crate::fold::LinearFold;
use crate::strings::BasisString;

/// Which algebra a basis string belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum BasisKind {
    Pauli = 0,
    Majorana = 1,
    /// A basis defined with the tracer (propaq.tracer). See docs/architecture/tracer.md.
    Custom = 2,
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
    /// with `diag_state` holding one bit per unit.
    fn trace(mono: &BasisString<W>, n_units: usize, diag_state: &[u64]) -> f64;

    // ---- Generalized interface (tracer refactor, docs/architecture/tracer.md) ----
    // Every item below has a default that reproduces the +-1 class, so the
    // existing Pauli and Majorana implementations compile unchanged.

    /// Number of branches one rotation can produce from a term, counting the
    /// term itself. The +-1 class has 2 (cos part kept, sin part emitted).
    const BRANCHES: usize = 2;

    /// Modulus M of the commutation character: G T = w_M^chi T G, w_M = exp(2 pi i / M).
    const CHAR_MODULUS: u32 = 2;

    /// Order D of the phase group. Phases are exp(2 pi i e / D).
    const PHASE_ORDER: u32 = 4;

    /// Commutation character chi(g, t) in Z_M. Zero means the term is unchanged.
    #[inline]
    fn character(ctx: &Self::GenContext, mono: &BasisString<W>) -> u32 {
        u32::from(Self::anticommutes(ctx, mono))
    }

    /// The character as a linear form in the bits of the term, if it is one.
    /// The engine then evaluates it from bit columns for all terms at once.
    #[inline]
    fn linear_fold(ctx: &Self::GenContext) -> Option<LinearFold<W>> {
        Some(LinearFold::z2(*Self::fold_generator(ctx), Self::fold_needs_odd_correction(ctx)))
    }

    /// Branch k (1 <= k < BRANCHES) of `mono`: the new key and its phase.
    /// For the +-1 class, branch 1 is the product G * mono.
    #[inline]
    fn branch(ctx: &Self::GenContext, mono: &BasisString<W>, k: usize) -> (BasisString<W>, Complex64) {
        debug_assert_eq!(k, 1, "the +-1 class has one emitted branch");
        Self::product(ctx, mono)
    }

    /// Per-gate coefficient table for BRANCHES > 2, row-major [chi][k] with
    /// CHAR_MODULUS rows and BRANCHES columns, for rotation angle `theta`.
    /// The +-1 class does not use it (the engine keeps its cos/sin path).
    #[inline]
    fn branch_table(_ctx: &Self::GenContext, _theta: f64) -> Vec<Complex64> {
        Vec::new()
    }
}
