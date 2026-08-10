///
/// The algebra a propagator needs from its basis, expressed over
/// [`crate::monomial::Monomial`].
///
/// This mirrors monoprop's `algebra/Algebra.h` policy shape rather than the
/// older `TermBasis` word-plane trait. The difference that matters is
/// `GenContext`: everything derivable from a rotation's generator alone is
/// computed once per gate, so the per-term methods reduce to a masked popcount
/// over a compile-time number of words.
///
use num_complex::Complex64;

use crate::monomial::Monomial;

/// One basis (Pauli, Majorana) over monomials of `W` storage words.
pub trait Algebra<const W: usize>: Send + Sync + 'static {
    /// Per-gate precomputation derived from the generator.
    type GenContext: Send + Sync;

    /// Builds the per-gate context. Called once per rotation, not per term.
    fn make_gen_context(gen: &Monomial<W>) -> Self::GenContext {
        Self::make_signed_gen_context(gen, 1.0)
    }

    /// Builds a context for a generator carrying an overall sign.
    ///
    /// Conjugating a generator through a Clifford frame can produce `-G'`, and
    /// a rotation about `-G'` is a rotation about `G'` by the negated angle.
    /// Rather than negate the angle, which is impossible for a non-numeric
    /// `GateParam`, the sign is folded into the product phase: the cosine
    /// branch is even in the angle and so is unaffected, and the sine branch
    /// picks up exactly this factor.
    fn make_signed_gen_context(gen: &Monomial<W>, sign: f64) -> Self::GenContext;

    /// The generator this context was built from.
    fn generator(ctx: &Self::GenContext) -> &Monomial<W>;

    /// True if `mono` anticommutes with the context's generator, and therefore
    /// branches under a rotation about it.
    fn anticommutes(ctx: &Self::GenContext, mono: &Monomial<W>) -> bool;

    /// The monomial whose set positions name the inverted-index columns to XOR
    /// for the anticommutation fold.
    ///
    /// This is generally not the generator itself. For Pauli it is `J(G)`, the
    /// generator with each unit's two bits exchanged, because anticommutation
    /// pairs a term's X against the generator's Z and vice versa.
    fn fold_generator(ctx: &Self::GenContext) -> &Monomial<W>;

    /// True if the column fold needs a per-row `parity(|M|)` correction.
    ///
    /// Bases whose fold is exact return false. A basis returning true has the
    /// index's row-parity column folded in (`InvertedIndex::apply_row_parity`),
    /// which is what lets odd-length Majorana generators use the index at all.
    fn fold_needs_odd_correction(_ctx: &Self::GenContext) -> bool {
        false
    }

    /// The product `G * mono`: its key and its phase factor.
    ///
    /// Only called for anticommuting terms, where the phase is always purely
    /// imaginary. The phase is returned as a `Complex64` so it feeds
    /// [`crate::coeff::CoeffRepr::apply_rotation`] unchanged.
    fn product(ctx: &Self::GenContext, mono: &Monomial<W>) -> (Monomial<W>, Complex64);

    /// The term's weight: the number of non-identity single-unit factors.
    ///
    /// This is the quantity a structural cutoff bounds, and it is the analogue
    /// of monoprop's support cutoff.
    ///
    /// `n_units` is passed because it is not always derivable from the monomial:
    /// a Pauli's weight is its support and ignores it, but a Majorana's is the
    /// weight of its Jordan-Wigner image, whose Z-string runs to the end of the
    /// register. `TermBasis::weight` takes it for the same reason.
    fn weight(mono: &Monomial<W>, n_units: usize) -> u32;

    /// The term's diagonal expectation against a computational basis state,
    /// with `fock` holding one bit per unit.
    fn trace(mono: &Monomial<W>, n_units: usize, fock: &[u64]) -> f64;
}
