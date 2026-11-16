//! Compile test: `define_basis_streamer!` expands for a minimal codec.

use num_complex::Complex64;

use propaq_core::basis::{Basis, BasisKind};
use propaq_core::raw_basis::{RawBasis, RawTerm};
use propaq_core::strings::BasisString;

/// A toy +-1 algebra, defined at every width.
struct Toy;

impl<const W: usize> Basis<W> for Toy {
    const KIND: BasisKind = BasisKind::Custom;
    type GenContext = BasisString<W>;
    fn make_signed_gen_context(gen: &BasisString<W>, _sign: f64) -> Self::GenContext {
        *gen
    }
    fn generator(ctx: &Self::GenContext) -> &BasisString<W> {
        ctx
    }
    fn anticommutes(ctx: &Self::GenContext, m: &BasisString<W>) -> bool {
        m.parity_and(ctx)
    }
    fn fold_generator(ctx: &Self::GenContext) -> &BasisString<W> {
        ctx
    }
    fn product(ctx: &Self::GenContext, m: &BasisString<W>) -> (BasisString<W>, Complex64) {
        (*m ^ *ctx, Complex64::new(0.0, 1.0))
    }
    fn weight(m: &BasisString<W>, _n: usize) -> u32 {
        m.count() as u32
    }
    fn trace(_m: &BasisString<W>, _n: usize, _d: &[u64]) -> f64 {
        0.0
    }
}

type Codec = RawBasis<Toy, 2>;
type Tm = RawTerm<Toy, 2>;

propaq_core::define_basis_streamer!(Codec, Tm, TmStreamer);

#[test]
fn the_streamer_class_exists() {
    assert!(std::mem::size_of::<TmStreamer>() > 0);
}
