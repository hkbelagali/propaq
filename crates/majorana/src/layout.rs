//!
//! Converts a `MajoranaMonomial` to the engine layout (bit i = gamma_i).
//! The surrogate crate uses this function.
//!

use propaq_core::strings::BasisString;

use crate::monomial::MajoranaMonomial;

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
