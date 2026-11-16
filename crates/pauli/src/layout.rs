//!
//! Converts a `PauliString` to the engine layout (bit 2q = x, bit 2q+1 = z).
//! The surrogate crate uses this function.
//!

use propaq_core::strings::BasisString;

use crate::string::PauliString;

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
