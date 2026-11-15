//!
//! Gate lowering and codec for the Pauli basis. The run loop lives in
//! `propaq_core::run`. The shared Python logic lives in
//! `propaq_core::basis_module`.
//!

use pyo3::prelude::*;

use propaq_core::basis_module::BasisCodec;

use crate::algebra::PauliAlgebra;
use propaq_core::bitset::Bitset;
use crate::string::{PauliBasis, PauliString};

pub use propaq_core::partitioned_termsum::PhaseStats;
pub use propaq_core::run::GateRecord;

/// Widest qubit count the dispatch covers.
pub const MAX_DISPATCH_QUBITS: usize = 2048;

impl BasisCodec for PauliBasis {
    type Alg = PauliAlgebra;

    const UNIT_NAME: &'static str = "qubits";
    const MAX_UNITS: usize = MAX_DISPATCH_QUBITS;

    fn engine_units(n_units: usize) -> Option<usize> {
        Some(n_units)
    }

    fn extract_term(obj: &Bound<'_, PyAny>) -> PyResult<PauliString> {
        obj.extract().map_err(Into::into)
    }

    fn to_words(term: &PauliString) -> Vec<u64> {
        words_of(term)
    }

    fn from_words(words: &[u64], n_units: usize) -> PauliString {
        term_from_words(words, n_units)
    }
}

/// The interleaved raw words of a `PauliString` (bit 2q = x, bit 2q+1 = z).
fn words_of(term: &PauliString) -> Vec<u64> {
    let mut out = vec![0u64; (2 * term.n_qubits).div_ceil(64).max(1)];
    let xw = term.x.as_words();
    let zw = term.z.as_words();
    for q in 0..term.n_qubits {
        let (word, bit) = (q / 64, q % 64);
        if xw.get(word).copied().unwrap_or(0) >> bit & 1 != 0 {
            out[(2 * q) / 64] |= 1u64 << ((2 * q) % 64);
        }
        if zw.get(word).copied().unwrap_or(0) >> bit & 1 != 0 {
            out[(2 * q + 1) / 64] |= 1u64 << ((2 * q + 1) % 64);
        }
    }
    out
}

/// Rebuilds a `PauliString` from raw interleaved words.
fn term_from_words(words: &[u64], n_qubits: usize) -> PauliString {
    let n_words = n_qubits.div_ceil(64).max(1);
    let mut xw = vec![0u64; n_words];
    let mut zw = vec![0u64; n_words];
    for q in 0..n_qubits {
        let (x_pos, z_pos) = (2 * q, 2 * q + 1);
        if words.get(x_pos / 64).copied().unwrap_or(0) >> (x_pos % 64) & 1 != 0 {
            xw[q / 64] |= 1u64 << (q % 64);
        }
        if words.get(z_pos / 64).copied().unwrap_or(0) >> (z_pos % 64) & 1 != 0 {
            zw[q / 64] |= 1u64 << (q % 64);
        }
    }
    let x = Bitset::from_words(xw);
    let z = Bitset::from_words(zw);
    let weight = (&x | &z).count_ones();
    PauliString { x, z, n_qubits, weight }
}
