//!
//! Gate lowering and codec for the Majorana basis. The run loop lives in
//! `propaq_core::run`. The shared Python logic lives in
//! `propaq_core::basis_module`.
//!

use pyo3::prelude::*;

use propaq_core::basis_module::BasisCodec;

use crate::algebra::MajoranaAlgebra;
use propaq_core::bitset::Bitset;
use crate::monomial::{MajoranaBasis, MajoranaMonomial};

pub use propaq_core::partitioned_termsum::PhaseStats;
pub use propaq_core::run::GateRecord;

/// Widest site count the dispatch covers.
pub const MAX_DISPATCH_SITES: usize = 2048;

impl BasisCodec for MajoranaBasis {
    type Alg = MajoranaAlgebra;

    const UNIT_NAME: &'static str = "modes";
    const MAX_UNITS: usize = MAX_DISPATCH_SITES;

    fn engine_units(n_units: usize) -> Option<usize> {
        n_units.is_multiple_of(2).then_some(n_units / 2)
    }

    fn extract_term(obj: &Bound<'_, PyAny>) -> PyResult<MajoranaMonomial> {
        obj.extract().map_err(Into::into)
    }

    fn to_words(term: &MajoranaMonomial) -> Vec<u64> {
        words_of(term)
    }

    fn from_words(words: &[u64], n_units: usize) -> MajoranaMonomial {
        term_from_words(words, n_units)
    }
}

/// The raw words of a `MajoranaMonomial` (bit i = gamma_i).
fn words_of(term: &MajoranaMonomial) -> Vec<u64> {
    term.modes.as_words().to_vec()
}

/// Rebuilds a `MajoranaMonomial` from raw words. Bits at or past `n_modes` are dropped.
fn term_from_words(words: &[u64], n_modes: usize) -> MajoranaMonomial {
    let mut out = vec![0u64; n_modes.div_ceil(64).max(1)];
    for (i, o) in out.iter_mut().enumerate() {
        *o = words.get(i).copied().unwrap_or(0);
    }
    let rem = n_modes % 64;
    if rem != 0 {
        if let Some(last) = out.get_mut(n_modes / 64) {
            *last &= (1u64 << rem) - 1;
        }
    }
    MajoranaMonomial::from_modes(Bitset::from_words(out), n_modes)
}
