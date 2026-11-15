//!
//! The Pauli term sum, propagator and term streamer. The classes come from
//! `propaq_core::define_basis_module!`. The logic is in core.
//!

use crate::streamer::PauliTermStreamer;
use crate::string::{PauliBasis, PauliString};

propaq_core::define_basis_classes! {
    codec: PauliBasis,
    term: PauliString,
    coeff: f64,
    propagator: PauliPropagator,
    termsum: PauliTermSum,
    streamer: PauliTermStreamer,
    streamer_repr: "PauliTermStreamer",
    state_doc: "     initial_state: Computational basis reference state as a bitstring integer.",
    obs_doc: "Pauli observable",
    circuit_doc: "PauliCircuit",
    add_doc: "Add *coeff* x *term* to the sum, accumulating if the monomial is already present.",
    item_doc: "monomial",
    sum_doc: "PauliTermSum",
    file_item_doc: "strings",
    streamer_doc: "PauliTermStreamer",
    propagator_docs: {
        /// Back-propagates Pauli observables through quantum circuits in the Heisenberg picture.
        ///
        /// Arguments:
        ///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
        ///     truncation: A list of truncators (WeightTruncator, CoefficientTruncator, TermBudget), a single such
        ///         truncator, a legacy TruncationPolicy (decomposed), or None. The
        ///         symbolic-only FrequencyTruncator is rejected.
        ///     n_threads: Number of worker threads. Defaults to the system thread count.
        ///     logger: Optional Logger for verbose JSON Lines event logging.
        ///     pin_threads: Bind each worker to its own CPU.
        ///     progress_bar: Draw a tqdm bar over the gate loop.
        ///     progress_every: Gates between progress bar ticks. Defaults to 1.
    },
    termsum_docs: {
        /// A mutable, weighted sum of Pauli strings with real coefficients.
        ///
        /// Arguments:
        ///     terms: Optional initial mapping of PauliString to real coefficient.
        ///     dtype: Coefficient precision, "float64" (default) or "float32".
    },
}

/// Flattens a stored term sum to a map of terms.
pub fn materialize<C: propaq_core::coeff::CoeffRepr>(
    terms: &propaq_core::store::TermSum<C>,
) -> rustc_hash::FxHashMap<PauliString, f64> {
    propaq_core::basis_module::materialize::<PauliBasis, C>(terms)
}

/// Builds a stored term sum from (term, coefficient) pairs.
pub fn term_sum_from_pairs<C: propaq_core::coeff::CoeffRepr>(
    pairs: Vec<(PauliString, C)>,
    n_units: usize,
) -> propaq_core::store::TermSum<C> {
    propaq_core::basis_module::term_sum_from_pairs::<PauliBasis, C>(pairs, n_units)
}
