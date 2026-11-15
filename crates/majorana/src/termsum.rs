//!
//! The Majorana term sum, propagator and term streamer. The classes come from
//! `propaq_core::define_basis_module!`. The logic is in core.
//!

use crate::monomial::{MajoranaBasis, MajoranaMonomial};
use crate::streamer::MajoranaTermStreamer;

propaq_core::define_basis_classes! {
    codec: MajoranaBasis,
    term: MajoranaMonomial,
    coeff: f64,
    propagator: MajoranaPropagator,
    termsum: MajoranaTermSum,
    streamer: MajoranaTermStreamer,
    streamer_repr: "MajoranaTermStreamer",
    state_doc: "     initial_state: Fock state as a bitstring integer.",
    obs_doc: "Majorana observable",
    circuit_doc: "MajoranaCircuit",
    add_doc: "Add *coeff* * *term* to the sum, accumulating if the monomial is already present.",
    item_doc: "monomial",
    sum_doc: "MajoranaTermSum",
    file_item_doc: "monomials",
    streamer_doc: "MajoranaTermStreamer",
    propagator_docs: {
        /// Back-propagates Majorana observables through quantum circuits in the Heisenberg picture.
        ///
        /// Arguments:
        ///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
        ///     truncation: A list of truncators
        ///         (WeightTruncator, CoefficientTruncator, TermBudget), a single such
        ///         truncator, a legacy TruncationPolicy (decomposed), or None. The
        ///         symbolic-only FrequencyTruncator is rejected.
        ///     n_threads: Number of worker threads. Defaults to the system thread count.
        ///     logger: Optional Logger for verbose JSON Lines event logging.
        ///     pin_threads: Bind each worker to its own CPU.
        ///     progress_bar: Draw a tqdm bar over the gate loop.
        ///     progress_every: Gates between progress bar ticks. Defaults to 1.
        ///
    },
    termsum_docs: {
        /// A mutable, weighted sum of Majorana monomials with real coefficients.
        ///
        /// Arguments:
        ///     terms: Optional initial mapping of MajoranaMonomial to real coefficient.
        ///     dtype: Coefficient precision, "float64" (default) or "float32".
    },
}

/// Flattens a stored term sum to a map of terms.
pub fn materialize<C: propaq_core::coeff::CoeffRepr>(
    terms: &propaq_core::store::TermSum<C>,
) -> rustc_hash::FxHashMap<MajoranaMonomial, f64> {
    propaq_core::basis_module::materialize::<MajoranaBasis, C>(terms)
}

/// Builds a stored term sum from (term, coefficient) pairs.
pub fn term_sum_from_pairs<C: propaq_core::coeff::CoeffRepr>(
    pairs: Vec<(MajoranaMonomial, C)>,
    n_units: usize,
) -> propaq_core::store::TermSum<C> {
    propaq_core::basis_module::term_sum_from_pairs::<MajoranaBasis, C>(pairs, n_units)
}
