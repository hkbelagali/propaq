pub mod layout;
pub mod engine;
pub mod propagator;
pub mod streamer;
pub mod string;
pub mod termsum;

pub use propagator::PauliPropagator;
pub use streamer::PauliTermStreamer;
pub use string::PauliString;
pub use termsum::PauliTermSum;

/// The basis type that this crate runs. Other crates must use this alias,
/// not a concrete algebra type (docs/architecture/tracer.md, section 12).
pub type PauliBasis = propaq_traced::pauli::Pauli;
