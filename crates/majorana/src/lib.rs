pub mod layout;
pub mod engine;
pub mod monomial;
pub mod propagator;
pub mod streamer;
pub mod termsum;

pub use monomial::MajoranaMonomial;
pub use propagator::MajoranaPropagator;
pub use streamer::MajoranaTermStreamer;
pub use termsum::MajoranaTermSum;

/// The basis type that this crate runs. Other crates must use this alias,
/// not a concrete algebra type (docs/architecture/tracer.md, section 12).
pub type MajoranaBasis = propaq_traced::majorana::Majorana;
