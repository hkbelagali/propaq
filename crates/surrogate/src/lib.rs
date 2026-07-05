pub mod symcoeff;
pub mod interning;
pub mod truncation;
pub mod model;
pub mod propagator;

pub use symcoeff::{GateParam, SymbolicCoeff};
pub use truncation::FrequencyTruncationPolicy;
// Composable truncators live in `propaq_core`; re-export for convenience.
pub use propaq_core::truncators::{
    CoefficientTruncator, FlushSchedule, FrequencyTruncator, MonomialBudget, TermBudget, Truncator,
    WeightTruncator,
};
pub use model::{SurrogateModel, PauliSurrogateModel, MajoranaSurrogateModel};
pub use propagator::{PauliSurrogatePropagator, MajoranaSurrogatePropagator};
