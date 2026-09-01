///
/// Export the surrogate propagator and accompanying machinery!
///
/// In numerical propagation, the anticommuting update rule
/// explicitly computes the cosine and sine of the gate's parameter.
/// Since propagation runs generally cost significant wall time, it
/// makes it challenging to sweep system parameters, or variationally
/// optimize ansatze with respect to gate parameters. The surrogate
/// propagator replaces this functionality by symbolically representing
/// the cosine and sine of the gate parameter, following a build-once
/// evaluate-many strategy. At the end of the propagation, one will
/// have a performant mapping
//        $\theta \mapsto tr(U(\theta)^\dagger H U(\theta) \rho)$).
/// The evaluation of this mapping can be heavily optimized by pruning
/// terms with structurally zero contributions, as well as parallelism/vectorization
/// of the evaluation of the remaining terms.
///
/// Representation of symbolic coefficients is significantly more expensive in time
/// and memory than numerical coefficients, often necessitating more aggressive
/// truncation of the surrogate model.
///
pub mod engine;
pub mod model;
pub mod propagator;
pub mod symcoeff;
pub mod truncation;

pub use symcoeff::{CompiledCoeff, GateParam, SymbolicCoeff};
pub use truncation::FrequencyTruncationPolicy;

pub use model::{MajoranaSurrogateModel, PauliSurrogateModel, SurrogateModel};
pub use propagator::{MajoranaSurrogatePropagator, PauliSurrogatePropagator};
pub use propaq_core::truncators::{
    CoefficientTruncator, FrequencyTruncator, Simplify, TermBudget, Truncator, WeightTruncator,
};
