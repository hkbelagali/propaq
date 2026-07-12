/// 
/// Main library for the propaq core.
///
pub mod bitset;
pub mod coeff;
pub mod traits;
pub mod truncation;
pub mod truncators;
pub mod noise;
pub mod helpers;
pub mod termsum;
pub mod propagator;
pub mod logger;
pub mod streamer;
pub mod soa;

pub use coeff::CoeffRepr;
pub use truncation::TruncationPolicy;
pub use truncators::{
    reject_surrogate_only, resolve_config, resolve_truncation, CoefficientTruncator, FlushSchedule,
    FrequencyTruncator, MonomialBudget, ResolvedConfig, TermBudget, Truncator, WeightTruncator,
};
pub use noise::{UniformNoiseModel, GateNoiseModel};
pub use propagator::PropagationResult;
pub use logger::Logger;
