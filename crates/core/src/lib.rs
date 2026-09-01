//!
//! Main library for the propaq core.
//! Some of the core architecture for propagation
//! was adopted from monoprop [1]
//!  
//! [1] https://github.com/Algorithmiq/monoprop
//!

#[path = "engine/affinity.rs"]
pub mod affinity;
#[path = "algebra/basis.rs"]
pub mod basis;
#[path = "algebra/bitset.rs"]
pub mod bitset;
#[path = "algebra/coeff.rs"]
pub mod coeff;
#[path = "interface/helpers.rs"]
pub mod helpers;
#[path = "storage/inverted_index.rs"]
pub mod inverted_index;
#[path = "interface/logger.rs"]
pub mod logger;
#[path = "policy/native_noise.rs"]
pub mod native_noise;
#[path = "policy/native_truncator.rs"]
pub mod native_truncator;
#[path = "policy/noise.rs"]
pub mod noise;
#[path = "engine/noise_resolver.rs"]
pub mod noise_resolver;
#[path = "storage/operator_index.rs"]
pub mod operator_index;
#[path = "engine/partitioned_termsum.rs"]
pub mod partitioned_termsum;
#[path = "interface/progress.rs"]
pub mod progress;
#[path = "interface/results.rs"]
pub mod results;
#[path = "engine/run_config.rs"]
pub mod run_config;
#[path = "storage/sparse.rs"]
pub mod sparse;
#[path = "storage/store.rs"]
pub mod store;
#[path = "interface/streamer.rs"]
pub mod streamer;
#[path = "algebra/strings.rs"]
pub mod strings;
#[path = "algebra/tableau.rs"]
pub mod tableau;
#[path = "interface/term_io.rs"]
pub mod term_io;
#[path = "policy/term_kernel.rs"]
pub mod term_kernel;
#[path = "engine/termsum.rs"]
pub mod termsum;
#[path = "algebra/traits.rs"]
pub mod traits;
#[path = "policy/truncators.rs"]
pub mod truncators;

pub use basis::BasisKind;
pub use coeff::CoeffRepr;
pub use logger::Logger;
pub use native_noise::NativeNoiseModel;
pub use native_truncator::NativeTruncator;
pub use noise::{GateNoiseModel, UniformNoiseModel};
pub use progress::Progress;
pub use results::PropagationResult;
pub use term_kernel::{NoiseKernel, TermView, TruncationKernel};
pub use truncators::{
    reject_numerical_only, reject_surrogate_only, resolve_config, resolve_truncation,
    CoefficientTruncator, FrequencyTruncator, ResolvedConfig, Simplify, TermBudget,
    TruncationPolicy, Truncator, WeightTruncator,
};
