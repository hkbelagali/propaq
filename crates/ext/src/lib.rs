///
/// Export the core Rust functionality to Python via PyO3.
/// 
/// The propagation (especially symbolic) will generate an
/// enormous number of memory allocations. Due to the shard-based
/// parallelism, the default allocator would likely suffer from 
/// contention and fragmentation. Therefore, we use mimalloc 
/// as the global allocator, which is designed for high-performance 
/// multithreaded workloads.
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use pyo3::prelude::*;

use propaq_core::{TruncationPolicy, UniformNoiseModel, GateNoiseModel, PropagationResult, Logger};
use propaq_core::truncators::{
    CoefficientTruncator, FlushSchedule, FrequencyTruncator, MonomialBudget, TermBudget,
    WeightTruncator,
};
use propaq_majorana::{MajoranaMonomial, MajoranaTermSum, MajoranaPropagator, MajoranaTermStreamer};
use propaq_pauli::{PauliString, PauliTermSum, PauliPropagator, PauliTermStreamer};
use propaq_surrogate::{
    FrequencyTruncationPolicy,
    PauliSurrogateModel, MajoranaSurrogateModel,
    PauliSurrogatePropagator, MajoranaSurrogatePropagator,
};

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<MajoranaMonomial>()?;
    m.add_class::<PauliString>()?;
    m.add_class::<MajoranaTermSum>()?;
    m.add_class::<MajoranaTermStreamer>()?;
    m.add_class::<PauliTermSum>()?;
    m.add_class::<PauliTermStreamer>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<MajoranaPropagator>()?;
    m.add_class::<PauliPropagator>()?;
    m.add_class::<PropagationResult>()?;
    m.add_class::<Logger>()?;
    m.add_class::<FrequencyTruncationPolicy>()?;
    m.add_class::<FlushSchedule>()?;
    m.add_class::<FrequencyTruncator>()?;
    m.add_class::<CoefficientTruncator>()?;
    m.add_class::<WeightTruncator>()?;
    m.add_class::<TermBudget>()?;
    m.add_class::<MonomialBudget>()?;
    m.add_class::<PauliSurrogateModel>()?;
    m.add_class::<MajoranaSurrogateModel>()?;
    m.add_class::<PauliSurrogatePropagator>()?;
    m.add_class::<MajoranaSurrogatePropagator>()?;
    Ok(())
}
