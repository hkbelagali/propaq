///
/// Export the core Rust functionality to Python via PyO3.
///
/// No longer overrides the global allocator (mimalloc was removed): that
/// was justified by the old shard-based propagator's cross-thread free
/// pattern (a term allocated by one thread's `apply_gate_inplace` call,
/// later freed by a different thread's outbox flush) — real-workload A/B
/// testing after the SoA rewrite showed mimalloc measurably *slower* than
/// the system allocator on the numerical propagators, which no longer have
/// that pattern (columnar buffers grow rarely via amortized doubling; the
/// hash-based merge's per-batch maps are allocated and dropped by the same
/// thread). The surrogate propagator still uses the shard-based engine, but
/// wasn't shown to depend on mimalloc either, and removing a global
/// allocator override is the simpler default absent evidence it helps.
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
