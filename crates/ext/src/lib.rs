///
/// Export the core Rust functionality to Python via PyO3.
///
use pyo3::prelude::*;

use propaq_core::{
    TruncationPolicy, UniformNoiseModel, GateNoiseModel, NativeNoiseModel, NativeTruncator,
    PropagationResult, Logger,
};
use propaq_core::truncators::{
    CoefficientTruncator, FlushSchedule, FrequencyTruncator, MonomialBudget, Simplify, TermBudget,
    WeightTruncator,
};
use propaq_majorana::{MajoranaMonomial, MajoranaTermSum, MajoranaPropagator, MajoranaTermStreamer};
use propaq_pauli::{PauliString, PauliTermSum, PauliPropagator, PauliTermStreamer};
use propaq_surrogate::{
    FrequencyTruncationPolicy,
    PauliSurrogateModel, MajoranaSurrogateModel,
    PauliSurrogatePropagator, MajoranaSurrogatePropagator,
};
use propaq_hybrid::pyapi::hybrid_expectation;

#[pyfunction]
fn rust_available() -> bool {
    true
}

/// High-water mark, in bytes, of temporary dense workspace held live at once
/// since the last reset.
///
/// Reported separately from resident key storage: workspaces are borrowed for
/// the duration of one kernel call and never persisted. Each propagation run
/// resets this at its start, so reading it after a run gives that run's peak.
#[pyfunction]
fn workspace_peak_bytes() -> usize {
    propaq_core::store::workspace_peak_bytes()
}

/// Resets the temporary dense workspace high-water mark.
#[pyfunction]
fn reset_workspace_peak() {
    propaq_core::store::reset_workspace_peak()
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_function(wrap_pyfunction!(workspace_peak_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(reset_workspace_peak, m)?)?;
    m.add_class::<MajoranaMonomial>()?;
    m.add_class::<PauliString>()?;
    m.add_class::<MajoranaTermSum>()?;
    m.add_class::<MajoranaTermStreamer>()?;
    m.add_class::<PauliTermSum>()?;
    m.add_class::<PauliTermStreamer>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<NativeNoiseModel>()?;
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
    m.add_class::<NativeTruncator>()?;
    m.add_class::<Simplify>()?;
    m.add_class::<PauliSurrogateModel>()?;
    m.add_class::<MajoranaSurrogateModel>()?;
    m.add_class::<PauliSurrogatePropagator>()?;
    m.add_class::<MajoranaSurrogatePropagator>()?;
    m.add_function(wrap_pyfunction!(hybrid_expectation, m)?)?;
    Ok(())
}
