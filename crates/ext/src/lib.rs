use pyo3::prelude::*;

use mimalloc::MiMalloc;

// Re-added for a controlled test isolating whether mimalloc itself was
// responsible for the earlier 250GB-vs-~1TB memory regression, versus the
// Monomial inline-capacity bump (8->16 factors) that was reverted separately
// in the same commit. If this combination (mimalloc + 8-factor capacity)
// still regresses, mimalloc's retention behavior is confirmed as the cause;
// if it now runs closer to ~1TB again, the capacity bump was the culprit.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use propaq_core::{TruncationPolicy, UniformNoiseModel, GateNoiseModel, PropagationResult, Logger};
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
    m.add_class::<PauliSurrogateModel>()?;
    m.add_class::<MajoranaSurrogateModel>()?;
    m.add_class::<PauliSurrogatePropagator>()?;
    m.add_class::<MajoranaSurrogatePropagator>()?;
    Ok(())
}
