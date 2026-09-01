///
/// Export the core Rust functionality to Python via PyO3.
///
use pyo3::prelude::*;
use pyo3_stub_gen::{derive::gen_stub_pyfunction, Result as StubResult, StubInfo};

use propaq_core::truncators::{
    CoefficientTruncator, FrequencyTruncator, Simplify, TermBudget, WeightTruncator,
};
use propaq_core::{
    GateNoiseModel, Logger, NativeNoiseModel, NativeTruncator, PropagationResult, TruncationPolicy,
    UniformNoiseModel,
};
use propaq_hybrid::pyapi::hybrid_expectation;
use propaq_majorana::{
    MajoranaMonomial, MajoranaPropagator, MajoranaTermStreamer, MajoranaTermSum,
};
use propaq_pauli::{PauliPropagator, PauliString, PauliTermStreamer, PauliTermSum};
use propaq_surrogate::{
    FrequencyTruncationPolicy, MajoranaSurrogateModel, MajoranaSurrogatePropagator,
    PauliSurrogateModel, PauliSurrogatePropagator,
};

#[gen_stub_pyfunction(module = "propaq._rust_core")]
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
    m.add_class::<NativeNoiseModel>()?;
    m.add_class::<MajoranaPropagator>()?;
    m.add_class::<PauliPropagator>()?;
    m.add_class::<PropagationResult>()?;
    m.add_class::<Logger>()?;
    m.add_class::<FrequencyTruncationPolicy>()?;
    m.add_class::<FrequencyTruncator>()?;
    m.add_class::<CoefficientTruncator>()?;
    m.add_class::<WeightTruncator>()?;
    m.add_class::<TermBudget>()?;
    m.add_class::<NativeTruncator>()?;
    m.add_class::<Simplify>()?;
    m.add_class::<PauliSurrogateModel>()?;
    m.add_class::<MajoranaSurrogateModel>()?;
    m.add_class::<PauliSurrogatePropagator>()?;
    m.add_class::<MajoranaSurrogatePropagator>()?;
    m.add_function(wrap_pyfunction!(hybrid_expectation, m)?)?;
    Ok(())
}

pub fn stub_info() -> StubResult<StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("../../pyproject.toml"))
}
