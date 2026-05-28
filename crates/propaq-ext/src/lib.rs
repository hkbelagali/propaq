use pyo3::prelude::*;

use propaq_core::{TruncationPolicy, UniformNoiseModel, GateNoiseModel, PropagationResult};
use propaq_majorana::{MajoranaMonomial, MajoranaTermSum, MajoranaPropagator};
use propaq_pauli::{PauliString, PauliTermSum, PauliPropagator};

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
    m.add_class::<PauliTermSum>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<MajoranaPropagator>()?;
    m.add_class::<PauliPropagator>()?;
    m.add_class::<PropagationResult>()?;
    Ok(())
}
