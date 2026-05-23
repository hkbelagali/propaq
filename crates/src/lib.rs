use pyo3::prelude::*;

mod bitset;
mod monomial;
mod truncation;
mod noise;
mod termsum;
mod propagator;

pub use monomial::MajoranaMonomial;
pub use termsum::MajoranaTermSum;
pub use truncation::TruncationPolicy;
pub use noise::{UniformNoiseModel, GateNoiseModel};
pub use propagator::MajoranaPropagator;

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<MajoranaMonomial>()?;
    m.add_class::<MajoranaTermSum>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<MajoranaPropagator>()?;
    Ok(())
}
