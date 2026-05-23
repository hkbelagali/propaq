use pyo3::prelude::*;

mod monomial;
mod termsum;
pub use monomial::MajoranaMonomial;
pub use termsum::MajoranaTermSum;

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<MajoranaMonomial>()?;
    m.add_class::<MajoranaTermSum>()?;
    Ok(())
}