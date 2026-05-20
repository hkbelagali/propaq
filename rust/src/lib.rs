use pyo3::prelude::*;

#[pyfunction]
fn rust_available() -> bool { 
    true 
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    Ok(())
}