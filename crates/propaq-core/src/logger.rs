use pyo3::prelude::*;

#[pyclass]
pub struct Logger {
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub log_every: usize,
}

#[pymethods]
impl Logger {
    #[new]
    #[pyo3(signature = (filename, log_every=1))]
    pub fn new(filename: String, log_every: usize) -> Self {
        Logger { filename, log_every: log_every.max(1) }
    }
}
