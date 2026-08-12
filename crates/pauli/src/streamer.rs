///
/// impl for Pauli propagators' lazy loading functionality.
///
use pyo3::prelude::*;

use propaq_core::streamer::TermStreamer;

use crate::string::PauliString;

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct PauliTermStreamer {
    pub inner: TermStreamer<PauliString>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PauliTermStreamer {
    /// Open a gzip-compressed binary file for lazy streaming.
    ///
    /// Arguments:
    ///     path: Path to a file written by `PauliTermSum.save()`.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Ok(Self {
            inner: TermStreamer::open(path)?,
        })
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<(PauliString, f64)>> {
        match self.inner.next() {
            None => Ok(None),
            Some(Ok(pair)) => Ok(Some(pair)),
            Some(Err(e)) => Err(e),
        }
    }
}
