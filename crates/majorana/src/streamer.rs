use num_complex::Complex64;
use pyo3::prelude::*;

use propaq_core::streamer::TermStreamer;

use crate::monomial::MajoranaMonomial;

#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaTermStreamer {
    pub inner: TermStreamer<MajoranaMonomial>,
}

#[pymethods]
impl MajoranaTermStreamer {
    /// Open a gzip-compressed binary file for lazy streaming.
    ///
    /// Arguments:
    ///     path: Path to a file written by `MajoranaTermSum.save()`.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        Ok(Self { inner: TermStreamer::open(path)? })
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<(MajoranaMonomial, Complex64)>> {
        match self.inner.next() {
            None => Ok(None),
            Some(Ok(pair)) => Ok(Some(pair)),
            Some(Err(e)) => Err(e),
        }
    }
}
