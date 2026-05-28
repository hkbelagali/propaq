use pyo3::prelude::*;

use propaq_core::propagator::{AbstractPropagator, PropagationResult};

use crate::monomial::MajoranaMonomial;
use crate::termsum::MajoranaTermSum;

#[pyclass]
pub struct MajoranaPropagator {
    inner: AbstractPropagator<MajoranaMonomial>,
}

#[pymethods]
impl MajoranaPropagator {
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, progress_bar=false, truncation_threshold=10_000_000))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
        n_threads: Option<usize>,
        progress_bar: bool,
        truncation_threshold: usize,
    ) -> PyResult<Self> {
        Ok(MajoranaPropagator {
            inner: AbstractPropagator::new(noise, truncation, n_threads, progress_bar, truncation_threshold)?,
        })
    }

    #[getter]
    fn truncation_threshold(&self) -> usize {
        self.inner.truncation_threshold
    }

    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<MajoranaTermSum> {
        let mut evolved = observable.inner.copy();
        self.inner.run_propagate(py, &mut evolved, circuit)?;
        Ok(MajoranaTermSum { inner: evolved })
    }

    #[pyo3(signature = (observable, circuit, fock_state=0))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<PropagationResult> {
        let mut evolved = observable.inner.copy();
        self.inner.run_expectation_value(py, &mut evolved, circuit, fock_state)
    }
}
