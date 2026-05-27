use pyo3::prelude::*;

use propaq_core::propagator::{AbstractPropagator, PropagationResult};

use crate::string::PauliString;
use crate::termsum::PauliTermSum;

#[pyclass]
pub struct PauliPropagator {
    inner: AbstractPropagator<PauliString>,
}

#[pymethods]
impl PauliPropagator {
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, progress_bar=false, truncation_interval=1))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
        n_threads: Option<usize>,
        progress_bar: bool,
        truncation_interval: usize,
    ) -> PyResult<Self> {
        Ok(PauliPropagator {
            inner: AbstractPropagator::new(noise, truncation, n_threads, progress_bar, truncation_interval)?,
        })
    }

    #[getter]
    fn truncation_interval(&self) -> usize {
        self.inner.truncation_interval
    }

    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<PauliTermSum> {
        let mut evolved = observable.inner.copy();
        self.inner.run_propagate(py, &mut evolved, circuit)?;
        Ok(PauliTermSum { inner: evolved })
    }

    #[pyo3(signature = (observable, circuit, fock_state=0))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<PropagationResult> {
        let mut evolved = observable.inner.copy();
        self.inner.run_expectation_value(py, &mut evolved, circuit, fock_state)
    }
}
