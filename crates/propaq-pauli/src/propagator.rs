use pyo3::prelude::*;

use propaq_core::propagator::{AbstractPropagator, PropagationResult};

use crate::string::PauliString;
use crate::termsum::PauliTermSum;

/// Back-propagates Pauli observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: Optional TruncationPolicy controlling weight and coefficient cutoffs.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliPropagator {
    inner: AbstractPropagator<PauliString>,
}

#[pymethods]
impl PauliPropagator {
    /// Initialize the Pauli propagator.
    ///
    /// Arguments:
    ///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
    ///     truncation: Optional TruncationPolicy controlling weight and coefficient cutoffs.
    ///     n_threads: Number of worker threads. Defaults to the system thread count.
    ///     progress_bar: Display a tqdm progress bar during propagation.
    ///     logger: Optional Logger for verbose JSON Lines event logging.
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        Ok(PauliPropagator {
            inner: AbstractPropagator::new(noise, truncation, n_threads, progress_bar, logger)?,
        })
    }

    /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A PauliCircuit whose rotations are applied in reverse.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, filename=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        filename: Option<String>,
    ) -> PyResult<PauliTermSum> {
        let mut evolved = observable.inner.copy();
        self.inner.run_propagate(py, &mut evolved, circuit, filename.as_deref())?;
        Ok(PauliTermSum { inner: evolved })
    }

    /// Compute the expectation value of *observable* in the state prepared by *circuit*.
    ///
    /// Arguments:
    ///     observable: The Pauli observable.
    ///     circuit: A PauliCircuit applied to the reference state.
    ///     initial_state: Computational basis reference state as a bitstring integer.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, initial_state=0, filename=None))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
        filename: Option<String>,
    ) -> PyResult<PropagationResult> {
        let mut evolved = observable.inner.copy();
        self.inner.run_expectation_value(py, &mut evolved, circuit, initial_state, filename.as_deref())
    }

    /// The noise model used during propagation, if any.
    #[getter]
    fn noise(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner.noise.as_ref().map(|n| n.clone_ref(py))
    }

    /// Set the noise model for this propagator.
    #[pyo3(signature = (noise=None))]
    fn set_noise(&mut self, noise: Option<PyObject>) {
        self.inner.noise = noise;
    }

    /// The truncation policy used during propagation, if any.
    #[getter]
    fn truncation(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner.truncation.as_ref().map(|t| t.clone_ref(py))
    }

    /// Set the truncation policy for this propagator.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<PyObject>) {
        self.inner.truncation = truncation;
    }
}
