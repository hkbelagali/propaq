///
/// impl for the Pauli propagator, which works with observables 
/// represented in the Pauli operator basis. The propagator is 
/// just a wrapper around the generic `AbstractPropagator`, incorporating 
/// the Pauli algebra and the Pauli string representation.
///
use pyo3::prelude::*;

use propaq_core::propagator::{AbstractPropagator, PropagationResult};
use propaq_core::truncators::{reject_surrogate_only, resolve_truncation, FlushSchedule};

use crate::string::PauliString;
use crate::termsum::PauliTermSum;

/// Back-propagates Pauli observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: A list of truncators (WeightTruncator, CoefficientTruncator, TermBudget), a single such
///         truncator, a legacy TruncationPolicy (decomposed), or None. The
///         symbolic-only FrequencyTruncator/MonomialBudget are rejected.
///     schedule: Optional FlushSchedule controlling the lossless merge cadence.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliPropagator {
    inner: AbstractPropagator<PauliString, f64>,
}

#[pymethods]
impl PauliPropagator {
    /// Initialize the Pauli propagator. See the class docstring for arguments.
    #[new]
    #[pyo3(signature = (noise=None, truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        noise: Option<PyObject>,
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        reject_surrogate_only(&truncators)?;
        Ok(PauliPropagator {
            inner: AbstractPropagator::new(noise, schedule, truncators, n_threads, progress_bar, logger)?,
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

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
    }

    /// The flush/merge schedule.
    #[getter]
    fn schedule(&self) -> FlushSchedule {
        self.inner.schedule.clone()
    }

    #[setter]
    fn set_schedule(&mut self, schedule: FlushSchedule) {
        self.inner.schedule = schedule;
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`); the current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        reject_surrogate_only(&truncators)?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}
