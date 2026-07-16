///
/// impl for the Majorana propagator, which works with observables
/// represented in the Majorana operator basis. The propagator is
/// just a wrapper around the generic `SoaPropagator`, incorporating
/// the Majorana algebra via `MajoranaBasis`.
///
use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::propagator::{save_terms_to_file, PropagationResult};
use propaq_core::soa::propagator::SoaPropagator;
use propaq_core::truncators::{reject_surrogate_only, resolve_truncation, FlushSchedule};

use crate::monomial::MajoranaBasis;
use crate::termsum::MajoranaTermSum;

/// Back-propagates Majorana observables through quantum circuits in the Heisenberg picture.
///
/// Arguments:
///     noise: Optional noise model (UniformNoiseModel, GateNoiseModel, or custom).
///     truncation: A list of truncators
///         (WeightTruncator, CoefficientTruncator, TermBudget), a single such
///         truncator, a legacy TruncationPolicy (decomposed), or None. The
///         symbolic-only FrequencyTruncator is rejected.
///     schedule: Optional FlushSchedule controlling the lossless merge cadence.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaPropagator {
    inner: SoaPropagator<MajoranaBasis>,
}

#[pymethods]
impl MajoranaPropagator {
    /// Initialize the Majorana propagator. See the class docstring for arguments.
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
        Ok(MajoranaPropagator {
            inner: SoaPropagator::new(noise, schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
    ///
    /// Arguments:
    ///     observable: The Majorana observable to back-propagate.
    ///     circuit: A MajoranaCircuit whose rotations are applied in reverse.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, filename=None))]
    fn propagate(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        filename: Option<String>,
    ) -> PyResult<MajoranaTermSum> {
        let mut evolved = observable.inner.copy();
        self.inner.run_propagate(py, &mut evolved, circuit)?;
        if let Some(path) = filename.as_deref() {
            save_terms_to_file(&crate::termsum::materialize(&evolved), path)?;
        }
        Ok(MajoranaTermSum::from_soa(evolved))
    }

    /// Compute the expectation value of *observable* in the state prepared by *circuit*.
    ///
    /// Arguments:
    ///     observable: The Majorana observable.
    ///     circuit: A MajoranaCircuit applied to the reference state.
    ///     initial_state: Fock state as a bitstring integer.
    ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
    #[pyo3(signature = (observable, circuit, initial_state=None, filename=None))]
    fn expectation_value(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
        filename: Option<String>,
    ) -> PyResult<PropagationResult> {
        let mut evolved = observable.inner.copy();
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.inner.n_units)?,
            None => Bitset::zero(),
        };
        let result = self.inner.run_expectation_value(py, &mut evolved, circuit, initial_state.as_words())?;
        if let Some(path) = filename.as_deref() {
            save_terms_to_file(&crate::termsum::materialize(&evolved), path)?;
        }
        Ok(result)
    }

    #[getter]
    fn noise(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner.noise.as_ref().map(|n| n.clone_ref(py))
    }

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
