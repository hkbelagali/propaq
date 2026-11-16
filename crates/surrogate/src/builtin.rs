//!
//! The Pauli and Majorana surrogate propagators. They call the generic
//! `SurrogatePropagator::build_model`, which runs on the core loop.
//!
use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::truncators::reject_numerical_only;

use crate::model::{MajoranaSurrogateModel, PauliSurrogateModel};
use crate::propagator::{resolve_truncation, SurrogatePropagator};
use propaq_majorana::monomial::MajoranaBasis;
use propaq_majorana::termsum::MajoranaTermSum;
use propaq_pauli::string::PauliBasis;
use propaq_pauli::termsum::PauliTermSum;

/// Back-propagates Pauli observables symbolically, producing a compiled model
/// that can be re-evaluated for any parameter assignment.
///
/// Arguments:
///     truncation: A list of truncator objects
///         (FrequencyTruncator, CoefficientTruncator, WeightTruncator,
///         TermBudget) applied at each flush, a single such truncator, a
///         legacy FrequencyTruncationPolicy (decomposed automatically), or None.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     logger: Optional Logger for verbose JSON Lines event logging.
///     progress_bar: Draw a tqdm bar over the build's gate loop.
///     progress_every: Gates between progress bar ticks. Defaults to 1.
///
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliBasis>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PauliSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, logger=None, progress_bar=false, progress_every=1))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        Ok(PauliSurrogatePropagator {
            inner: SurrogatePropagator::new(
                truncators,
                n_threads,
                logger,
                progress_bar,
                progress_every,
            )?,
        })
    }

    /// Compile the observable back-propagated through the circuit into a SurrogateModel.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A SurrogatePauliCircuit.
    ///     initial_state: Fock state as a bitstring integer (default 0).
    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PauliSurrogateModel> {
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.n_units())?,
            None => Bitset::zero(),
        };
        let obs: Vec<(propaq_pauli::string::PauliString, f64)> =
            propaq_pauli::termsum::materialize(&observable.as_f64())
                .into_iter()
                .collect();
        let inner =
            self.inner
                .build_model(py, &obs, observable.n_units(), circuit, &initial_state)?;
        Ok(PauliSurrogateModel { inner })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .truncators
            .iter()
            .map(|t| t.to_object(py))
            .collect()
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`).
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        self.inner.truncators = truncators;
        Ok(())
    }
}

/// Back-propagates Majorana observables symbolically.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogatePropagator {
    inner: SurrogatePropagator<MajoranaBasis>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl MajoranaSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, logger=None, progress_bar=false, progress_every=1))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(
                truncators,
                n_threads,
                logger,
                progress_bar,
                progress_every,
            )?,
        })
    }

    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<MajoranaSurrogateModel> {
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.n_units())?,
            None => Bitset::zero(),
        };
        let obs: Vec<(propaq_majorana::MajoranaMonomial, f64)> =
            propaq_majorana::termsum::materialize(&observable.as_f64())
                .into_iter()
                .collect();
        let inner =
            self.inner
                .build_model(py, &obs, observable.n_units(), circuit, &initial_state)?;
        Ok(MajoranaSurrogateModel { inner })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .truncators
            .iter()
            .map(|t| t.to_object(py))
            .collect()
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`).
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        self.inner.truncators = truncators;
        Ok(())
    }
}
