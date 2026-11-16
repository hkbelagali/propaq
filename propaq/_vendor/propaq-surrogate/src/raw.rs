//!
//! Surrogate classes for compiled traced bases.
//!
//! A generated basis crate calls `define_raw_surrogate!` and lists
//! `register_surrogate` in the `extra:` list of `define_raw_basis_module!`.
//! The module then holds `SurrogatePropagator` and `SurrogateModel`.
//!
use pyo3::prelude::*;

use crate::model::SurrogateModel;

#[doc(hidden)]
pub use propaq_core as __core;

/// Compiled surrogate model of a traced basis. It maps parameter angles to
/// an expectation value. It has the same methods as `PauliSurrogateModel`.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(name = "SurrogateModel", module = "propaq._rust_core")]
pub struct RawSurrogateModel {
    pub inner: SurrogateModel,
}

fn too_few(what: &str, got: usize, need: usize) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(format!(
        "{what} has {got} elements but model requires {need}"
    ))
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl RawSurrogateModel {
    /// Evaluate the expectation value. `params[i]` is the angle (radians) for parameter `i`.
    fn evaluate(&self, py: Python<'_>, params: Vec<f64>) -> PyResult<f64> {
        if params.len() < self.inner.n_params {
            return Err(too_few("params", params.len(), self.inner.n_params));
        }
        Ok(py.detach(|| self.inner.evaluate(&params)))
    }

    /// Evaluate for many parameter assignments at once.
    fn evaluate_batch(&self, py: Python<'_>, param_sets: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        for (i, params) in param_sets.iter().enumerate() {
            if params.len() < self.inner.n_params {
                return Err(too_few(
                    &format!("param_sets[{i}]"),
                    params.len(),
                    self.inner.n_params,
                ));
            }
        }
        Ok(py.detach(|| self.inner.evaluate_batch(&param_sets)))
    }

    /// Save to a gzip-compressed binary file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a model from a file produced by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = SurrogateModel::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(RawSurrogateModel { inner })
    }

    /// Number of parameter indices used by this model.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params
    }

    /// Number of compiled terms.
    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    #[getter]
    fn n_monomials(&self) -> u128 {
        self.inner.n_monomials()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurrogateModel(n_terms={}, n_params={})",
            self.inner.n_terms(),
            self.inner.n_params
        )
    }
}

/// Defines `SurrogatePropagator` and `register_surrogate` for one traced basis.
///
/// `codec` is the codec type of the basis (`Codec` in the module of
/// `define_raw_basis_module!`). `termsum` is the term sum class of that module.
/// Call it in the same module as `define_raw_basis_module!`.
#[macro_export]
macro_rules! define_raw_surrogate {
    (codec: $B:ty, termsum: $S:ident $(,)?) => {
        /// Back-propagates observables of a traced basis symbolically, producing a compiled
        /// model that can be re-evaluated for any parameter assignment.
        ///
        /// A basis with more than 2 branches per rotation raises ValueError in `build`.
        ///
        /// Arguments:
        ///     truncation: A truncator from this module, a list of them, or None.
        ///     n_threads: Number of worker threads. Defaults to the system thread count.
        ///     logger: Optional Logger for verbose JSON Lines event logging.
        ///     progress_bar: Draw a tqdm bar over the build's gate loop.
        ///     progress_every: Gates between progress bar ticks. Defaults to 1.
        #[::pyo3_stub_gen::derive::gen_stub_pyclass]
        #[::pyo3::pyclass(module = "propaq._rust_core")]
        pub struct SurrogatePropagator {
            inner: $crate::propagator::SurrogatePropagator<$B>,
        }

        #[::pyo3_stub_gen::derive::gen_stub_pymethods]
        #[::pyo3::pymethods]
        impl SurrogatePropagator {
            #[new]
            #[pyo3(signature = (truncation=None, n_threads=None, logger=None, progress_bar=false, progress_every=1))]
            fn new(
                truncation: Option<::pyo3::Bound<'_, ::pyo3::PyAny>>,
                n_threads: Option<usize>,
                logger: Option<::pyo3::Py<::pyo3::PyAny>>,
                progress_bar: bool,
                progress_every: usize,
            ) -> ::pyo3::PyResult<Self> {
                let truncators = $crate::propagator::resolve_truncation(truncation.as_ref())?;
                $crate::raw::__core::truncators::reject_numerical_only(&truncators)?;
                Ok(SurrogatePropagator {
                    inner: $crate::propagator::SurrogatePropagator::new(
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
            ///     observable: The observable to back-propagate (real coefficients).
            ///     circuit: A circuit with `layers` and `n_params`. Each rotation has a
            ///         `generator` term, a `param_index` or an `angle`, and `qiskit_gate_idx`.
            ///     initial_state: Computational basis reference state as a bitstring integer.
            #[pyo3(signature = (observable, circuit, initial_state=None))]
            fn build(
                &mut self,
                py: ::pyo3::Python<'_>,
                observable: &$S,
                circuit: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                initial_state: Option<&::pyo3::Bound<'_, ::pyo3::PyAny>>,
            ) -> ::pyo3::PyResult<$crate::raw::RawSurrogateModel> {
                $crate::build::check_supported::<<$B as $crate::raw::__core::basis_module::BasisCodec>::Alg>()?;
                let n_units = observable.core.n_units();
                let initial_state = match initial_state {
                    Some(v) => $crate::raw::__core::helpers::pyint_to_bitset(v, n_units)?,
                    None => $crate::raw::__core::bitset::Bitset::zero(),
                };
                let obs: Vec<_> = $crate::raw::__core::basis_module::materialize::<$B, f64>(
                    &observable.core.as_f64(),
                )
                .into_iter()
                .collect();
                let inner = self
                    .inner
                    .build_model(py, &obs, n_units, circuit, &initial_state)?;
                Ok($crate::raw::RawSurrogateModel { inner })
            }

            /// The active truncation pipeline as a list of truncator objects.
            #[getter]
            fn truncators(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<Vec<::pyo3::Py<::pyo3::PyAny>>> {
                self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
            }

            /// Replace the truncation pipeline.
            #[pyo3(signature = (truncation=None))]
            fn set_truncation(
                &mut self,
                truncation: Option<::pyo3::Bound<'_, ::pyo3::PyAny>>,
            ) -> ::pyo3::PyResult<()> {
                let truncators = $crate::propagator::resolve_truncation(truncation.as_ref())?;
                $crate::raw::__core::truncators::reject_numerical_only(&truncators)?;
                self.inner.truncators = truncators;
                Ok(())
            }
        }

        /// Adds the surrogate classes to a module.
        pub fn register_surrogate(
            m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>,
        ) -> ::pyo3::PyResult<()> {
            use ::pyo3::types::PyModuleMethods;
            m.add_class::<SurrogatePropagator>()?;
            m.add_class::<$crate::raw::RawSurrogateModel>()?;
            Ok(())
        }
    };
}
