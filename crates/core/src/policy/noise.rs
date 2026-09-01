//!
//! Noise models for propagator runs.
//!
//! Currently, only uniform depolarising noise is implemented
//! in the Rust core. However, the `GateNoiseModel` class
//! allows one to wrap a custom Python noise model object
//! that implements the same interface as `UniformNoiseModel`.
//! propaq also supports dynamically loaded noise models from
//! shared libraries written in compatible languages. These are
//! significantly faster than Python noise models due to GIL overhead,
//! and are competitive with built-in noise model performance.
//! We strongly recommend custom noise models be implemented
//! via the plugin ABI rather than Python.
//!

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Exponential damping noise: each term of weight w is scaled by \(\exp(-\gamma w)\), where \(w\) is the term's Pauli weight.
///
/// Arguments:
///     damping: Damping rate \(\gamma\).
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct UniformNoiseModel {
    #[pyo3(get, set)]
    pub damping: f64,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl UniformNoiseModel {
    /// Initialize the uniform noise model.
    ///
    /// Arguments:
    ///     damping: Per-weight damping rate \(\gamma\). Each term is multiplied by \(\exp(-\gamma w)\).
    #[new]
    fn new(damping: f64) -> Self {
        UniformNoiseModel { damping }
    }

    /// Return \(\exp(-\gamma w)\): the multiplicative factor applied to a term's coefficient.
    ///
    /// Arguments:
    ///     term_weight: Pauli weight of the term.
    ///     active_modes: Unused for uniform noise; present for API compatibility.
    // `active_modes` is unused by uniform damping but is part of the documented
    // noise-model interface, and pyo3 exposes the parameter name to Python, so
    // renaming it would break `damping_factor(w, active_modes=0)`.
    #[allow(unused_variables)]
    fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        (-self.damping * term_weight as f64).exp()
    }

    /// Apply uniform damping to all terms in *term_sum* in-place.
    ///
    /// Arguments:
    ///     term_sum: A MajoranaTermSum or PauliTermSum to damp in-place.
    fn apply_noise(&self, py: Python<'_>, term_sum: &Bound<'_, PyAny>) -> PyResult<()> {
        term_sum.call_method1("apply_damping", (self.clone().into_pyobject(py)?, 0u32))?;
        Ok(())
    }
}

/// Base class for a custom Python noise model. Subclass it and define
/// `damping_factor` or `damping_factor_term` directly on the subclass.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct GateNoiseModel;

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl GateNoiseModel {
    /// A subclass's own `__init__` constructor arguments are accepted and
    /// ignored here (`*args`/`**kwargs`)
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        GateNoiseModel
    }

    // Override this in subclasses
    fn damping_factor(&self, _term_weight: u32, _active_modes: u32) -> PyResult<f64> {
        Err(not_overridden("damping_factor"))
    }

    // We don't need to scaffold `damping_factor_term` because its existence
    // determines if the model is key-aware or not.
}

fn not_overridden(method: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "GateNoiseModel.{method} must be overridden by a subclass"
    ))
}
