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

/// Exponential damping noise: each term of weight w is scaled by $\exp(-\gamma w)$, where $w$ is the term's Pauli weight.
///
/// Arguments:
///     damping: Damping rate $\gamma$.
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
    ///     damping: Per-weight damping rate $\gamma$. Each term is multiplied by $\exp(-\gamma w)$.
    #[new]
    fn new(damping: f64) -> Self {
        UniformNoiseModel { damping }
    }

    /// Return $\exp(-\gamma w)$: the multiplicative factor applied to a term's coefficient.
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
// TODO: Add dephasing noise model.

/// Noise model that delegates to an inner Python object's damping_factor and apply_noise.
///
/// Arguments:
///     inner: Python object exposing damping_factor(weight, active_modes) -> float
///            and apply_noise(term_sum) methods.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct GateNoiseModel {
    inner: Py<PyAny>,
}

impl GateNoiseModel {
    /// The wrapped object.
    ///
    /// Noise resolution unwraps the model before deciding its shape: what makes
    /// a model weight-only or key-aware is the object inside, and this wrapper
    /// forwards to it either way.
    pub fn inner(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner.clone_ref(py)
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl GateNoiseModel {
    /// Initialize the gate noise model wrapping a custom Python noise object.
    ///
    /// Arguments:
    ///     inner: Python object providing `damping_factor(weight, active_modes) -> float`
    ///            and `apply_noise(term_sum)` methods.
    #[new]
    fn new(inner: Py<PyAny>) -> Self {
        GateNoiseModel { inner }
    }

    /// The wrapped Python noise model object.
    #[getter]
    fn get_inner(&self, py: Python<'_>) -> Py<PyAny> {
        self.inner.clone_ref(py)
    }

    /// Delegate to the wrapped model's `damping_factor` method.
    fn damping_factor(&self, py: Python<'_>, term_weight: u32, active_modes: u32) -> PyResult<f64> {
        self.inner
            .call_method1(py, "damping_factor", (term_weight, active_modes))?
            .extract(py)
    }

    /// Delegate to the wrapped model's `damping_factor_term` method.
    ///
    /// Only meaningful for a wrapped object that defines one; propagation
    /// unwraps this model and calls the inner object directly, so this exists
    /// for symmetry with `damping_factor` rather than as the hot path.
    ///
    /// Arguments:
    ///     basis_kind: 0 for Pauli, 1 for Majorana.
    ///     words: The term's raw basis-string words, two bits per unit.
    ///     n_units: Qubits (Pauli) or modes (Majorana) of the register.
    ///     weight: The term's weight.
    fn damping_factor_term(
        &self,
        py: Python<'_>,
        basis_kind: u32,
        words: Vec<u64>,
        n_units: usize,
        weight: u32,
    ) -> PyResult<f64> {
        self.inner
            .call_method1(
                py,
                "damping_factor_term",
                (basis_kind, words, n_units, weight),
            )?
            .extract(py)
    }

    /// Delegate to the wrapped model's `apply_noise` method.
    fn apply_noise(&self, py: Python<'_>, term_sum: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.call_method1(py, "apply_noise", (term_sum,))?;
        Ok(())
    }
}
