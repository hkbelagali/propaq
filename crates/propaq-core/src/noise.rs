use pyo3::prelude::*;

#[pyclass(subclass)]
#[derive(Clone)]
pub struct UniformNoiseModel {
    #[pyo3(get, set)]
    pub damping: f64,
}

#[pymethods]
impl UniformNoiseModel {
    #[new]
    fn new(damping: f64) -> Self {
        UniformNoiseModel { damping }
    }

    fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        (-self.damping * term_weight as f64).exp()
    }

    fn apply_noise(&self, py: Python<'_>, term_sum: &Bound<'_, PyAny>) -> PyResult<()> {
        term_sum.call_method1("apply_damping", (self.clone().into_pyobject(py)?, 0u32))?;
        Ok(())
    }
}

#[pyclass(subclass)]
pub struct GateNoiseModel {
    inner: PyObject,
}

#[pymethods]
impl GateNoiseModel {
    #[new]
    fn new(inner: PyObject) -> Self {
        GateNoiseModel { inner }
    }

    #[getter]
    fn get_inner(&self, py: Python<'_>) -> PyObject {
        self.inner.clone_ref(py)
    }

    fn damping_factor(&self, py: Python<'_>, term_weight: u32, active_modes: u32) -> PyResult<f64> {
        self.inner
            .call_method1(py, "damping_factor", (term_weight, active_modes))?
            .extract(py)
    }

    fn apply_noise(&self, py: Python<'_>, term_sum: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.call_method1(py, "apply_noise", (term_sum,))?;
        Ok(())
    }
}
