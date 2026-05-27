use pyo3::prelude::*;
use pyo3::types::PyDict;
use num_complex::Complex64;

use propaq_core::termsum::AbstractTermSum;

use crate::string::PauliString;

#[pyclass(subclass)]
pub struct PauliTermSum {
    pub inner: AbstractTermSum<PauliString>,
}

#[pymethods]
impl PauliTermSum {
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = AbstractTermSum::new();
        if let Some(dict) = terms {
            inner.terms.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: PauliString = k.extract()?;
                let val: Complex64 = v.extract()?;
                inner.terms.push((key, val));
            }
        }
        Ok(PauliTermSum { inner })
    }

    fn add(&mut self, term: PauliString, coeff: Complex64) {
        self.inner.add(term, coeff);
    }

    fn scale(&mut self, factor: Complex64) {
        self.inner.scale(factor);
    }

    fn merge(&mut self, other: &PauliTermSum) {
        self.inner.merge(&other.inner);
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.truncate(policy)
    }

    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        self.inner.apply_damping(noise, active_modes)
    }

    fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    fn items(&self) -> Vec<(PauliString, Complex64)> {
        self.inner.terms.clone()
    }

    fn __len__(&self) -> usize {
        self.inner.terms.len()
    }

    fn __setitem__(&mut self, term: PauliString, coeff: Complex64) {
        self.inner.terms.retain(|(t, _)| t != &term);
        self.inner.terms.push((term, coeff));
    }

    fn __getitem__(&self, term: &PauliString) -> Complex64 {
        self.inner.terms
            .iter()
            .filter(|(t, _)| t == term)
            .map(|(_, c)| *c)
            .sum()
    }

    fn copy(&self) -> PauliTermSum {
        PauliTermSum { inner: self.inner.copy() }
    }
}
