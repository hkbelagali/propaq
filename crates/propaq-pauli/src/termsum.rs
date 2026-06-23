use pyo3::prelude::*;
use pyo3::types::PyDict;
use num_complex::Complex64;

use propaq_core::termsum::AbstractTermSum;

use crate::string::PauliString;

/// A mutable, weighted sum of Pauli strings with complex coefficients.
///
/// Arguments:
///     terms: Optional initial mapping of PauliString to complex coefficient.
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct PauliTermSum {
    pub inner: AbstractTermSum<PauliString>,
}

#[pymethods]
impl PauliTermSum {
    /// Initialize a Pauli term sum.
    ///
    /// Arguments:
    ///     terms: Optional initial mapping of PauliString to complex coefficient.
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = AbstractTermSum::new();
        if let Some(dict) = terms {
            inner.terms.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: PauliString = k.extract()?;
                let val: Complex64 = v.extract()?;
                inner.terms.insert(key, val);
            }
        }
        Ok(PauliTermSum { inner })
    }

    /// Add *coeff* × *term* to the sum, accumulating if the monomial is already present.
    fn add(&mut self, term: PauliString, coeff: Complex64) {
        self.inner.add(term, coeff);
    }

    /// Multiply every coefficient by *factor* in-place.
    fn scale(&mut self, factor: Complex64) {
        self.inner.scale(factor);
    }

    /// Add all terms from *other* into this sum.
    fn merge(&mut self, other: &PauliTermSum) {
        self.inner.merge(&other.inner);
    }

    /// Deduplicate and remove terms according to *policy*.
    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.truncate(policy)
    }

    /// Apply noise damping to every coefficient.
    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        self.inner.apply_damping(noise, active_modes)
    }

    /// Return the sum of |coefficient|² over all terms.
    fn norm_squared(&self) -> f64 {
        self.inner.norm_squared()
    }

    /// Return all (monomial, coefficient) pairs.
    fn items(&self) -> Vec<(PauliString, Complex64)> {
        self.inner.terms.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    fn __len__(&self) -> usize {
        self.inner.terms.len()
    }

    fn __setitem__(&mut self, term: PauliString, coeff: Complex64) {
        self.inner.terms.insert(term, coeff);
    }

    fn __getitem__(&self, term: &PauliString) -> Complex64 {
        self.inner.terms.get(term).copied().unwrap_or_default()
    }

    /// Return a shallow copy of this term sum.
    fn copy(&self) -> PauliTermSum {
        PauliTermSum { inner: self.inner.copy() }
    }
}
