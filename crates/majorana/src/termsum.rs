///
/// Represent a linear combination of Majorana monomials with real coefficients. 
///
use pyo3::prelude::*;
use pyo3::types::PyDict;

use propaq_core::propagator::{load_terms_from_file, save_terms_to_file};
use propaq_core::termsum::AbstractTermSum;

use crate::monomial::MajoranaMonomial;
use crate::streamer::MajoranaTermStreamer;

/// A mutable, weighted sum of Majorana monomials with real coefficients.
///
/// Arguments:
///     terms: Optional initial mapping of MajoranaMonomial to real coefficient.
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct MajoranaTermSum {
    pub inner: AbstractTermSum<MajoranaMonomial>,
}

#[pymethods]
impl MajoranaTermSum {
    /// Initialize a Majorana term sum.
    ///
    /// Arguments:
    ///     terms: Optional initial mapping of MajoranaMonomial to real coefficient.
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = AbstractTermSum::new();
        if let Some(dict) = terms {
            inner.terms.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: MajoranaMonomial = k.extract()?;
                let val: f64 = v.extract()?;
                inner.terms.insert(key, val);
            }
        }
        Ok(MajoranaTermSum { inner })
    }

    /// Add *coeff* × *term* to the sum, accumulating if the monomial is already present.
    fn add(&mut self, term: MajoranaMonomial, coeff: f64) {
        self.inner.add(term, coeff);
    }

    /// Multiply every coefficient by *factor* in-place.
    fn scale(&mut self, factor: f64) {
        self.inner.scale(factor);
    }

    /// Add all terms from *other* into this sum.
    fn merge(&mut self, other: &MajoranaTermSum) {
        self.inner.merge(&other.inner);
    }

    /// Stream terms from a file and merge them into this sum one at a time,
    /// accumulating coefficients for monomials already present.
    ///
    /// Arguments:
    ///     streamer: A MajoranaTermStreamer opened with MajoranaTermStreamer.from_file().
    fn merge_from_file(&mut self, streamer: &mut MajoranaTermStreamer) -> PyResult<()> {
        self.inner.merge_from_streamer(&mut streamer.inner)
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
    fn items(&self) -> Vec<(MajoranaMonomial, f64)> {
        self.inner.terms.iter().map(|(k, v)| (k.clone(), *v)).collect()
    }

    fn __len__(&self) -> usize {
        self.inner.terms.len()
    }

    fn __setitem__(&mut self, term: MajoranaMonomial, coeff: f64) {
        self.inner.terms.insert(term, coeff);
    }

    fn __getitem__(&self, term: &MajoranaMonomial) -> f64 {
        self.inner.terms.get(term).copied().unwrap_or_default()
    }

    /// Return a shallow copy of this term sum.
    fn copy(&self) -> MajoranaTermSum {
        MajoranaTermSum { inner: self.inner.copy() }
    }

    /// Load a MajoranaTermSum from a gzip-compressed binary file saved by `propagate` or
    /// `expectation_value`.
    ///
    /// Arguments:
    ///     path: Path to the file written by the `filename` parameter.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<MajoranaTermSum> {
        let terms = load_terms_from_file::<MajoranaMonomial>(path)?;
        Ok(MajoranaTermSum { inner: AbstractTermSum { terms } })
    }

    /// Save this term sum to a gzip-compressed binary file.
    ///
    /// Arguments:
    ///     path: Destination file path.
    fn save(&self, path: &str) -> PyResult<()> {
        save_terms_to_file(&self.inner.terms, path)
    }
}
