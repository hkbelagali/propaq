//!
//! What a propagation run reports back to Python.
//!
use pyo3::prelude::*;

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct PropagationResult {
    #[pyo3(get)]
    pub n_terms: Vec<usize>,
    #[pyo3(get)]
    pub expectation_value: f64,
    /// Bytes of resident sparse term keys held by the evolved term sum at the
    /// end of the run.
    #[pyo3(get)]
    pub sparse_key_bytes: usize,
    /// Live terms whose magnitude is below the coefficient cutoff.
    #[pyo3(get)]
    pub terms_below_cutoff: usize,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PropagationResult {
    #[new]
    #[pyo3(signature = (expectation_value, n_terms, sparse_key_bytes=0, terms_below_cutoff=0))]
    fn new(
        expectation_value: f64,
        n_terms: Vec<usize>,
        sparse_key_bytes: usize,
        terms_below_cutoff: usize,
    ) -> Self {
        PropagationResult {
            n_terms,
            expectation_value,
            sparse_key_bytes,
            terms_below_cutoff,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PropagationResult(expectation_value={}, n_terms=[{} entries], \
             sparse_key_bytes={}, terms_below_cutoff={})",
            self.expectation_value,
            self.n_terms.len(),
            self.sparse_key_bytes,
            self.terms_below_cutoff
        )
    }
}
