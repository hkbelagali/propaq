///
/// Represent a linear combination of Pauli strings with real coefficients.
///
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use propaq_core::propagator::{load_terms_from_file, save_terms_to_file};
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::truncation::TruncationPolicy;

use crate::string::{PauliBasis, PauliString};
use crate::streamer::PauliTermStreamer;

/// A mutable, weighted sum of Pauli strings with real coefficients.
///
/// Arguments:
///     terms: Optional initial mapping of PauliString to real coefficient.
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct PauliTermSum {
    pub inner: SoaTermSum<f64>,
    index: FxHashMap<PauliString, usize>,
}

/// If `inner` is still empty and hasn't been sized yet, (re)initialize it for
/// `n_qubits`. 
fn ensure_sized(inner: &mut SoaTermSum<f64>, n_qubits: usize) {
    if inner.len() == 0 && inner.n_units != n_qubits {
        *inner = SoaTermSum::new(n_qubits, PauliBasis::stride_words(n_qubits));
    }
}

fn planes_of(term: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    let mut gz = vec![0u64; stride];
    PauliBasis::term_into_planes(term, term.n_qubits, [&mut gx, &mut gz]);
    (gx, gz)
}

pub fn materialize(terms: &SoaTermSum<f64>) -> FxHashMap<PauliString, f64> {
    let n = terms.len();
    let mut map = FxHashMap::default();
    map.reserve(n);
    for i in 0..n {
        let term = PauliBasis::term_from_planes(terms.term_planes(i), terms.n_units);
        map.insert(term, *terms.coeff(i));
    }
    map
}

impl PauliTermSum {
    /// Wrap a `SoaTermSum` produced by the propagator (or loaded from a
    /// file), rebuilding the key index it doesn't carry itself.
    pub fn from_soa(inner: SoaTermSum<f64>) -> Self {
        let mut index = FxHashMap::default();
        index.reserve(inner.len());
        for i in 0..inner.len() {
            let term = PauliBasis::term_from_planes(inner.term_planes(i), inner.n_units);
            index.insert(term, i);
        }
        PauliTermSum { inner, index }
    }
}

#[pymethods]
impl PauliTermSum {
    /// Initialize a Pauli term sum.
    ///
    /// Arguments:
    ///     terms: Optional initial mapping of PauliString to real coefficient.
    #[new]
    #[pyo3(signature = (terms=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut inner = SoaTermSum::new(0, PauliBasis::stride_words(0));
        let mut index = FxHashMap::default();
        if let Some(dict) = terms {
            index.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: PauliString = k.extract()?;
                let val: f64 = v.extract()?;
                ensure_sized(&mut inner, key.n_qubits);
                let (gx, gz) = planes_of(&key, inner.stride);
                let row = inner.len();
                inner.push([&gx, &gz], val);
                index.insert(key, row);
            }
        }
        Ok(PauliTermSum { inner, index })
    }

    /// Add *coeff* × *term* to the sum, accumulating if the monomial is already present.
    pub fn add(&mut self, term: PauliString, coeff: f64) {
        ensure_sized(&mut self.inner, term.n_qubits);
        if let Some(&row) = self.index.get(&term) {
            self.inner.coeffs[row] += coeff;
            return;
        }
        let (gx, gz) = planes_of(&term, self.inner.stride);
        let row = self.inner.len();
        self.inner.push([&gx, &gz], coeff);
        self.index.insert(term, row);
    }

    /// Multiply every coefficient by *factor* in-place.
    fn scale(&mut self, factor: f64) {
        let n = self.inner.len();
        for c in self.inner.coeffs[..n].iter_mut() {
            *c *= factor;
        }
    }

    /// Add all terms from *other* into this sum.
    pub fn merge(&mut self, other: &PauliTermSum) {
        let n = other.inner.len();
        for i in 0..n {
            let term = PauliBasis::term_from_planes(other.inner.term_planes(i), other.inner.n_units);
            self.add(term, *other.inner.coeff(i));
        }
    }

    /// Stream terms from a file and merge them into this sum one at a time,
    /// accumulating coefficients for strings already present.
    ///
    /// Arguments:
    ///     streamer: A PauliTermStreamer opened with PauliTermStreamer.from_file().
    fn merge_from_file(&mut self, streamer: &mut PauliTermStreamer) -> PyResult<()> {
        for result in streamer.inner.by_ref() {
            let (term, coeff) = result?;
            self.add(term, coeff);
        }
        Ok(())
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.len();
        let stride = self.inner.stride;
        let mut kept = SoaTermSum::new(self.inner.n_units, stride);

        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            for i in 0..n {
                let term = self.inner.term_planes(i);
                let w = PauliBasis::weight(term, self.inner.n_units);
                let c = *self.inner.coeff(i);
                if wc.is_none_or(|ww| w <= ww) && c.abs() >= cc {
                    kept.push(term, c);
                }
            }
        } else {
            for i in 0..n {
                let term = self.inner.term_planes(i);
                let w = PauliBasis::weight(term, self.inner.n_units);
                let c = *self.inner.coeff(i);
                let should_remove: bool =
                    policy.call_method1("should_truncate", (w, c.abs()))?.extract()?;
                if !should_remove {
                    kept.push(term, c);
                }
            }
        }

        *self = PauliTermSum::from_soa(kept);
        Ok(())
    }

    /// Apply noise damping to every coefficient.
    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        use propaq_core::noise::UniformNoiseModel;
        let n = self.inner.len();
        if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
            let d = unm.damping;
            for i in 0..n {
                let w = PauliBasis::weight(self.inner.term_planes(i), self.inner.n_units);
                self.inner.coeffs[i] *= (-d * w as f64).exp();
            }
            return Ok(());
        }
        for i in 0..n {
            let w = PauliBasis::weight(self.inner.term_planes(i), self.inner.n_units);
            let damping: f64 = noise.call_method1("damping_factor", (w, active_modes))?.extract()?;
            self.inner.coeffs[i] *= damping;
        }
        Ok(())
    }

    /// Return the sum of |coefficient|² over all terms.
    pub fn norm_squared(&self) -> f64 {
        self.inner.coeffs[..self.inner.len()].iter().map(|c| c * c).sum()
    }

    /// Return all (monomial, coefficient) pairs.
    fn items(&self) -> Vec<(PauliString, f64)> {
        let n = self.inner.len();
        (0..n)
            .map(|i| (PauliBasis::term_from_planes(self.inner.term_planes(i), self.inner.n_units), *self.inner.coeff(i)))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __setitem__(&mut self, term: PauliString, coeff: f64) {
        ensure_sized(&mut self.inner, term.n_qubits);
        if let Some(&row) = self.index.get(&term) {
            self.inner.coeffs[row] = coeff;
            return;
        }
        let (gx, gz) = planes_of(&term, self.inner.stride);
        let row = self.inner.len();
        self.inner.push([&gx, &gz], coeff);
        self.index.insert(term, row);
    }

    fn __getitem__(&self, term: &PauliString) -> f64 {
        self.index.get(term).map(|&row| self.inner.coeffs[row]).unwrap_or_default()
    }

    /// Return a shallow copy of this term sum.
    fn copy(&self) -> PauliTermSum {
        PauliTermSum { inner: self.inner.copy(), index: self.index.clone() }
    }

    /// Load a PauliTermSum from a gzip-compressed binary file saved by `propagate` or
    /// `expectation_value`.
    ///
    /// Arguments:
    ///     path: Path to the file written by the `filename` parameter.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<PauliTermSum> {
        let map = load_terms_from_file::<PauliString>(path)?;
        let n_qubits = map.keys().next().map_or(0, |t| t.n_qubits);
        let mut inner = SoaTermSum::new(n_qubits, PauliBasis::stride_words(n_qubits));
        let mut index = FxHashMap::default();
        index.reserve(map.len());
        for (term, coeff) in map {
            let (gx, gz) = planes_of(&term, inner.stride);
            let row = inner.len();
            inner.push([&gx, &gz], coeff);
            index.insert(term, row);
        }
        Ok(PauliTermSum { inner, index })
    }

    /// Save this term sum to a gzip-compressed binary file.
    ///
    /// Arguments:
    ///     path: Destination file path.
    fn save(&self, path: &str) -> PyResult<()> {
        save_terms_to_file(&materialize(&self.inner), path)
    }
}
