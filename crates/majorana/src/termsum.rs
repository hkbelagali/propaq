///
/// Represent a linear combination of Majorana monomials with real coefficients.
///
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use propaq_core::propagator::{load_terms_from_file, save_terms_to_file};
use propaq_core::soa::{SoaBasis, SoaTermSum};
use propaq_core::truncation::TruncationPolicy;

use crate::monomial::{MajoranaBasis, MajoranaMonomial};
use crate::streamer::MajoranaTermStreamer;

/// A mutable, weighted sum of Majorana monomials with real coefficients.
///
/// Arguments:
///     terms: Optional initial mapping of MajoranaMonomial to real coefficient.
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct MajoranaTermSum {
    pub inner: SoaTermSum<f64>,
    index: FxHashMap<MajoranaMonomial, usize>,
}

/// See `propaq_pauli::termsum::ensure_sized`.
fn ensure_sized(inner: &mut SoaTermSum<f64>, n_modes: usize) {
    if inner.len() == 0 && inner.n_units != n_modes {
        *inner = SoaTermSum::new(n_modes, MajoranaBasis::stride_words(n_modes));
    }
}

fn planes_of(term: &MajoranaMonomial, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut g0 = vec![0u64; stride];
    let mut g1 = vec![0u64; stride];
    MajoranaBasis::term_into_planes(term, term.n_modes, [&mut g0, &mut g1]);
    (g0, g1)
}

/// Materialize the columnar storage into the flat map format the existing
/// file I/O and `AbstractTerm` machinery already understand (see
/// `propaq_pauli::termsum::materialize` for why the surrogate propagator no
/// longer needs this as its bridge).
pub fn materialize(terms: &SoaTermSum<f64>) -> FxHashMap<MajoranaMonomial, f64> {
    let n = terms.len();
    let mut map = FxHashMap::default();
    map.reserve(n);
    for i in 0..n {
        let term = MajoranaBasis::term_from_planes(terms.term_planes(i), terms.n_units);
        map.insert(term, *terms.coeff(i));
    }
    map
}

impl MajoranaTermSum {
    /// Wrap a `SoaTermSum` produced by the propagator (or loaded from a
    /// file), rebuilding the key index it doesn't carry itself.
    pub fn from_soa(inner: SoaTermSum<f64>) -> Self {
        let mut index = FxHashMap::default();
        index.reserve(inner.len());
        for i in 0..inner.len() {
            let term = MajoranaBasis::term_from_planes(inner.term_planes(i), inner.n_units);
            index.insert(term, i);
        }
        MajoranaTermSum { inner, index }
    }
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
        let mut inner = SoaTermSum::new(0, MajoranaBasis::stride_words(0));
        let mut index = FxHashMap::default();
        if let Some(dict) = terms {
            index.reserve(dict.len());
            for (k, v) in dict.iter() {
                let key: MajoranaMonomial = k.extract()?;
                let val: f64 = v.extract()?;
                ensure_sized(&mut inner, key.n_modes);
                let (g0, g1) = planes_of(&key, inner.stride);
                let row = inner.len();
                inner.push([&g0, &g1], val);
                index.insert(key, row);
            }
        }
        Ok(MajoranaTermSum { inner, index })
    }

    /// Add *coeff* * *term* to the sum, accumulating if the monomial is already present.
    fn add(&mut self, term: MajoranaMonomial, coeff: f64) {
        ensure_sized(&mut self.inner, term.n_modes);
        if let Some(&row) = self.index.get(&term) {
            self.inner.coeffs[row] += coeff;
            return;
        }
        let (g0, g1) = planes_of(&term, self.inner.stride);
        let row = self.inner.len();
        self.inner.push([&g0, &g1], coeff);
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
    fn merge(&mut self, other: &MajoranaTermSum) {
        let n = other.inner.len();
        for i in 0..n {
            let term = MajoranaBasis::term_from_planes(other.inner.term_planes(i), other.inner.n_units);
            self.add(term, *other.inner.coeff(i));
        }
    }

    /// Stream terms from a file and merge them into this sum one at a time,
    /// accumulating coefficients for monomials already present.
    ///
    /// Arguments:
    ///     streamer: A MajoranaTermStreamer opened with MajoranaTermStreamer.from_file().
    fn merge_from_file(&mut self, streamer: &mut MajoranaTermStreamer) -> PyResult<()> {
        for result in streamer.inner.by_ref() {
            let (term, coeff) = result?;
            self.add(term, coeff);
        }
        Ok(())
    }

    /// Deduplicate and remove terms according to *policy*.
    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.inner.len();
        let stride = self.inner.stride;
        let n_units = self.inner.n_units;
        let mut kept = SoaTermSum::new(n_units, stride);

        if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
            let wc = tp.weight_cutoff;
            let cc = tp.coeff_cutoff;
            for i in 0..n {
                let term = self.inner.term_planes(i);
                let w = MajoranaBasis::weight(term, n_units);
                let c = *self.inner.coeff(i);
                if wc.is_none_or(|ww| w <= ww) && c.abs() >= cc {
                    kept.push(term, c);
                }
            }
        } else {
            for i in 0..n {
                let term = self.inner.term_planes(i);
                let w = MajoranaBasis::weight(term, n_units);
                let c = *self.inner.coeff(i);
                let should_remove: bool =
                    policy.call_method1("should_truncate", (w, c.abs()))?.extract()?;
                if !should_remove {
                    kept.push(term, c);
                }
            }
        }

        *self = MajoranaTermSum::from_soa(kept);
        Ok(())
    }

    /// Apply noise damping to every coefficient.
    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        use propaq_core::noise::UniformNoiseModel;
        let n = self.inner.len();
        let n_units = self.inner.n_units;
        if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
            let d = unm.damping;
            for i in 0..n {
                let w = MajoranaBasis::weight(self.inner.term_planes(i), n_units);
                self.inner.coeffs[i] *= (-d * w as f64).exp();
            }
            return Ok(());
        }
        for i in 0..n {
            let w = MajoranaBasis::weight(self.inner.term_planes(i), n_units);
            let damping: f64 = noise.call_method1("damping_factor", (w, active_modes))?.extract()?;
            self.inner.coeffs[i] *= damping;
        }
        Ok(())
    }

    /// Return the sum of |coefficient|² over all terms.
    fn norm_squared(&self) -> f64 {
        self.inner.coeffs[..self.inner.len()].iter().map(|c| c * c).sum()
    }

    /// Return all (monomial, coefficient) pairs.
    fn items(&self) -> Vec<(MajoranaMonomial, f64)> {
        let n = self.inner.len();
        (0..n)
            .map(|i| (MajoranaBasis::term_from_planes(self.inner.term_planes(i), self.inner.n_units), *self.inner.coeff(i)))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __setitem__(&mut self, term: MajoranaMonomial, coeff: f64) {
        ensure_sized(&mut self.inner, term.n_modes);
        if let Some(&row) = self.index.get(&term) {
            self.inner.coeffs[row] = coeff;
            return;
        }
        let (g0, g1) = planes_of(&term, self.inner.stride);
        let row = self.inner.len();
        self.inner.push([&g0, &g1], coeff);
        self.index.insert(term, row);
    }

    fn __getitem__(&self, term: &MajoranaMonomial) -> f64 {
        self.index.get(term).map(|&row| self.inner.coeffs[row]).unwrap_or_default()
    }

    /// Return a shallow copy of this term sum.
    fn copy(&self) -> MajoranaTermSum {
        MajoranaTermSum { inner: self.inner.copy(), index: self.index.clone() }
    }

    /// Load a MajoranaTermSum from a gzip-compressed binary file saved by `propagate` or
    /// `expectation_value`.
    ///
    /// Arguments:
    ///     path: Path to the file written by the `filename` parameter.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<MajoranaTermSum> {
        let map = load_terms_from_file::<MajoranaMonomial>(path)?;
        let n_modes = map.keys().next().map_or(0, |t| t.n_modes);
        let mut inner = SoaTermSum::new(n_modes, MajoranaBasis::stride_words(n_modes));
        let mut index = FxHashMap::default();
        index.reserve(map.len());
        for (term, coeff) in map {
            let (g0, g1) = planes_of(&term, inner.stride);
            let row = inner.len();
            inner.push([&g0, &g1], coeff);
            index.insert(term, row);
        }
        Ok(MajoranaTermSum { inner, index })
    }

    /// Save this term sum to a gzip-compressed binary file.
    ///
    /// Arguments:
    ///     path: Destination file path.
    fn save(&self, path: &str) -> PyResult<()> {
        save_terms_to_file(&materialize(&self.inner), path)
    }
}
