//!
//! Represent a linear combination of Pauli strings with real coefficients.
//!
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustc_hash::FxHashMap;

use propaq_core::coeff::CoeffRepr;
use propaq_core::store::{TermBasis, TermSum};
use propaq_core::term_io::{load_terms_from_file, save_terms_to_file};
use propaq_core::truncators::TruncationPolicy;

use crate::streamer::PauliTermStreamer;
use crate::string::{PauliBasis, PauliString};

/// Backing storage for a `PauliTermSum`
pub(crate) enum Storage {
    F64(TermSum<f64>),
    F32(TermSum<f32>),
}

impl Storage {
    fn len(&self) -> usize {
        match self {
            Storage::F64(s) => s.len(),
            Storage::F32(s) => s.len(),
        }
    }

    fn n_units(&self) -> usize {
        match self {
            Storage::F64(s) => s.n_units,
            Storage::F32(s) => s.n_units,
        }
    }
}

/// A mutable, weighted sum of Pauli strings with real coefficients.
///
/// Arguments:
///     terms: Optional initial mapping of PauliString to real coefficient.
///     dtype: Coefficient precision, "float64" (default) or "float32".
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct PauliTermSum {
    pub(crate) inner: Storage,
    index: FxHashMap<PauliString, usize>,
}

fn parse_dtype(dtype: Option<&str>) -> PyResult<&str> {
    match dtype.unwrap_or("float64") {
        "float64" => Ok("float64"),
        "float32" => Ok("float32"),
        other => Err(PyValueError::new_err(format!("unknown dtype: {other}"))),
    }
}

fn ensure_sized<C: CoeffRepr>(inner: &mut TermSum<C>, n_qubits: usize) {
    if inner.is_empty() && inner.n_units != n_qubits {
        *inner = TermSum::new(n_qubits, PauliBasis::stride_words(n_qubits));
    }
}

fn planes_of(term: &PauliString, stride: usize) -> (Vec<u64>, Vec<u64>) {
    let mut gx = vec![0u64; stride];
    let mut gz = vec![0u64; stride];
    PauliBasis::term_into_planes(term, term.n_qubits, [&mut gx, &mut gz]);
    (gx, gz)
}

/// Builds a term sum from the (key, coefficient) pairs the partitioned engine
/// produces.
pub fn term_sum_from_pairs<C: CoeffRepr>(
    pairs: Vec<(PauliString, C)>,
    n_units: usize,
) -> TermSum<C> {
    let mut inner = TermSum::<C>::new(n_units, PauliBasis::stride_words(n_units));
    let mut index = FxHashMap::default();
    index.reserve(pairs.len());
    for (term, coeff) in pairs {
        add_raw(&mut inner, &mut index, term, coeff);
    }
    inner
}

fn add_raw<C: CoeffRepr>(
    inner: &mut TermSum<C>,
    index: &mut FxHashMap<PauliString, usize>,
    term: PauliString,
    coeff: C,
) {
    ensure_sized(inner, term.n_qubits);
    if let Some(&row) = index.get(&term) {
        inner.coeffs[row].add_assign(coeff);
        return;
    }
    let (gx, gz) = planes_of(&term, inner.stride);
    let row = inner.len();
    inner.push([&gx, &gz], coeff);
    index.insert(term, row);
}

struct RowDecoder {
    buf: Vec<u64>,
}

impl RowDecoder {
    fn new(stride: usize) -> Self {
        RowDecoder {
            buf: vec![0u64; 2 * stride],
        }
    }

    fn term<C: CoeffRepr>(&mut self, terms: &TermSum<C>, i: usize) -> PauliString {
        PauliBasis::term_from_planes(terms.decode_row(i, &mut self.buf), terms.n_units)
    }
}

pub fn materialize<C>(terms: &TermSum<C>) -> FxHashMap<PauliString, f64>
where
    C: CoeffRepr,
{
    let n = terms.len();
    let mut map = FxHashMap::default();
    map.reserve(n);
    let mut decoder = RowDecoder::new(terms.stride);
    for i in 0..n {
        map.insert(decoder.term(terms, i), terms.coeff(i).to_f64());
    }
    map
}

impl PauliTermSum {
    pub fn from_store(inner: TermSum<f64>) -> Self {
        Self::from_storage(Storage::F64(inner))
    }

    /// Same as `from_store`, for an f32-backed `TermSum`.
    pub fn from_store_f32(inner: TermSum<f32>) -> Self {
        Self::from_storage(Storage::F32(inner))
    }

    fn from_storage(inner: Storage) -> Self {
        let mut index = FxHashMap::default();
        match &inner {
            Storage::F64(s) => {
                index.reserve(s.len());
                let mut decoder = RowDecoder::new(s.stride);
                for i in 0..s.len() {
                    index.insert(decoder.term(s, i), i);
                }
            }
            Storage::F32(s) => {
                index.reserve(s.len());
                let mut decoder = RowDecoder::new(s.stride);
                for i in 0..s.len() {
                    index.insert(decoder.term(s, i), i);
                }
            }
        }
        PauliTermSum { inner, index }
    }

    /// Number of qubits (precision-independent).
    pub fn n_units(&self) -> usize {
        self.inner.n_units()
    }

    /// This term sum's coefficients widened to f64, regardless of storage
    /// precision.
    pub fn as_f64(&self) -> TermSum<f64> {
        match &self.inner {
            Storage::F64(s) => s.map_coeffs(|c| *c),
            Storage::F32(s) => s.map_coeffs(|c| *c as f64),
        }
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PauliTermSum {
    /// Initialize a Pauli term sum.
    ///
    /// Arguments:
    ///     terms: Optional initial mapping of PauliString to real coefficient.
    ///     dtype: Coefficient precision, "float64" (default) or "float32".
    #[new]
    #[pyo3(signature = (terms=None, dtype=None))]
    fn new(terms: Option<&Bound<'_, PyDict>>, dtype: Option<&str>) -> PyResult<Self> {
        match parse_dtype(dtype)? {
            "float32" => {
                let mut inner = TermSum::<f32>::new(0, PauliBasis::stride_words(0));
                let mut index = FxHashMap::default();
                if let Some(dict) = terms {
                    index.reserve(dict.len());
                    for (k, v) in dict.iter() {
                        let key: PauliString = k.extract()?;
                        let val: f64 = v.extract()?;
                        add_raw(&mut inner, &mut index, key, val as f32);
                    }
                }
                Ok(PauliTermSum {
                    inner: Storage::F32(inner),
                    index,
                })
            }
            _ => {
                let mut inner = TermSum::<f64>::new(0, PauliBasis::stride_words(0));
                let mut index = FxHashMap::default();
                if let Some(dict) = terms {
                    index.reserve(dict.len());
                    for (k, v) in dict.iter() {
                        let key: PauliString = k.extract()?;
                        let val: f64 = v.extract()?;
                        add_raw(&mut inner, &mut index, key, val);
                    }
                }
                Ok(PauliTermSum {
                    inner: Storage::F64(inner),
                    index,
                })
            }
        }
    }

    /// Coefficient precision: "float64" or "float32".
    #[getter]
    fn dtype(&self) -> &str {
        match &self.inner {
            Storage::F64(_) => "float64",
            Storage::F32(_) => "float32",
        }
    }

    /// Add *coeff* x *term* to the sum, accumulating if the monomial is already present.
    pub fn add(&mut self, term: PauliString, coeff: f64) {
        match &mut self.inner {
            Storage::F64(inner) => add_raw(inner, &mut self.index, term, coeff),
            Storage::F32(inner) => add_raw(inner, &mut self.index, term, coeff as f32),
        }
    }

    /// Multiply every coefficient by *factor* in-place.
    fn scale(&mut self, factor: f64) {
        match &mut self.inner {
            Storage::F64(inner) => {
                let n = inner.len();
                for c in inner.coeffs[..n].iter_mut() {
                    c.scale_real(factor);
                }
            }
            Storage::F32(inner) => {
                let n = inner.len();
                for c in inner.coeffs[..n].iter_mut() {
                    c.scale_real(factor);
                }
            }
        }
    }

    /// Add all terms from *other* into this sum. Both sums must share the same dtype.
    pub fn merge(&mut self, other: &PauliTermSum) -> PyResult<()> {
        match (&mut self.inner, &other.inner) {
            (Storage::F64(dst), Storage::F64(src)) => {
                let n = src.len();
                let mut decoder = RowDecoder::new(src.stride);
                for i in 0..n {
                    let term = decoder.term(src, i);
                    add_raw(dst, &mut self.index, term, *src.coeff(i));
                }
            }
            (Storage::F32(dst), Storage::F32(src)) => {
                let n = src.len();
                let mut decoder = RowDecoder::new(src.stride);
                for i in 0..n {
                    let term = decoder.term(src, i);
                    add_raw(dst, &mut self.index, term, *src.coeff(i));
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "cannot merge PauliTermSums with different dtypes",
                ))
            }
        }
        Ok(())
    }

    /// Stream terms from a file and merge them into this sum one at a time,
    /// accumulating coefficients for strings already present. The file is
    /// always f64; values are cast down if this sum is float32.
    ///
    /// Arguments:
    ///     streamer: A PauliTermStreamer opened with PauliTermStreamer.from_file().
    fn merge_from_file(
        &mut self,
        #[gen_stub(override_type(type_repr = "PauliTermStreamer"))]
        streamer: &mut PauliTermStreamer,
    ) -> PyResult<()> {
        match &mut self.inner {
            Storage::F64(inner) => {
                for result in streamer.inner.by_ref() {
                    let (term, coeff) = result?;
                    add_raw(inner, &mut self.index, term, coeff);
                }
            }
            Storage::F32(inner) => {
                for result in streamer.inner.by_ref() {
                    let (term, coeff) = result?;
                    add_raw(inner, &mut self.index, term, coeff as f32);
                }
            }
        }
        Ok(())
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_self = match &self.inner {
            Storage::F64(inner) => PauliTermSum::from_store(truncate_impl(inner, policy)?),
            Storage::F32(inner) => PauliTermSum::from_store_f32(truncate_impl(inner, policy)?),
        };
        *self = new_self;
        Ok(())
    }

    /// Apply noise damping to every coefficient.
    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        match &mut self.inner {
            Storage::F64(inner) => apply_damping_impl(inner, noise, active_modes),
            Storage::F32(inner) => apply_damping_impl(inner, noise, active_modes),
        }
    }

    /// Bytes of resident sparse key storage held by this term sum.
    ///
    /// Keys only: coefficients and merge metadata are excluded.
    #[getter]
    fn sparse_key_bytes(&self) -> usize {
        match &self.inner {
            Storage::F64(s) => s.sparse_key_bytes(),
            Storage::F32(s) => s.sparse_key_bytes(),
        }
    }

    /// Return the sum of |coefficient|^2 over all terms.
    pub fn norm_squared(&self) -> f64 {
        match &self.inner {
            Storage::F64(inner) => norm_squared_impl(inner),
            Storage::F32(inner) => norm_squared_impl(inner),
        }
    }

    /// Return all (monomial, coefficient) pairs.
    fn items(&self) -> Vec<(PauliString, f64)> {
        match &self.inner {
            Storage::F64(inner) => items_impl(inner),
            Storage::F32(inner) => items_impl(inner),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __setitem__(&mut self, term: PauliString, coeff: f64) {
        match &mut self.inner {
            Storage::F64(inner) => setitem_impl(inner, &mut self.index, term, coeff),
            Storage::F32(inner) => setitem_impl(inner, &mut self.index, term, coeff as f32),
        }
    }

    fn __getitem__(&self, term: &PauliString) -> f64 {
        match &self.inner {
            Storage::F64(inner) => getitem_impl(inner, &self.index, term),
            Storage::F32(inner) => getitem_impl(inner, &self.index, term),
        }
    }

    /// Return a shallow copy of this term sum.
    fn copy(&self) -> PauliTermSum {
        let inner = match &self.inner {
            Storage::F64(s) => Storage::F64(s.copy()),
            Storage::F32(s) => Storage::F32(s.copy()),
        };
        PauliTermSum {
            inner,
            index: self.index.clone(),
        }
    }

    /// Load a PauliTermSum from a gzip-compressed binary file saved by `propagate` or
    /// `expectation_value`. Always loads as float64 (the file format's precision).
    ///
    /// Arguments:
    ///     path: Path to the file written by the `filename` parameter.
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<PauliTermSum> {
        let map = load_terms_from_file::<PauliString>(path)?;
        let n_qubits = map.keys().next().map_or(0, |t| t.n_qubits);
        let mut inner = TermSum::new(n_qubits, PauliBasis::stride_words(n_qubits));
        let mut index = FxHashMap::default();
        index.reserve(map.len());
        for (term, coeff) in map {
            let (gx, gz) = planes_of(&term, inner.stride);
            let row = inner.len();
            inner.push([&gx, &gz], coeff);
            index.insert(term, row);
        }
        Ok(PauliTermSum {
            inner: Storage::F64(inner),
            index,
        })
    }

    /// Save this term sum to a gzip-compressed binary file. Coefficients are
    /// always widened to f64 on disk regardless of in-memory precision.
    ///
    /// Arguments:
    ///     path: Destination file path.
    fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            Storage::F64(inner) => save_terms_to_file(&materialize(inner), path),
            Storage::F32(inner) => save_terms_to_file(&materialize(inner), path),
        }
    }
}

fn truncate_impl<C: CoeffRepr>(
    inner: &TermSum<C>,
    policy: &Bound<'_, PyAny>,
) -> PyResult<TermSum<C>> {
    let n = inner.len();
    let stride = inner.stride;
    let plane_span = inner.plane_span();
    let mut kept = TermSum::new(inner.n_units, stride);

    if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        for i in 0..n {
            let row = inner.row_positions(i);
            let w = PauliBasis::weight_sparse(row, plane_span, inner.n_units);
            let c = inner.coeff(i);
            if wc.is_none_or(|ww| w <= ww) && c.passes_coeff_cutoff(cc) {
                kept.push_positions(row, c.clone());
            }
        }
    } else {
        for i in 0..n {
            let row = inner.row_positions(i);
            let w = PauliBasis::weight_sparse(row, plane_span, inner.n_units);
            let c = inner.coeff(i);
            let should_remove: bool = policy
                .call_method1("should_truncate", (w, c.magnitude()))?
                .extract()?;
            if !should_remove {
                kept.push_positions(row, c.clone());
            }
        }
    }
    Ok(kept)
}

fn apply_damping_impl<C: CoeffRepr>(
    inner: &mut TermSum<C>,
    noise: &Bound<'_, PyAny>,
    active_modes: u32,
) -> PyResult<()> {
    use propaq_core::noise::UniformNoiseModel;
    let n = inner.len();
    let plane_span = inner.plane_span();
    if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
        let d = unm.damping;
        for i in 0..n {
            let w = PauliBasis::weight_sparse(inner.row_positions(i), plane_span, inner.n_units);
            inner.coeffs[i].scale_real((-d * w as f64).exp());
        }
        return Ok(());
    }
    for i in 0..n {
        let w = PauliBasis::weight_sparse(inner.row_positions(i), plane_span, inner.n_units);
        let damping: f64 = noise
            .call_method1("damping_factor", (w, active_modes))?
            .extract()?;
        inner.coeffs[i].scale_real(damping);
    }
    Ok(())
}

fn norm_squared_impl<C: CoeffRepr>(inner: &TermSum<C>) -> f64 {
    inner.coeffs[..inner.len()]
        .iter()
        .map(|c| {
            let v = c.to_f64();
            v * v
        })
        .sum()
}

fn items_impl<C: CoeffRepr>(inner: &TermSum<C>) -> Vec<(PauliString, f64)> {
    let n = inner.len();
    let mut decoder = RowDecoder::new(inner.stride);
    (0..n)
        .map(|i| (decoder.term(inner, i), inner.coeff(i).to_f64()))
        .collect()
}

fn setitem_impl<C: CoeffRepr>(
    inner: &mut TermSum<C>,
    index: &mut FxHashMap<PauliString, usize>,
    term: PauliString,
    coeff: C,
) {
    ensure_sized(inner, term.n_qubits);
    if let Some(&row) = index.get(&term) {
        inner.coeffs[row] = coeff;
        return;
    }
    let (gx, gz) = planes_of(&term, inner.stride);
    let row = inner.len();
    inner.push([&gx, &gz], coeff);
    index.insert(term, row);
}

fn getitem_impl<C: CoeffRepr>(
    inner: &TermSum<C>,
    index: &FxHashMap<PauliString, usize>,
    term: &PauliString,
) -> f64 {
    index
        .get(term)
        .map(|&row| inner.coeff(row).to_f64())
        .unwrap_or_default()
}
