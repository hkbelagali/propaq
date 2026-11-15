//!
//! Shared Python-facing logic for any basis, and the macro that wraps it.
//!
//! A basis crate implements `BasisCodec` (how a term crosses the Python
//! boundary and how gates are lowered to raw words). It then calls
//! `define_basis_module!` to get its `Propagator`, `TermSum` and term
//! streamer classes. All logic lives here, once.
//!

use std::io::Write;
use std::marker::PhantomData;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use num_complex::Complex64;
use pyo3_stub_gen::{PyStubType, TypeInfo};
use rustc_hash::FxHashMap;

use crate::basis::BasisKind;
use crate::bitset::Bitset;
use crate::coeff::CoeffRepr;
use crate::partitioned_termsum::PhaseStats;
use crate::results::PropagationResult;
use crate::run::{run_circuit, BasisAll, GateRecord, RunGate};
use crate::run_config::RunConfig;
use crate::store::{TermBasis, TermSum};
use crate::term_io::{load_terms_from_file, save_terms_to_file};
use crate::traits::AbstractTerm;
use crate::truncators::{resolve_config, ResolvedConfig, TruncationPolicy};

/// How one basis meets Python. Implemented once per basis crate.
pub trait BasisCodec: TermBasis + Sized
where
    Self::Term: AbstractTerm,
{
    /// The engine algebra, available at every dispatched width.
    type Alg: BasisAll;

    /// Name used in the "too wide" error, e.g. "qubits".
    const UNIT_NAME: &'static str;
    /// Widest unit count the engine takes (the error message shows it).
    const MAX_UNITS: usize;
    /// Bits per engine unit, for width dispatch.
    const BITS_PER_UNIT: usize = 2;
    /// What the engine calls a unit, given the term's own size.
    /// `None` means the engine cannot run that size.
    fn engine_units(n_units: usize) -> Option<usize>;

    /// Reads one term from a Python object.
    fn extract_term(obj: &Bound<'_, PyAny>) -> PyResult<Self::Term>;
    /// A term as raw words in the engine layout.
    fn to_words(term: &Self::Term) -> Vec<u64>;
    /// A term from raw engine words.
    fn from_words(words: &[u64], n_units: usize) -> Self::Term;
}


// ------------------------------------------------------------ coefficients

/// A coefficient type a `TermSum` can store, seen from the Python boundary.
pub trait StoreCoeff: CoeffRepr {
    /// True if the type holds a complex value.
    const IS_COMPLEX: bool;
    /// The value as a complex number.
    fn to_complex(&self) -> Complex64;
    /// A coefficient from a complex number. Real types keep the real part.
    fn from_complex(z: Complex64) -> Self;
}

impl StoreCoeff for f64 {
    const IS_COMPLEX: bool = false;
    fn to_complex(&self) -> Complex64 {
        Complex64::new(*self, 0.0)
    }
    fn from_complex(z: Complex64) -> Self {
        z.re
    }
}

impl StoreCoeff for f32 {
    const IS_COMPLEX: bool = false;
    fn to_complex(&self) -> Complex64 {
        Complex64::new(f64::from(*self), 0.0)
    }
    fn from_complex(z: Complex64) -> Self {
        z.re as f32
    }
}

impl StoreCoeff for Complex64 {
    const IS_COMPLEX: bool = true;
    fn to_complex(&self) -> Complex64 {
        *self
    }
    fn from_complex(z: Complex64) -> Self {
        z
    }
}

/// The Python type of a term sum's coefficients: `f64` or `Cx`.
pub trait PyCoeff:
    Copy + Send + 'static + for<'a, 'py> FromPyObject<'a, 'py> + for<'py> IntoPyObject<'py> + PyStubType
{
    /// True if the module stores complex coefficients.
    const COMPLEX: bool;
    fn py_to_complex(self) -> Complex64;
    fn py_from_complex(z: Complex64) -> Self;
}

impl PyCoeff for f64 {
    const COMPLEX: bool = false;
    fn py_to_complex(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn py_from_complex(z: Complex64) -> Self {
        z.re
    }
}

/// A complex coefficient as a Python `complex`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cx(pub Complex64);

impl<'a, 'py> FromPyObject<'a, 'py> for Cx {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        obj.extract::<Complex64>().map(Cx)
    }
}

impl<'py> IntoPyObject<'py> for Cx {
    type Target = pyo3::types::PyComplex;
    type Output = Bound<'py, pyo3::types::PyComplex>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        self.0.into_pyobject(py)
    }
}

impl PyStubType for Cx {
    fn type_output() -> TypeInfo {
        TypeInfo::builtin("complex")
    }
    fn type_input() -> TypeInfo {
        TypeInfo::builtin("complex")
    }
}

impl PyCoeff for Cx {
    const COMPLEX: bool = true;
    fn py_to_complex(self) -> Complex64 {
        self.0
    }
    fn py_from_complex(z: Complex64) -> Self {
        Cx(z)
    }
}

// ---------------------------------------------------------------- storage

/// Backing storage for a term sum.
pub enum Storage {
    F64(TermSum<f64>),
    F32(TermSum<f32>),
    /// Complex coefficients, for bases whose coefficients leave the reals.
    C64(TermSum<Complex64>),
}

impl Storage {
    pub fn len(&self) -> usize {
        match self {
            Storage::F64(s) => s.len(),
            Storage::F32(s) => s.len(),
            Storage::C64(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn n_units(&self) -> usize {
        match self {
            Storage::F64(s) => s.n_units,
            Storage::F32(s) => s.n_units,
            Storage::C64(s) => s.n_units,
        }
    }
}

/// A mutable, weighted sum of terms of basis `B`.
pub struct GenericTermSum<B: BasisCodec>
where
    B::Term: AbstractTerm,
{
    pub inner: Storage,
    index: FxHashMap<B::Term, usize>,
    _basis: PhantomData<fn() -> B>,
}

/// Maps a `dtype` string to a storage kind. The coefficient type `K` of the
/// module decides which names are valid.
fn parse_dtype<K: PyCoeff>(dtype: Option<&str>) -> PyResult<&'static str> {
    match (K::COMPLEX, dtype) {
        (false, None | Some("float64")) => Ok("float64"),
        (false, Some("float32")) => Ok("float32"),
        (true, None | Some("complex128")) => Ok("complex128"),
        (_, Some(other)) => Err(PyValueError::new_err(format!("unknown dtype: {other}"))),
    }
}

fn units_of<B: BasisCodec>(term: &B::Term) -> usize
where
    B::Term: AbstractTerm,
{
    term.system_size() as usize
}


fn ensure_sized<B: BasisCodec, C: CoeffRepr>(inner: &mut TermSum<C>, n_units: usize)
where
    B::Term: AbstractTerm,
{
    if inner.is_empty() && inner.n_units != n_units {
        *inner = TermSum::new(n_units, B::stride_words(n_units));
    }
}

fn planes_of<B: BasisCodec>(term: &B::Term, stride: usize) -> (Vec<u64>, Vec<u64>)
where
    B::Term: AbstractTerm,
{
    let mut a = vec![0u64; stride];
    let mut b = vec![0u64; stride];
    B::term_into_planes(term, units_of::<B>(term), [&mut a, &mut b]);
    (a, b)
}

fn add_raw<B: BasisCodec, C: CoeffRepr>(
    inner: &mut TermSum<C>,
    index: &mut FxHashMap<B::Term, usize>,
    term: B::Term,
    coeff: C,
) where
    B::Term: AbstractTerm,
{
    ensure_sized::<B, C>(inner, units_of::<B>(&term));
    if let Some(&row) = index.get(&term) {
        inner.coeffs[row].add_assign(coeff);
        return;
    }
    let (a, b) = planes_of::<B>(&term, inner.stride);
    let row = inner.len();
    inner.push([&a, &b], coeff);
    index.insert(term, row);
}

/// Builds a term sum from the (key, coefficient) pairs the engine produces.
/// Keys are distinct there, but `add_raw` folds duplicates anyway.
pub fn term_sum_from_pairs<B: BasisCodec, C: CoeffRepr>(
    pairs: Vec<(B::Term, C)>,
    n_units: usize,
) -> TermSum<C>
where
    B::Term: AbstractTerm,
{
    let mut inner = TermSum::<C>::new(n_units, B::stride_words(n_units));
    let mut index = FxHashMap::default();
    index.reserve(pairs.len());
    for (term, coeff) in pairs {
        add_raw::<B, C>(&mut inner, &mut index, term, coeff);
    }
    inner
}

/// A one-row decode buffer, reused across rows.
struct RowDecoder {
    buf: Vec<u64>,
}

impl RowDecoder {
    fn new(stride: usize) -> Self {
        RowDecoder { buf: vec![0u64; 2 * stride] }
    }

    fn term<B: BasisCodec, C: CoeffRepr>(&mut self, terms: &TermSum<C>, i: usize) -> B::Term
    where
        B::Term: AbstractTerm,
    {
        B::term_from_planes(terms.decode_row(i, &mut self.buf), terms.n_units)
    }
}

/// The flat map the file I/O understands.
pub fn materialize<B: BasisCodec, C: CoeffRepr>(terms: &TermSum<C>) -> FxHashMap<B::Term, f64>
where
    B::Term: AbstractTerm,
{
    let mut map = FxHashMap::default();
    map.reserve(terms.len());
    let mut decoder = RowDecoder::new(terms.stride);
    for i in 0..terms.len() {
        map.insert(decoder.term::<B, C>(terms, i), terms.coeff(i).to_f64());
    }
    map
}

fn truncate_impl<B: BasisCodec, C: CoeffRepr>(
    inner: &TermSum<C>,
    policy: &Bound<'_, PyAny>,
) -> PyResult<TermSum<C>>
where
    B::Term: AbstractTerm,
{
    let n = inner.len();
    let plane_span = inner.plane_span();
    let n_units = inner.n_units;
    let mut kept = TermSum::new(n_units, inner.stride);

    if let Ok(tp) = policy.extract::<PyRef<TruncationPolicy>>() {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        for i in 0..n {
            let row = inner.row_positions(i);
            let w = B::weight_sparse(row, plane_span, n_units);
            let c = inner.coeff(i);
            if wc.is_none_or(|ww| w <= ww) && c.passes_coeff_cutoff(cc) {
                kept.push_positions(row, c.clone());
            }
        }
    } else {
        for i in 0..n {
            let row = inner.row_positions(i);
            let w = B::weight_sparse(row, plane_span, n_units);
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

fn apply_damping_impl<B: BasisCodec, C: CoeffRepr>(
    inner: &mut TermSum<C>,
    noise: &Bound<'_, PyAny>,
    active_modes: u32,
) -> PyResult<()>
where
    B::Term: AbstractTerm,
{
    use crate::noise::UniformNoiseModel;
    let n = inner.len();
    let plane_span = inner.plane_span();
    let n_units = inner.n_units;
    if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
        let d = unm.damping;
        for i in 0..n {
            let w = B::weight_sparse(inner.row_positions(i), plane_span, n_units);
            inner.coeffs[i].scale_real((-d * w as f64).exp());
        }
        return Ok(());
    }
    for i in 0..n {
        let w = B::weight_sparse(inner.row_positions(i), plane_span, n_units);
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
            let v = c.magnitude();
            v * v
        })
        .sum()
}

fn items_impl<B: BasisCodec, C: StoreCoeff, K: PyCoeff>(inner: &TermSum<C>) -> Vec<(B::Term, K)>
where
    B::Term: AbstractTerm,
{
    let mut decoder = RowDecoder::new(inner.stride);
    (0..inner.len())
        .map(|i| (decoder.term::<B, C>(inner, i), K::py_from_complex(inner.coeff(i).to_complex())))
        .collect()
}

/// The (term, coefficient) pairs of a stored sum, with the stored coefficient type.
pub fn pairs_of<B: BasisCodec, C: CoeffRepr>(inner: &TermSum<C>) -> Vec<(B::Term, C)>
where
    B::Term: AbstractTerm,
{
    let mut decoder = RowDecoder::new(inner.stride);
    (0..inner.len())
        .map(|i| (decoder.term::<B, C>(inner, i), inner.coeff(i).clone()))
        .collect()
}

fn setitem_impl<B: BasisCodec, C: CoeffRepr>(
    inner: &mut TermSum<C>,
    index: &mut FxHashMap<B::Term, usize>,
    term: B::Term,
    coeff: C,
) where
    B::Term: AbstractTerm,
{
    ensure_sized::<B, C>(inner, units_of::<B>(&term));
    if let Some(&row) = index.get(&term) {
        inner.coeffs[row] = coeff;
        return;
    }
    let (a, b) = planes_of::<B>(&term, inner.stride);
    let row = inner.len();
    inner.push([&a, &b], coeff);
    index.insert(term, row);
}

fn getitem_impl<B: BasisCodec, C: StoreCoeff, K: PyCoeff>(
    inner: &TermSum<C>,
    index: &FxHashMap<B::Term, usize>,
    term: &B::Term,
) -> K
where
    B::Term: AbstractTerm,
{
    index
        .get(term)
        .map(|&row| K::py_from_complex(inner.coeff(row).to_complex()))
        .unwrap_or_else(|| K::py_from_complex(Complex64::new(0.0, 0.0)))
}

fn fill_index<B: BasisCodec, C: CoeffRepr>(s: &TermSum<C>) -> FxHashMap<B::Term, usize>
where
    B::Term: AbstractTerm,
{
    let mut index = FxHashMap::default();
    index.reserve(s.len());
    let mut decoder = RowDecoder::new(s.stride);
    for i in 0..s.len() {
        index.insert(decoder.term::<B, C>(s, i), i);
    }
    index
}

fn no_files_for_complex() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "term files hold real coefficients; complex term sums cannot be saved or streamed",
    )
}

impl<B: BasisCodec> GenericTermSum<B>
where
    B::Term: AbstractTerm,
{
    pub fn from_store(inner: TermSum<f64>) -> Self {
        Self::from_storage(Storage::F64(inner))
    }

    /// Same as `from_store`, for an f32-backed `TermSum`.
    pub fn from_store_f32(inner: TermSum<f32>) -> Self {
        Self::from_storage(Storage::F32(inner))
    }

    /// Same as `from_store`, for a complex-backed `TermSum`.
    pub fn from_store_c64(inner: TermSum<Complex64>) -> Self {
        Self::from_storage(Storage::C64(inner))
    }

    pub fn from_storage(inner: Storage) -> Self {
        let index = match &inner {
            Storage::F64(s) => fill_index::<B, f64>(s),
            Storage::F32(s) => fill_index::<B, f32>(s),
            Storage::C64(s) => fill_index::<B, Complex64>(s),
        };
        GenericTermSum { inner, index, _basis: PhantomData }
    }

    /// Number of units (precision-independent).
    pub fn n_units(&self) -> usize {
        self.inner.n_units()
    }

    /// Coefficients widened to f64, regardless of storage precision.
    /// Complex coefficients lose their imaginary part.
    pub fn as_f64(&self) -> TermSum<f64> {
        match &self.inner {
            Storage::F64(s) => s.map_coeffs(|c| *c),
            Storage::F32(s) => s.map_coeffs(|c| *c as f64),
            Storage::C64(s) => s.map_coeffs(|c| c.re),
        }
    }

    /// Builds a term sum from a Python dict of term to coefficient.
    pub fn from_dict<K: PyCoeff>(
        terms: Option<&Bound<'_, PyDict>>,
        dtype: Option<&str>,
    ) -> PyResult<Self> {
        fn fill<B: BasisCodec, C: StoreCoeff, K: PyCoeff>(
            terms: Option<&Bound<'_, PyDict>>,
            index: &mut FxHashMap<B::Term, usize>,
        ) -> PyResult<TermSum<C>>
        where
            B::Term: AbstractTerm,
        {
            let mut inner = TermSum::<C>::new(0, B::stride_words(0));
            if let Some(dict) = terms {
                index.reserve(dict.len());
                for (k, v) in dict.iter() {
                    let key = B::extract_term(&k)?;
                    let val: K = v.extract().map_err(Into::into)?;
                    add_raw::<B, C>(&mut inner, index, key, C::from_complex(val.py_to_complex()));
                }
            }
            Ok(inner)
        }
        let mut index = FxHashMap::default();
        let inner = match parse_dtype::<K>(dtype)? {
            "float32" => Storage::F32(fill::<B, f32, K>(terms, &mut index)?),
            "complex128" => Storage::C64(fill::<B, Complex64, K>(terms, &mut index)?),
            _ => Storage::F64(fill::<B, f64, K>(terms, &mut index)?),
        };
        Ok(GenericTermSum { inner, index, _basis: PhantomData })
    }

    pub fn dtype(&self) -> &'static str {
        match &self.inner {
            Storage::F64(_) => "float64",
            Storage::F32(_) => "float32",
            Storage::C64(_) => "complex128",
        }
    }

    pub fn add<K: PyCoeff>(&mut self, term: B::Term, coeff: K) {
        let z = coeff.py_to_complex();
        match &mut self.inner {
            Storage::F64(inner) => add_raw::<B, f64>(inner, &mut self.index, term, f64::from_complex(z)),
            Storage::F32(inner) => add_raw::<B, f32>(inner, &mut self.index, term, f32::from_complex(z)),
            Storage::C64(inner) => add_raw::<B, Complex64>(inner, &mut self.index, term, z),
        }
    }

    pub fn scale(&mut self, factor: f64) {
        match &mut self.inner {
            Storage::F64(inner) => {
                let n = inner.len();
                inner.coeffs[..n].iter_mut().for_each(|c| c.scale_real(factor));
            }
            Storage::F32(inner) => {
                let n = inner.len();
                inner.coeffs[..n].iter_mut().for_each(|c| c.scale_real(factor));
            }
            Storage::C64(inner) => {
                let n = inner.len();
                inner.coeffs[..n].iter_mut().for_each(|c| c.scale_real(factor));
            }
        }
    }

    /// `merge`. `type_name` appears in the dtype-mismatch error.
    pub fn merge(&mut self, other: &Self, type_name: &str) -> PyResult<()> {
        fn merge_rows<B: BasisCodec, C: CoeffRepr>(
            dst: &mut TermSum<C>,
            index: &mut FxHashMap<B::Term, usize>,
            src: &TermSum<C>,
        ) where
            B::Term: AbstractTerm,
        {
            let mut decoder = RowDecoder::new(src.stride);
            for i in 0..src.len() {
                let term = decoder.term::<B, C>(src, i);
                add_raw::<B, C>(dst, index, term, src.coeff(i).clone());
            }
        }
        match (&mut self.inner, &other.inner) {
            (Storage::F64(dst), Storage::F64(src)) => merge_rows::<B, f64>(dst, &mut self.index, src),
            (Storage::F32(dst), Storage::F32(src)) => merge_rows::<B, f32>(dst, &mut self.index, src),
            (Storage::C64(dst), Storage::C64(src)) => {
                merge_rows::<B, Complex64>(dst, &mut self.index, src)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "cannot merge {type_name}s with different dtypes"
                )))
            }
        }
        Ok(())
    }

    /// Merges a stream of terms, one at a time.
    pub fn merge_stream(
        &mut self,
        stream: impl Iterator<Item = PyResult<(B::Term, f64)>>,
    ) -> PyResult<()> {
        match &mut self.inner {
            Storage::F64(inner) => {
                for result in stream {
                    let (term, coeff) = result?;
                    add_raw::<B, f64>(inner, &mut self.index, term, coeff);
                }
            }
            Storage::F32(inner) => {
                for result in stream {
                    let (term, coeff) = result?;
                    add_raw::<B, f32>(inner, &mut self.index, term, coeff as f32);
                }
            }
            Storage::C64(_) => return Err(no_files_for_complex()),
        }
        Ok(())
    }

    pub fn truncate(&mut self, policy: &Bound<'_, PyAny>) -> PyResult<()> {
        let new_self = match &self.inner {
            Storage::F64(inner) => Self::from_store(truncate_impl::<B, f64>(inner, policy)?),
            Storage::F32(inner) => Self::from_store_f32(truncate_impl::<B, f32>(inner, policy)?),
            Storage::C64(inner) => Self::from_store_c64(truncate_impl::<B, Complex64>(inner, policy)?),
        };
        *self = new_self;
        Ok(())
    }

    pub fn apply_damping(&mut self, noise: &Bound<'_, PyAny>, active_modes: u32) -> PyResult<()> {
        match &mut self.inner {
            Storage::F64(inner) => apply_damping_impl::<B, f64>(inner, noise, active_modes),
            Storage::F32(inner) => apply_damping_impl::<B, f32>(inner, noise, active_modes),
            Storage::C64(inner) => apply_damping_impl::<B, Complex64>(inner, noise, active_modes),
        }
    }

    pub fn sparse_key_bytes(&self) -> usize {
        match &self.inner {
            Storage::F64(s) => s.sparse_key_bytes(),
            Storage::F32(s) => s.sparse_key_bytes(),
            Storage::C64(s) => s.sparse_key_bytes(),
        }
    }

    pub fn norm_squared(&self) -> f64 {
        match &self.inner {
            Storage::F64(inner) => norm_squared_impl(inner),
            Storage::F32(inner) => norm_squared_impl(inner),
            Storage::C64(inner) => norm_squared_impl(inner),
        }
    }

    pub fn items<K: PyCoeff>(&self) -> Vec<(B::Term, K)> {
        match &self.inner {
            Storage::F64(inner) => items_impl::<B, f64, K>(inner),
            Storage::F32(inner) => items_impl::<B, f32, K>(inner),
            Storage::C64(inner) => items_impl::<B, Complex64, K>(inner),
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn set_item<K: PyCoeff>(&mut self, term: B::Term, coeff: K) {
        let z = coeff.py_to_complex();
        match &mut self.inner {
            Storage::F64(inner) => setitem_impl::<B, f64>(inner, &mut self.index, term, f64::from_complex(z)),
            Storage::F32(inner) => setitem_impl::<B, f32>(inner, &mut self.index, term, f32::from_complex(z)),
            Storage::C64(inner) => setitem_impl::<B, Complex64>(inner, &mut self.index, term, z),
        }
    }

    pub fn get_item<K: PyCoeff>(&self, term: &B::Term) -> K {
        match &self.inner {
            Storage::F64(inner) => getitem_impl::<B, f64, K>(inner, &self.index, term),
            Storage::F32(inner) => getitem_impl::<B, f32, K>(inner, &self.index, term),
            Storage::C64(inner) => getitem_impl::<B, Complex64, K>(inner, &self.index, term),
        }
    }

    pub fn copy(&self) -> Self {
        let inner = match &self.inner {
            Storage::F64(s) => Storage::F64(s.copy()),
            Storage::F32(s) => Storage::F32(s.copy()),
            Storage::C64(s) => Storage::C64(s.copy()),
        };
        GenericTermSum { inner, index: self.index.clone(), _basis: PhantomData }
    }

    pub fn from_file(path: &str) -> PyResult<Self> {
        let map = load_terms_from_file::<B::Term>(path)?;
        let n_units = map.keys().next().map_or(0, |t| units_of::<B>(t));
        let mut inner = TermSum::new(n_units, B::stride_words(n_units));
        let mut index = FxHashMap::default();
        index.reserve(map.len());
        for (term, coeff) in map {
            let (a, b) = planes_of::<B>(&term, inner.stride);
            let row = inner.len();
            inner.push([&a, &b], coeff);
            index.insert(term, row);
        }
        Ok(GenericTermSum { inner: Storage::F64(inner), index, _basis: PhantomData })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        match &self.inner {
            Storage::F64(inner) => save_terms_to_file(&materialize::<B, f64>(inner), path),
            Storage::F32(inner) => save_terms_to_file(&materialize::<B, f32>(inner), path),
            Storage::C64(_) => Err(no_files_for_complex()),
        }
    }
}

// ------------------------------------------------------------- propagation

/// The result of a run with terms as `B::Term`.
pub struct TermRunOutput<T, C> {
    pub result: PropagationResult,
    pub gates: Vec<GateRecord>,
    pub terms: Option<Vec<(T, C)>>,
    pub phases: PhaseStats,
}

/// Reads a circuit's layers into lowered gates in application order.
fn extract_layers<B: BasisCodec, C: CoeffRepr>(
    py: Python<'_>,
    circuit: &Bound<'_, PyAny>,
) -> PyResult<Vec<Vec<RunGate<C>>>>
where
    B::Term: AbstractTerm,
{
    let layers: Vec<Vec<Py<PyAny>>> = circuit.getattr("layers")?.extract()?;
    let mut out = Vec::with_capacity(layers.len());
    for layer in layers.iter().rev() {
        let mut gates = Vec::with_capacity(layer.len());
        for rot_obj in layer.iter().rev() {
            let rot = rot_obj.bind(py);
            let generator = B::extract_term(&rot.getattr("generator")?)?;
            let angle = C::extract_gate_param(rot)?;
            let qiskit_gate_idx = rot
                .getattr("qiskit_gate_idx")
                .ok()
                .and_then(|v| v.extract::<Option<usize>>().ok())
                .flatten();
            gates.push(RunGate { generator: B::to_words(&generator), angle, qiskit_gate_idx });
        }
        out.push(gates);
    }
    Ok(out)
}

/// Runs `observable` through `circuit` on the partitioned engine.
#[allow(clippy::too_many_arguments)]
pub fn run_basis<B: BasisCodec, C: CoeffRepr>(
    py: Python<'_>,
    observable: &[(B::Term, C)],
    circuit: &Bound<'_, PyAny>,
    diag_state: Option<&[u64]>,
    n_units: usize,
    cfg: &ResolvedConfig,
    pool: &rayon::ThreadPool,
    n_threads: Option<usize>,
    noise: Option<&Bound<'_, PyAny>>,
    collect_counts: bool,
    want_terms: bool,
    log_gates: bool,
    progress_bar: bool,
    progress_every: usize,
) -> PyResult<Option<TermRunOutput<B::Term, C>>>
where
    B::Term: AbstractTerm,
{
    let Some(engine_units) = B::engine_units(n_units) else {
        return Ok(None);
    };
    if engine_units > B::MAX_UNITS {
        return Ok(None);
    }
    let layers = extract_layers::<B, C>(py, circuit)?;
    let obs: Vec<(Vec<u64>, C)> =
        observable.iter().map(|(t, c)| (B::to_words(t), c.clone())).collect();
    let out = run_circuit::<B::Alg, C>(
        py,
        &obs,
        &layers,
        diag_state,
        engine_units,
        B::BITS_PER_UNIT,
        cfg,
        pool,
        n_threads,
        noise,
        collect_counts,
        want_terms,
        log_gates,
        progress_bar,
        progress_every,
    )?;
    Ok(out.map(|o| TermRunOutput {
        result: o.result,
        gates: o.gates,
        phases: o.phases,
        terms: o.terms.map(|ts| {
            ts.into_iter().map(|(w, c)| (B::from_words(&w, n_units), c)).collect()
        }),
    }))
}

/// Replays a run's per-gate records into the verbose log.
pub fn write_gate_log(
    log_filename: Option<&str>,
    log_every: usize,
    cfg: &ResolvedConfig,
    gates: &[GateRecord],
    phases: &PhaseStats,
) -> PyResult<()> {
    let Some(filename) = log_filename else {
        return Ok(());
    };
    let f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut log = std::io::BufWriter::new(f);
    let every = log_every.max(1);
    let wc = cfg.weight.map_or_else(|| "null".to_string(), |w| w.to_string());
    let cc = cfg.coefficient.unwrap_or(0.0);
    for g in gates.iter().filter(|g| g.gate_idx % every == 0) {
        let qki = g
            .qiskit_gate_idx
            .map_or_else(|| "null".to_string(), |v| v.to_string());
        let _ = writeln!(
            log,
            r#"{{"event":"gate","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"terms":{},"ms_per_gate":{:.3e}}}"#,
            g.gate_idx, g.layer_idx, qki, g.terms_after, g.elapsed_ms
        );
        let _ = writeln!(
            log,
            r#"{{"event":"truncation","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":{},"trigger":"emit","truncation_model":"emit_gate","terms_before":{},"terms_after":{},"terms_gained":{},"terms_discarded":{},"discarded_coeff_l1":{:.6e},"discarded_coeff_max":{:.6e},"weight_cutoff":{},"coeff_cutoff":{:.6e},"elapsed_ms":{:.3e}}}"#,
            g.gate_idx,
            g.layer_idx,
            qki,
            g.terms_before,
            g.terms_after,
            g.terms_gained,
            g.declined,
            g.discarded_coeff_l1,
            g.discarded_coeff_max,
            wc,
            cc,
            g.elapsed_ms
        );
    }
    write_phase_event(&mut log, phases);
    Ok(())
}

/// One closing record with the run's phase split and kernel counters.
fn write_phase_event(log: &mut impl Write, p: &PhaseStats) {
    let occupancy = |busy: f64, wall: f64| busy / (wall.max(1e-12) * p.partitions as f64);
    let share = |part: u64, whole: u64| part as f64 / whole.max(1) as f64;
    let _ = writeln!(
        log,
        r#"{{"event":"engine_phases","partitions":{},"scan_s":{:.6e},"absorb_s":{:.6e},"claims_s":{:.6e},"scan_occupancy":{:.4},"absorb_occupancy":{:.4},"claims_occupancy":{:.4},"terms":{},"inline_positions":{},"overflow_rows":{},"overflow_share":{:.4},"visited":{},"emitted":{},"declined":{},"emitted_share":{:.4},"declined_share":{:.4},"exchange_hits":{},"exchange_hit_share":{:.4}}}"#,
        p.partitions,
        p.scan_seconds,
        p.absorb_seconds,
        p.claims_seconds,
        occupancy(p.scan_busy_seconds, p.scan_seconds),
        occupancy(p.absorb_busy_seconds, p.absorb_seconds),
        occupancy(p.claims_busy_seconds, p.claims_seconds),
        p.terms,
        p.inline_positions,
        p.overflow_rows,
        p.overflow_rows as f64 / p.terms.max(1) as f64,
        p.visited,
        p.emitted,
        p.declined,
        share(p.emitted, p.visited),
        share(p.declined, p.visited),
        p.exchange_hits,
        share(p.exchange_hits, p.emitted),
    );
}

fn too_wide<B: BasisCodec>(n_units: usize) -> PyErr
where
    B::Term: AbstractTerm,
{
    PyValueError::new_err(format!(
        "propaq: {} {} exceeds the engine's maximum of {}",
        n_units,
        B::UNIT_NAME,
        B::MAX_UNITS
    ))
}

/// Runs `observable` through `circuit` on one storage type. Returns the
/// evolved terms and the run's records, or `None` if the key is too wide.
#[allow(clippy::type_complexity)]
fn propagate_typed<B: BasisCodec, C: StoreCoeff>(
    config: &RunConfig,
    py: Python<'_>,
    src: &TermSum<C>,
    circuit: &Bound<'_, PyAny>,
    n: usize,
    cfg: &ResolvedConfig,
) -> PyResult<Option<(TermSum<C>, Vec<GateRecord>, PhaseStats)>>
where
    B::Term: AbstractTerm,
{
    let pool = &config.pool;
    let threads = pool.current_num_threads().into();
    let noise = config.noise.as_ref().map(|x| x.bind(py));
    let log = config.log_filename.is_some();
    let terms = pairs_of::<B, C>(src);
    Ok(run_basis::<B, C>(
        py, &terms, circuit, None, n, cfg, pool, threads, noise, false, true, log,
        config.progress_bar, config.progress_every,
    )?
    .and_then(|o| {
        let (recs, phases) = (o.gates, o.phases);
        o.terms.map(|t| (term_sum_from_pairs::<B, C>(t, n), recs, phases))
    }))
}

/// `propagate`: back-propagates `observable` through `circuit`.
pub fn propagate<B: BasisCodec>(
    config: &RunConfig,
    py: Python<'_>,
    observable: &GenericTermSum<B>,
    circuit: &Bound<'_, PyAny>,
    filename: Option<String>,
) -> PyResult<GenericTermSum<B>>
where
    B::Term: AbstractTerm,
{
    if filename.is_some() && matches!(observable.inner, Storage::C64(_)) {
        return Err(no_files_for_complex());
    }
    let cfg = resolve_config(&config.truncators);
    let n = observable.n_units();
    let evolved = match &observable.inner {
        Storage::F64(src) => propagate_typed::<B, f64>(config, py, src, circuit, n, &cfg)?
            .map(|(s, r, p)| (Storage::F64(s), r, p)),
        Storage::F32(src) => propagate_typed::<B, f32>(config, py, src, circuit, n, &cfg)?
            .map(|(s, r, p)| (Storage::F32(s), r, p)),
        Storage::C64(src) => propagate_typed::<B, Complex64>(config, py, src, circuit, n, &cfg)?
            .map(|(s, r, p)| (Storage::C64(s), r, p)),
    };
    let Some((storage, recs, phases)) = evolved else {
        return Err(too_wide::<B>(n));
    };
    write_gate_log(config.log_filename.as_deref(), config.log_every, &cfg, &recs, &phases)?;
    let evolved = GenericTermSum::<B>::from_storage(storage);
    if let Some(path) = filename.as_deref() {
        evolved.save(path)?;
    }
    Ok(evolved)
}

/// One expectation run on one storage type.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn expectation_typed<B: BasisCodec, C: StoreCoeff>(
    config: &RunConfig,
    py: Python<'_>,
    src: &TermSum<C>,
    circuit: &Bound<'_, PyAny>,
    initial_state: &Bitset,
    n: usize,
    cfg: &ResolvedConfig,
    want_terms: bool,
) -> PyResult<Option<(PropagationResult, Vec<GateRecord>, PhaseStats, Option<TermSum<C>>)>>
where
    B::Term: AbstractTerm,
{
    let pool = &config.pool;
    let threads = pool.current_num_threads().into();
    let log = config.log_filename.is_some();
    let noise = config.noise.as_ref().map(|x| x.bind(py));
    let terms = pairs_of::<B, C>(src);
    Ok(run_basis::<B, C>(
        py, &terms, circuit, Some(initial_state.as_words()), n, cfg, pool, threads, noise, true,
        want_terms, log, config.progress_bar, config.progress_every,
    )?
    .map(|o| (o.result, o.gates, o.phases, o.terms.map(|t| term_sum_from_pairs::<B, C>(t, n)))))
}

/// `expectation_value`. For complex storage the value is the real part.
pub fn expectation_value<B: BasisCodec>(
    config: &RunConfig,
    py: Python<'_>,
    observable: &GenericTermSum<B>,
    circuit: &Bound<'_, PyAny>,
    initial_state: Bitset,
    filename: Option<String>,
) -> PyResult<PropagationResult>
where
    B::Term: AbstractTerm,
{
    // `filename` asks for the evolved operator on disk as well as the
    // expectation value, so the run has to hand its terms back for it.
    let want_terms = filename.is_some();
    if want_terms && matches!(observable.inner, Storage::C64(_)) {
        return Err(no_files_for_complex());
    }
    let cfg = resolve_config(&config.truncators);
    let n = observable.n_units();
    let st = &initial_state;
    let served = match &observable.inner {
        Storage::F64(s) => expectation_typed::<B, f64>(config, py, s, circuit, st, n, &cfg, want_terms)?
            .map(|(r, g, p, t)| (r, g, p, t.map(Storage::F64))),
        Storage::F32(s) => expectation_typed::<B, f32>(config, py, s, circuit, st, n, &cfg, want_terms)?
            .map(|(r, g, p, t)| (r, g, p, t.map(Storage::F32))),
        Storage::C64(s) => {
            expectation_typed::<B, Complex64>(config, py, s, circuit, st, n, &cfg, want_terms)?
                .map(|(r, g, p, t)| (r, g, p, t.map(Storage::C64)))
        }
    };
    if let Some((result, recs, phases, terms)) = served {
        write_gate_log(config.log_filename.as_deref(), config.log_every, &cfg, &recs, &phases)?;
        if let (Some(path), Some(storage)) = (filename.as_deref(), terms) {
            GenericTermSum::<B>::from_storage(storage).save(path)?;
        }
        return Ok(result);
    }
    Err(too_wide::<B>(n))
}


pub fn codec_kind<B: BasisCodec>() -> BasisKind
where
    B::Term: AbstractTerm,
{
    <B::Alg as crate::basis::Basis<1>>::KIND
}

/// Defines the Python classes of one basis: a propagator, a term sum and a
/// term streamer. All logic is in core. The caller needs `pyo3` and
/// `pyo3-stub-gen` as dependencies.
///
/// ```ignore
/// propaq_core::define_basis_module! {
///     codec: MyCodec,           // implements BasisCodec
///     term: MyTerm,             // the Python term class (== MyCodec::Term)
///     coeff: f64,               // Python coefficient type: f64, or `Cx` for complex sums
///     propagator: MyPropagator,
///     termsum: MyTermSum,
///     streamer: MyTermStreamer,
///     streamer_repr: "MyTermStreamer",
///     state_doc: "Reference state as a bitstring integer.",
///     propagator_docs: { /// Class doc lines. },
///     termsum_docs: { /// Class doc lines. },
/// }
/// ```
#[macro_export]
macro_rules! define_basis_classes {
    (
        codec: $B:ty,
        term: $T:ty,
        coeff: $K:ty,
        propagator: $P:ident,
        termsum: $S:ident,
        streamer: $St:ident,
        streamer_repr: $sr:literal,
        state_doc: $state:literal,
        obs_doc: $obs:literal,
        circuit_doc: $circ:literal,
        add_doc: $adddoc:literal,
        item_doc: $item:literal,
        sum_doc: $sumname:literal,
        file_item_doc: $fitem:literal,
        streamer_doc: $stname:literal,
        propagator_docs: { $(#[$pdoc:meta])* },
        termsum_docs: { $(#[$sdoc:meta])* } $(,)?
    ) => {
        $(#[$pdoc])*
        #[::pyo3_stub_gen::derive::gen_stub_pyclass]
        #[::pyo3::pyclass(module = "propaq._rust_core")]
        pub struct $P {
            inner: $crate::run_config::RunConfig,
        }

        #[::pyo3_stub_gen::derive::gen_stub_pymethods]
        #[::pyo3::pymethods]
        impl $P {
            /// Initialize the propagator. See the class docstring for arguments.
            #[new]
            #[pyo3(signature = (noise=None, truncation=None, n_threads=None, logger=None, pin_threads=true, progress_bar=false, progress_every=1))]
            fn new(
                noise: Option<::pyo3::Py<::pyo3::PyAny>>,
                truncation: Option<::pyo3::Bound<'_, ::pyo3::PyAny>>,
                n_threads: Option<usize>,
                logger: Option<::pyo3::Py<::pyo3::PyAny>>,
                pin_threads: bool,
                progress_bar: bool,
                progress_every: usize,
            ) -> ::pyo3::PyResult<Self> {
                let truncators = $crate::truncators::resolve_truncation(truncation.as_ref())?;
                $crate::truncators::reject_surrogate_only(&truncators)?;
                Ok($P {
                    inner: $crate::run_config::RunConfig::new(
                        noise,
                        truncators,
                        n_threads,
                        logger,
                        pin_threads,
                        progress_bar,
                        progress_every,
                    )?,
                })
            }

            /// Back-propagate *circuit* through *observable*, returning the evolved term sum.
            ///
            /// Arguments:
            #[doc = concat!("    observable: The ", $obs, " to back-propagate.")]
            #[doc = concat!("    circuit: A ", $circ, " whose rotations are applied in reverse.")]
            ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
            #[pyo3(signature = (observable, circuit, filename=None))]
            fn propagate(
                &mut self,
                py: ::pyo3::Python<'_>,
                observable: &$S,
                circuit: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                filename: Option<String>,
            ) -> ::pyo3::PyResult<$S> {
                $crate::basis_module::propagate::<$B>(
                    &self.inner,
                    py,
                    &observable.core,
                    circuit,
                    filename,
                )
                .map(|core| $S { core })
            }

            /// Compute the expectation value of *observable* in the state prepared by *circuit*.
            ///
            /// Arguments:
            #[doc = concat!("    observable: The ", $obs, ".")]
            #[doc = concat!("    circuit: A ", $circ, " applied to the reference state.")]
            #[doc = $state]
            ///     filename: If given, save the final terms to a gzip-compressed binary file at this path.
            #[pyo3(signature = (observable, circuit, initial_state=None, filename=None))]
            fn expectation_value(
                &mut self,
                py: ::pyo3::Python<'_>,
                observable: &$S,
                circuit: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                initial_state: Option<&::pyo3::Bound<'_, ::pyo3::PyAny>>,
                filename: Option<String>,
            ) -> ::pyo3::PyResult<$crate::results::PropagationResult> {
                let initial_state = match initial_state {
                    Some(v) => $crate::helpers::pyint_to_bitset(v, observable.core.n_units())?,
                    None => $crate::bitset::Bitset::zero(),
                };
                $crate::basis_module::expectation_value::<$B>(
                    &self.inner,
                    py,
                    &observable.core,
                    circuit,
                    initial_state,
                    filename,
                )
            }

            /// The noise model used during propagation, if any.
            #[getter]
            fn noise(&self, py: ::pyo3::Python<'_>) -> Option<::pyo3::Py<::pyo3::PyAny>> {
                self.inner.noise.as_ref().map(|n| n.clone_ref(py))
            }

            /// Set the noise model for this propagator.
            #[pyo3(signature = (noise=None))]
            fn set_noise(&mut self, noise: Option<::pyo3::Py<::pyo3::PyAny>>) {
                self.inner.noise = noise;
            }

            /// The active truncation pipeline as a list of truncator objects.
            #[getter]
            fn truncators(&self, py: ::pyo3::Python<'_>) -> ::pyo3::PyResult<Vec<::pyo3::Py<::pyo3::PyAny>>> {
                self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
            }

            /// Replace the truncation pipeline (accepts the same forms as the
            /// constructor's `truncation`).
            #[pyo3(signature = (truncation=None))]
            fn set_truncation(
                &mut self,
                truncation: Option<::pyo3::Bound<'_, ::pyo3::PyAny>>,
            ) -> ::pyo3::PyResult<()> {
                let truncators = $crate::truncators::resolve_truncation(truncation.as_ref())?;
                $crate::truncators::reject_surrogate_only(&truncators)?;
                self.inner.truncators = truncators;
                Ok(())
            }
        }

        $(#[$sdoc])*
        #[::pyo3_stub_gen::derive::gen_stub_pyclass]
        #[::pyo3::pyclass(subclass, module = "propaq._rust_core")]
        pub struct $S {
            pub core: $crate::basis_module::GenericTermSum<$B>,
        }

        impl $S {
            pub fn from_store(inner: $crate::store::TermSum<f64>) -> Self {
                $S { core: $crate::basis_module::GenericTermSum::from_store(inner) }
            }

            /// Same as `from_store`, for an f32-backed `TermSum`.
            pub fn from_store_f32(inner: $crate::store::TermSum<f32>) -> Self {
                $S { core: $crate::basis_module::GenericTermSum::from_store_f32(inner) }
            }

            /// Number of units (precision-independent).
            pub fn n_units(&self) -> usize {
                self.core.n_units()
            }

            /// Coefficients widened to f64, regardless of storage precision.
            pub fn as_f64(&self) -> $crate::store::TermSum<f64> {
                self.core.as_f64()
            }
        }

        #[::pyo3_stub_gen::derive::gen_stub_pymethods]
        #[::pyo3::pymethods]
        impl $S {
            /// Initialize a term sum.
            ///
            /// Arguments:
            ///     terms: Optional initial mapping of term to real coefficient.
            ///     dtype: Coefficient precision, "float64" (default) or "float32".
            #[new]
            #[pyo3(signature = (terms=None, dtype=None))]
            fn new(
                terms: Option<&::pyo3::Bound<'_, ::pyo3::types::PyDict>>,
                dtype: Option<&str>,
            ) -> ::pyo3::PyResult<Self> {
                Ok($S { core: $crate::basis_module::GenericTermSum::from_dict::<$K>(terms, dtype)? })
            }

            /// Coefficient precision: "float64" or "float32".
            #[getter]
            fn dtype(&self) -> &str {
                self.core.dtype()
            }

            #[doc = $adddoc]
            pub fn add(&mut self, term: $T, coeff: $K) {
                self.core.add(term, coeff)
            }

            /// Multiply every coefficient by *factor* in-place.
            fn scale(&mut self, factor: f64) {
                self.core.scale(factor)
            }

            /// Add all terms from *other* into this sum. Both sums must share the same dtype.
            pub fn merge(&mut self, other: &$S) -> ::pyo3::PyResult<()> {
                self.core.merge(&other.core, stringify!($S))
            }

            /// Stream terms from a file and merge them into this sum one at a time,
            #[doc = concat!("accumulating coefficients for ", $fitem, " already present. The file is")]
            /// always f64; values are cast down if this sum is float32.
            ///
            /// Arguments:
            #[doc = concat!("    streamer: A ", $stname, " opened with ", $stname, ".from_file().")]
            fn merge_from_file(
                &mut self,
                #[gen_stub(override_type(type_repr = $sr))] streamer: &mut $St,
            ) -> ::pyo3::PyResult<()> {
                self.core.merge_stream(streamer.inner.by_ref())
            }

            /// Deduplicate and remove terms according to *policy*.
            pub fn truncate(&mut self, policy: &::pyo3::Bound<'_, ::pyo3::PyAny>) -> ::pyo3::PyResult<()> {
                self.core.truncate(policy)
            }

            /// Apply noise damping to every coefficient.
            pub fn apply_damping(
                &mut self,
                noise: &::pyo3::Bound<'_, ::pyo3::PyAny>,
                active_modes: u32,
            ) -> ::pyo3::PyResult<()> {
                self.core.apply_damping(noise, active_modes)
            }

            /// Bytes of resident sparse key storage held by this term sum.
            ///
            /// Keys only: coefficients and merge metadata are excluded.
            #[getter]
            fn sparse_key_bytes(&self) -> usize {
                self.core.sparse_key_bytes()
            }

            /// Return the sum of |coefficient|^2 over all terms.
            pub fn norm_squared(&self) -> f64 {
                self.core.norm_squared()
            }

            #[doc = concat!("Return all (", $item, ", coefficient) pairs.")]
            fn items(&self) -> Vec<($T, $K)> {
                self.core.items::<$K>()
            }

            fn __len__(&self) -> usize {
                self.core.len()
            }

            fn __setitem__(&mut self, term: $T, coeff: $K) {
                self.core.set_item(term, coeff)
            }

            fn __getitem__(&self, term: $T) -> $K {
                self.core.get_item::<$K>(&term)
            }

            /// Return a shallow copy of this term sum.
            fn copy(&self) -> $S {
                $S { core: self.core.copy() }
            }

            #[doc = concat!("Load a ", $sumname, " from a gzip-compressed binary file saved by `propagate` or")]
            /// `expectation_value`. Always loads as float64 (the file format's precision).
            ///
            /// Arguments:
            ///     path: Path to the file written by the `filename` parameter.
            #[staticmethod]
            fn from_file(path: &str) -> ::pyo3::PyResult<$S> {
                Ok($S { core: $crate::basis_module::GenericTermSum::from_file(path)? })
            }

            /// Save this term sum to a gzip-compressed binary file. Coefficients are
            /// always widened to f64 on disk regardless of in-memory precision.
            ///
            /// Arguments:
            ///     path: Destination file path.
            fn save(&self, path: &str) -> ::pyo3::PyResult<()> {
                self.core.save(path)
            }
        }
    };
}

/// Defines the term streamer class of one basis.
#[macro_export]
macro_rules! define_basis_streamer {
    ($T:ty, $St:ident) => {
        /// Lazy loading of terms from a term file.
        #[::pyo3_stub_gen::derive::gen_stub_pyclass]
        #[::pyo3::pyclass(module = "propaq._rust_core")]
        pub struct $St {
            pub inner: $crate::streamer::TermStreamer<$T>,
        }

        #[::pyo3_stub_gen::derive::gen_stub_pymethods]
        #[::pyo3::pymethods]
        impl $St {
            /// Open a gzip-compressed binary file for lazy streaming.
            ///
            /// Arguments:
            ///     path: Path to a file written by `TermSum.save()`.
            #[staticmethod]
            fn from_file(path: &str) -> ::pyo3::PyResult<Self> {
                Ok(Self { inner: $crate::streamer::TermStreamer::open(path)? })
            }

            fn __iter__(slf: ::pyo3::PyRef<Self>) -> ::pyo3::PyRef<Self> {
                slf
            }

            fn __next__(&mut self) -> ::pyo3::PyResult<Option<($T, f64)>> {
                match self.inner.next() {
                    None => Ok(None),
                    Some(Ok(pair)) => Ok(Some(pair)),
                    Some(Err(e)) => Err(e),
                }
            }
        }

    };
}

/// Defines the propagator, term sum and term streamer of one basis.
/// A crate that keeps its own streamer type calls `define_basis_classes!`
/// instead and passes that type as `streamer:`.
#[macro_export]
macro_rules! define_basis_module {
    (
        codec: $B:ty,
        term: $T:ty,
        coeff: $K:ty,
        propagator: $P:ident,
        termsum: $S:ident,
        streamer: $St:ident,
        streamer_repr: $sr:literal,
        state_doc: $state:literal,
        obs_doc: $obs:literal,
        circuit_doc: $circ:literal,
        add_doc: $adddoc:literal,
        item_doc: $item:literal,
        sum_doc: $sumname:literal,
        file_item_doc: $fitem:literal,
        streamer_doc: $stname:literal,
        propagator_docs: { $(#[$pdoc:meta])* },
        termsum_docs: { $(#[$sdoc:meta])* } $(,)?
    ) => {
        $crate::define_basis_streamer!($T, $St);
        $crate::define_basis_classes! {
            codec: $B,
            term: $T,
            coeff: $K,
            propagator: $P,
            termsum: $S,
            streamer: $St,
            streamer_repr: $sr,
            state_doc: $state,
            obs_doc: $obs,
            circuit_doc: $circ,
            add_doc: $adddoc,
            item_doc: $item,
            sum_doc: $sumname,
            file_item_doc: $fitem,
            streamer_doc: $stname,
            propagator_docs: { $(#[$pdoc])* },
            termsum_docs: { $(#[$sdoc])* },
        }
    };
}
