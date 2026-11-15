//!
//! A generic raw-word term codec.
//!
//! A basis made with the tracer has no Rust term type of its own. A term is
//! a list of `u64` words and a unit count. The bits of unit `u` sit at bits
//! `u * B .. (u + 1) * B`. This file turns any `Basis<W>` algebra into a
//! `BasisCodec` with such terms, so that `define_raw_basis_module!` can
//! build the Python classes without a caller term type.
//!
//! Storage keeps the raw words in plane 0 of a `TermSum` row. Plane 1 stays
//! empty. The width-generic algebra is reached through `at_width!`.
//!

use std::hash::{Hash, Hasher};
use std::marker::PhantomData;

use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3_stub_gen::{PyStubType, TypeInfo};
use smallvec::{smallvec, SmallVec};

use crate::basis_module::BasisCodec;
use crate::run::BasisAll;
use crate::sparse::{self, Position};
use crate::store::TermBasis;
use crate::strings::BasisString;
use crate::traits::AbstractTerm;

/// Widest key the raw codec covers, in words. Same as the engine dispatch.
pub const MAX_RAW_WORDS: usize = 64;

/// Runs `$body` with `$W` set to the power-of-two word count that holds `$n` words.
#[macro_export]
macro_rules! at_width {
    ($n:expr, |$W:ident| $body:expr) => {{
        match ($n).max(1).next_power_of_two() {
            1 => {
                const $W: usize = 1;
                $body
            }
            2 => {
                const $W: usize = 2;
                $body
            }
            4 => {
                const $W: usize = 4;
                $body
            }
            8 => {
                const $W: usize = 8;
                $body
            }
            16 => {
                const $W: usize = 16;
                $body
            }
            32 => {
                const $W: usize = 32;
                $body
            }
            64 => {
                const $W: usize = 64;
                $body
            }
            n => panic!("propaq: a raw term of {n} words is too wide"),
        }
    }};
}

fn key_of<const W: usize>(words: &[u64]) -> BasisString<W> {
    let mut w = [0u64; W];
    let n = words.len().min(W);
    w[..n].copy_from_slice(&words[..n]);
    BasisString::from_words(w)
}

fn words_of<const W: usize>(key: &BasisString<W>, len: usize) -> Vec<u64> {
    key.words()[..len.min(W)].to_vec()
}

/// Number of words a term of `n_units` units holds.
#[inline]
pub fn raw_stride(n_units: usize, bits_per_unit: usize) -> usize {
    (n_units * bits_per_unit).div_ceil(64).max(1)
}

/// The python-visible term of a traced basis: raw words and a unit count.
///
/// Equal terms have equal words. Words beyond the unit count must be zero.
#[::pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(name = "Term", module = "propaq._rust_core", frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RawKey {
    words: Vec<u64>,
    n_units: usize,
}

impl RawKey {
    /// The canonical form: trailing zero words are dropped, one word stays.
    fn canonical(mut words: Vec<u64>, n_units: usize) -> Self {
        while words.len() > 1 && words.last() == Some(&0) {
            words.pop();
        }
        if words.is_empty() {
            words.push(0);
        }
        RawKey { words, n_units }
    }
}

#[::pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl RawKey {
    /// A term from raw words. `words` may be shorter than the full key; it is zero-extended.
    ///
    /// Arguments:
    ///     words: The key words, bit `u * b + j` is bit `j` of unit `u`.
    ///     n_units: Number of units (qudits, qubits, sites) of the system.
    #[new]
    fn new(words: Vec<u64>, n_units: usize) -> PyResult<Self> {
        Ok(RawKey::canonical(words, n_units))
    }

    /// The raw words of the key.
    #[getter(words)]
    fn words_py(&self) -> Vec<u64> {
        self.words.clone()
    }

    /// Number of units of the system.
    #[getter(n_units)]
    fn n_units_py(&self) -> usize {
        self.n_units
    }

    fn __repr__(&self) -> String {
        let words: Vec<String> = self.words.iter().map(|w| format!("{w:#x}")).collect();
        format!("Term(words=[{}], n_units={})", words.join(", "), self.n_units)
    }
}

/// A term of an algebra `A` with `B` bits per unit. Rust side of `RawKey`.
pub struct RawTerm<A, const B: usize> {
    pub words: Vec<u64>,
    pub n_units: usize,
    _alg: PhantomData<fn() -> A>,
}

impl<A, const B: usize> RawTerm<A, B> {
    /// Builds a term. Words are cut or zero-extended to the key length.
    pub fn new(words: &[u64], n_units: usize) -> Self {
        let len = raw_stride(n_units, B);
        let mut w = vec![0u64; len];
        let n = words.len().min(len);
        w[..n].copy_from_slice(&words[..n]);
        RawTerm { words: w, n_units, _alg: PhantomData }
    }
}

impl<A, const B: usize> Clone for RawTerm<A, B> {
    fn clone(&self) -> Self {
        RawTerm { words: self.words.clone(), n_units: self.n_units, _alg: PhantomData }
    }
}

impl<A, const B: usize> PartialEq for RawTerm<A, B> {
    fn eq(&self, other: &Self) -> bool {
        self.n_units == other.n_units && self.words == other.words
    }
}

impl<A, const B: usize> Eq for RawTerm<A, B> {}

impl<A, const B: usize> Hash for RawTerm<A, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.n_units.hash(state);
        self.words.hash(state);
    }
}

impl<'py, A, const B: usize> IntoPyObject<'py> for RawTerm<A, B> {
    type Target = RawKey;
    type Output = Bound<'py, RawKey>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(py, RawKey::canonical(self.words, self.n_units))
    }
}

impl<'a, 'py, A, const B: usize> FromPyObject<'a, 'py> for RawTerm<A, B> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        let key = obj.cast::<RawKey>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("expected a Term of this basis module")
        })?;
        let key = key.get();
        let len = raw_stride(key.n_units, B);
        if key.words.len() > len && key.words[len..].iter().any(|&w| w != 0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "term has set bits beyond its unit count",
            ));
        }
        Ok(RawTerm::new(&key.words, key.n_units))
    }
}

impl<A, const B: usize> PyStubType for RawTerm<A, B> {
    fn type_output() -> TypeInfo {
        <RawKey as PyStubType>::type_output()
    }
    fn type_input() -> TypeInfo {
        <RawKey as PyStubType>::type_input()
    }
}

// ------------------------------------------------------- algebra helpers

/// Weight of the raw term `words`.
pub fn raw_weight<A: BasisAll>(words: &[u64], n_units: usize) -> u32 {
    at_width!(words.len(), |W| <A as crate::basis::Basis<W>>::weight(&key_of::<W>(words), n_units))
}

/// The character chi in Z_M of `t` against the generator `g`.
pub fn raw_character<A: BasisAll>(g: &[u64], t: &[u64]) -> u32 {
    at_width!(t.len().max(g.len()), |W| {
        let ctx = <A as crate::basis::Basis<W>>::make_gen_context(&key_of::<W>(g));
        <A as crate::basis::Basis<W>>::character(&ctx, &key_of::<W>(t))
    })
}

/// The exponent `e` with `phase = exp(2 pi i e / D)`, for the phase order `D` of `A`.
pub fn phase_exponent<A: BasisAll>(phase: Complex64) -> u32 {
    let d = <A as crate::basis::Basis<1>>::PHASE_ORDER as f64;
    let e = (phase.arg() * d / std::f64::consts::TAU).round() as i64;
    e.rem_euclid(d as i64) as u32
}

/// True if the character of `t` against `g` is zero, so a rotation about `g` leaves `t` unchanged.
pub fn raw_commutes<A: BasisAll>(t: &[u64], g: &[u64]) -> bool {
    at_width!(t.len().max(g.len()), |W| {
        let ctx = <A as crate::basis::Basis<W>>::make_gen_context(&key_of::<W>(g));
        <A as crate::basis::Basis<W>>::character(&ctx, &key_of::<W>(t)) == 0
    })
}

/// The product `g * t`: key words (length `t.len()`) and phase.
pub fn raw_product<A: BasisAll>(t: &[u64], g: &[u64]) -> (Vec<u64>, Complex64) {
    at_width!(t.len().max(g.len()), |W| {
        let ctx = <A as crate::basis::Basis<W>>::make_gen_context(&key_of::<W>(g));
        let (key, phase) = <A as crate::basis::Basis<W>>::product(&ctx, &key_of::<W>(t));
        (words_of::<W>(&key, t.len()), phase)
    })
}

/// Diagonal expectation of the term `t` against `diag_state` (one bit per unit).
pub fn raw_trace<A: BasisAll>(t: &[u64], n_units: usize, diag_state: &[u64]) -> f64 {
    at_width!(t.len(), |W| <A as crate::basis::Basis<W>>::trace(&key_of::<W>(t), n_units, diag_state))
}

/// Decodes a sparse row into raw words (plane 0 only).
fn decode_words(row: &[Position], plane_span: usize) -> SmallVec<[u64; 8]> {
    let stride = plane_span / 64;
    let mut buf: SmallVec<[u64; 8]> = smallvec![0u64; 2 * stride];
    let (a, b) = buf.split_at_mut(stride);
    sparse::decode_row_into(row, plane_span, [a, b]);
    buf.truncate(stride);
    buf
}

// ------------------------------------------------------------------ codec

/// The codec of algebra `A` with `B` bits per unit.
pub struct RawBasis<A, const B: usize>(PhantomData<fn() -> A>);

impl<A: BasisAll, const B: usize> AbstractTerm for RawTerm<A, B> {
    fn weight(&self) -> u32 {
        raw_weight::<A>(&self.words, self.n_units)
    }

    fn commutes_with(&self, other: &Self) -> bool {
        raw_commutes::<A>(&self.words, &other.words)
    }

    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) {
        // `other * self`, the same order as `product(gen = other, term = self)`.
        let (words, phase) = raw_product::<A>(&self.words, &other.words);
        (phase, RawTerm::new(&words, self.n_units))
    }

    fn trace_with_diag_state(&self, diag_state: &crate::bitset::Bitset) -> f64 {
        raw_trace::<A>(&self.words, self.n_units, diag_state.as_words())
    }

    fn to_bytes_vec(&self) -> Vec<u8> {
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn partition_key(&self) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        self.words.hash(&mut h);
        h.finish()
    }

    fn system_size(&self) -> u64 {
        self.n_units as u64
    }

    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self {
        let words: Vec<u64> = bytes.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect();
        RawTerm::new(&words, system_size as usize)
    }
}

impl<A: BasisAll, const B: usize> TermBasis for RawBasis<A, B> {
    type Term = RawTerm<A, B>;

    fn stride_words(n_units: usize) -> usize {
        raw_stride(n_units, B)
    }

    fn commutes(term: [&[u64]; 2], gen: [&[u64]; 2]) -> bool {
        raw_commutes::<A>(term[0], gen[0])
    }

    fn product(term: [&[u64]; 2], gen: [&[u64]; 2], out: [&mut [u64]; 2]) -> Complex64 {
        let (words, phase) = raw_product::<A>(term[0], gen[0]);
        out[0].copy_from_slice(&words);
        out[1].fill(0);
        phase
    }

    fn weight(term: [&[u64]; 2], n_units: usize) -> u32 {
        raw_weight::<A>(term[0], n_units)
    }

    fn trace(term: [&[u64]; 2], n_units: usize, diag_state: &[u64]) -> f64 {
        raw_trace::<A>(term[0], n_units, diag_state)
    }

    fn key_hash(term: [&[u64]; 2]) -> u64 {
        let mut h = rustc_hash::FxHasher::default();
        term[0].hash(&mut h);
        h.finish()
    }

    fn key_eq(a: [&[u64]; 2], b: [&[u64]; 2]) -> bool {
        a[0] == b[0]
    }

    fn term_from_planes(term: [&[u64]; 2], n_units: usize) -> RawTerm<A, B> {
        RawTerm::new(term[0], n_units)
    }

    fn term_into_planes(term: &RawTerm<A, B>, _n_units: usize, out: [&mut [u64]; 2]) {
        out[0].fill(0);
        out[1].fill(0);
        let n = term.words.len().min(out[0].len());
        out[0][..n].copy_from_slice(&term.words[..n]);
    }

    fn weight_sparse(row: &[Position], plane_span: usize, n_units: usize) -> u32 {
        raw_weight::<A>(&decode_words(row, plane_span), n_units)
    }
}

impl<A: BasisAll, const B: usize> BasisCodec for RawBasis<A, B> {
    type Alg = A;

    const UNIT_NAME: &'static str = "units";
    const MAX_UNITS: usize = MAX_RAW_WORDS * 64 / B;
    const BITS_PER_UNIT: usize = B;

    fn engine_units(n_units: usize) -> Option<usize> {
        Some(n_units)
    }

    fn extract_term(obj: &Bound<'_, PyAny>) -> PyResult<RawTerm<A, B>> {
        obj.extract()
    }

    fn to_words(term: &RawTerm<A, B>) -> Vec<u64> {
        term.words.clone()
    }

    fn from_words(words: &[u64], n_units: usize) -> RawTerm<A, B> {
        RawTerm::new(words, n_units)
    }
}

/// Adds the engine classes that a propagator takes as arguments to `m`.
///
/// A compiled basis is its own shared library. It holds its own copy of
/// these classes. A truncator made in `propaq` is a different Python type
/// from the one in the compiled module, so the module must export its own.
pub fn register_engine_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    use crate::{
        CoefficientTruncator, FrequencyTruncator, GateNoiseModel, Logger, NativeNoiseModel,
        NativeTruncator, PropagationResult, Simplify, TermBudget, TruncationPolicy,
        UniformNoiseModel, WeightTruncator,
    };
    m.add_class::<RawKey>()?;
    m.add_class::<PropagationResult>()?;
    m.add_class::<Logger>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<NativeNoiseModel>()?;
    m.add_class::<FrequencyTruncator>()?;
    m.add_class::<CoefficientTruncator>()?;
    m.add_class::<WeightTruncator>()?;
    m.add_class::<TermBudget>()?;
    m.add_class::<NativeTruncator>()?;
    m.add_class::<Simplify>()?;
    Ok(())
}

/// Defines a complete PyO3 module for a traced basis.
///
/// The module holds `Propagator`, `TermSum`, `TermStreamer` and `Term`, and
/// the engine classes of `register_engine_classes`. Terms are raw words,
/// so no caller term type is needed.
///
/// ```ignore
/// propaq_core::define_raw_basis_module! {
///     module: propaq_basis_0123456789abcdef,
///     algebra: MyBasis,        // implements Basis<W> for every W
///     bits_per_unit: 4,
///     coeff: f64,          // or `propaq_core::basis_module::Cx` for complex coefficients
/// }
/// ```
#[macro_export]
macro_rules! define_raw_basis_module {
    (module: $m:ident, algebra: $A:ty, bits_per_unit: $b:literal, coeff: $K:ty $(,)?) => {
        /// The codec of this module's basis.
        pub type Codec = $crate::raw_basis::RawBasis<$A, $b>;
        /// The Rust side of this module's `Term` class.
        pub type TermT = $crate::raw_basis::RawTerm<$A, $b>;

        $crate::define_basis_module! {
            codec: Codec,
            term: TermT,
            coeff: $K,
            propagator: Propagator,
            termsum: TermSum,
            streamer: TermStreamer,
            streamer_repr: "TermStreamer",
            state_doc: "     initial_state: Computational basis reference state as a bitstring integer, one bit per unit.",
            obs_doc: "observable",
            circuit_doc: "circuit",
            add_doc: "Add *coeff* x *term* to the sum, accumulating if the term is already present.",
            item_doc: "term",
            sum_doc: "term sum",
            file_item_doc: "terms",
            streamer_doc: "TermStreamer",
            propagator_docs: {
                /// Back-propagates observables of a traced basis through circuits in the Heisenberg picture.
                ///
                /// Arguments:
                ///     noise: Optional noise model.
                ///     truncation: A truncator from this module, a list of them, or None.
                ///     n_threads: Number of worker threads. Defaults to the system thread count.
                ///     logger: Optional Logger for verbose JSON Lines event logging.
                ///     pin_threads: Bind each worker to its own CPU.
                ///     progress_bar: Draw a tqdm bar over the gate loop.
                ///     progress_every: Gates between progress bar ticks. Defaults to 1.
            },
            termsum_docs: {
                /// A mutable, weighted sum of terms of a traced basis with real coefficients.
                ///
                /// Arguments:
                ///     terms: Optional initial mapping of Term to real coefficient.
                ///     dtype: Coefficient precision, "float64" (default) or "float32".
            },
        }

        /// Character chi in Z_M of the term `t` against the generator `g` (raw words).
        #[::pyo3::pyfunction]
        #[pyo3(name = "character")]
        fn raw_character_py(g: Vec<u64>, t: Vec<u64>) -> u32 {
            $crate::raw_basis::raw_character::<$A>(&g, &t)
        }

        /// The product `g * t` as (key words, e), with phase exp(2 pi i e / D).
        #[::pyo3::pyfunction]
        #[pyo3(name = "product")]
        fn raw_product_py(g: Vec<u64>, t: Vec<u64>) -> (Vec<u64>, u32) {
            let (words, phase) = $crate::raw_basis::raw_product::<$A>(&t, &g);
            (words, $crate::raw_basis::phase_exponent::<$A>(phase))
        }

        /// Weight of the term `t` in a system of `n_units` units.
        #[::pyo3::pyfunction]
        #[pyo3(name = "weight")]
        fn raw_weight_py(t: Vec<u64>, n_units: usize) -> u32 {
            $crate::raw_basis::raw_weight::<$A>(&t, n_units)
        }

        /// Diagonal expectation of `t` against `diag` (one bit per unit).
        #[::pyo3::pyfunction]
        #[pyo3(name = "trace")]
        fn raw_trace_py(t: Vec<u64>, diag: Vec<u64>, n_units: usize) -> f64 {
            $crate::raw_basis::raw_trace::<$A>(&t, n_units, &diag)
        }

        #[::pyo3::pymodule]
        fn $m(m: &::pyo3::Bound<'_, ::pyo3::types::PyModule>) -> ::pyo3::PyResult<()> {
            use ::pyo3::types::PyModuleMethods;
            $crate::raw_basis::register_engine_classes(m)?;
            m.add_function(::pyo3::wrap_pyfunction!(raw_character_py, m)?)?;
            m.add_function(::pyo3::wrap_pyfunction!(raw_product_py, m)?)?;
            m.add_function(::pyo3::wrap_pyfunction!(raw_weight_py, m)?)?;
            m.add_function(::pyo3::wrap_pyfunction!(raw_trace_py, m)?)?;
            m.add_class::<Propagator>()?;
            m.add_class::<TermSum>()?;
            m.add_class::<TermStreamer>()?;
            Ok(())
        }
    };
}
