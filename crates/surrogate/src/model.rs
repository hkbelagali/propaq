use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::OpenOptions;

use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use pyo3::prelude::*;
use rayon::prelude::*;

use propaq_core::traits::AbstractTerm;
use propaq_pauli::string::PauliString;
use propaq_majorana::monomial::MajoranaMonomial;

use crate::factors::Factors;
use crate::symcoeff::{SymbolicCoeff, Monomial, TrigFactor};

/// A single compiled term: a Pauli/Majorana string with its structural overlap and
/// symbolic coefficient that maps parameter angles to a numerical contribution.
pub struct SurrogateTerm<M: AbstractTerm> {
    pub term: M,
    /// `term.trace_with_fock_state(initial_state)`; nonzero by construction.
    pub overlap: f64,
    pub coeff: SymbolicCoeff,
}

/// Compiled output of a surrogate propagation run.
///
/// Contains only terms with nonzero structural overlap (filter is structural,
/// not coefficient-dependent). Call `evaluate` with parameter angles to obtain
/// the expectation value without re-running propagation.
pub struct SurrogateModel<M: AbstractTerm> {
    pub terms: Vec<SurrogateTerm<M>>,
    pub n_params: usize,
}

impl<M: AbstractTerm> SurrogateModel<M> {
    pub fn new(terms: Vec<SurrogateTerm<M>>, n_params: usize) -> Self {
        SurrogateModel { terms, n_params }
    }

    /// Evaluate the expectation value for the given parameter angles.
    ///
    /// `params[i]` is the angle (in radians) for parameter index `i`.
    /// Length must be at least `self.n_params`.
    pub fn evaluate(&self, params: &[f64]) -> f64 {
        let lut: Vec<(f64, f64)> = params.iter().map(|&t| (t.cos(), t.sin())).collect();
        self.terms
            .par_iter()
            .map(|t| t.overlap * t.coeff.evaluate(&lut))
            .sum()
    }

    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Save to a gzip-compressed binary file.
    ///
    /// Header (little-endian u64): n_params, n_terms, system_size, key_stride.
    /// Per term: key_bytes, overlap (f64le), n_monomials (u64le).
    /// Per monomial: scalar (f64le), n_factors (u64le), factors (u32le each).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .create(true).write(true).truncate(true)
            .open(path)?;
        let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());

        let first = self.terms.first();
        let key_stride: u64 = first.map_or(0, |t| t.term.to_bytes_vec().len() as u64);
        let system_size: u64 = first.map_or(0, |t| t.term.system_size());

        enc.write_all(&(self.n_params as u64).to_le_bytes())?;
        enc.write_all(&(self.terms.len() as u64).to_le_bytes())?;
        enc.write_all(&system_size.to_le_bytes())?;
        enc.write_all(&key_stride.to_le_bytes())?;

        for st in &self.terms {
            enc.write_all(&st.term.to_bytes_vec())?;
            enc.write_all(&st.overlap.to_le_bytes())?;
            let n_mono = st.coeff.monomials.len() as u64;
            enc.write_all(&n_mono.to_le_bytes())?;
            for m in &st.coeff.monomials {
                enc.write_all(&m.scalar.to_le_bytes())?;
                enc.write_all(&(m.factors.len() as u64).to_le_bytes())?;
                for f in m.factors.iter() {
                    enc.write_all(&f.0.to_le_bytes())?;
                }
            }
        }

        enc.finish()?;
        Ok(())
    }

    /// Load from a file produced by `save`.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mut dec = BufReader::new(GzDecoder::new(file));

        let mut u64_buf = [0u8; 8];
        let mut u32_buf = [0u8; 4];

        macro_rules! read_u64 {
            () => {{
                dec.read_exact(&mut u64_buf)?;
                u64::from_le_bytes(u64_buf)
            }};
        }
        macro_rules! read_f64 {
            () => {{
                dec.read_exact(&mut u64_buf)?;
                f64::from_le_bytes(u64_buf)
            }};
        }
        macro_rules! read_u32 {
            () => {{
                dec.read_exact(&mut u32_buf)?;
                u32::from_le_bytes(u32_buf)
            }};
        }

        let n_params = read_u64!() as usize;
        let n_terms  = read_u64!() as usize;
        let system_size = read_u64!();
        let key_stride  = read_u64!() as usize;

        let mut key_buf = vec![0u8; key_stride];
        let mut terms = Vec::with_capacity(n_terms);

        for _ in 0..n_terms {
            dec.read_exact(&mut key_buf)?;
            let term = M::from_bytes_vec(&key_buf, system_size);
            let overlap = read_f64!();
            let n_mono = read_u64!() as usize;
            let mut monomials = Vec::with_capacity(n_mono);
            for _ in 0..n_mono {
                let scalar = read_f64!();
                let n_factors = read_u64!() as usize;
                let mut factors = Factors::with_capacity(n_factors);
                for _ in 0..n_factors {
                    factors.push(TrigFactor(read_u32!()));
                }
                monomials.push(Monomial { scalar, factors });
            }
            terms.push(SurrogateTerm { term, overlap, coeff: SymbolicCoeff { monomials } });
        }

        Ok(SurrogateModel { terms, n_params })
    }
}

/// Compiled surrogate model for Pauli observables.
///
/// Produced by `PauliSurrogatePropagator.build`. Call `evaluate(params)` to
/// obtain the expectation value for a specific parameter assignment without
/// re-running propagation. Use `save`/`load` for persistence.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogateModel {
    pub(crate) inner: SurrogateModel<PauliString>,
}

#[pymethods]
impl PauliSurrogateModel {
    /// Evaluate the expectation value. `params[i]` is the angle (radians) for parameter `i`.
    fn evaluate(&self, params: Vec<f64>) -> PyResult<f64> {
        if params.len() < self.inner.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "params has {} elements but model requires {}",
                params.len(), self.inner.n_params
            )));
        }
        Ok(self.inner.evaluate(&params))
    }

    /// Save to a gzip-compressed binary file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a model from a file produced by `save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = SurrogateModel::<PauliString>::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PauliSurrogateModel { inner })
    }

    /// Number of parameter indices used by this model.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params
    }

    /// Number of compiled terms.
    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliSurrogateModel(n_terms={}, n_params={})",
            self.inner.n_terms(), self.inner.n_params
        )
    }
}

/// Compiled surrogate model for Majorana observables.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogateModel {
    pub(crate) inner: SurrogateModel<MajoranaMonomial>,
}

#[pymethods]
impl MajoranaSurrogateModel {
    /// Evaluate the expectation value. `params[i]` is the angle (radians) for parameter `i`.
    fn evaluate(&self, params: Vec<f64>) -> PyResult<f64> {
        if params.len() < self.inner.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "params has {} elements but model requires {}",
                params.len(), self.inner.n_params
            )));
        }
        Ok(self.inner.evaluate(&params))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = SurrogateModel::<MajoranaMonomial>::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(MajoranaSurrogateModel { inner })
    }

    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params
    }

    #[getter]
    fn n_terms(&self) -> usize {
        self.inner.n_terms()
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaSurrogateModel(n_terms={}, n_params={})",
            self.inner.n_terms(), self.inner.n_params
        )
    }
}
