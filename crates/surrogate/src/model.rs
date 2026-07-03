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

use crate::symcoeff::{MonomialUnit, SymbolicCoeff};

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
        let lut = Self::make_lut(params);
        self.terms
            .par_iter()
            .map(|t| t.overlap * t.coeff.evaluate(&lut))
            .sum()
    }

    /// Evaluate for many parameter assignments at once. Parallelizes across
    /// assignments (each of which still parallelizes across terms/monomials
    /// internally — rayon's work stealing handles the nesting), which is the
    /// natural shape for the build-once/evaluate-many workloads this model
    /// exists for.
    pub fn evaluate_batch(&self, param_sets: &[Vec<f64>]) -> Vec<f64> {
        param_sets.par_iter().map(|p| self.evaluate(p)).collect()
    }

    /// Flat evaluation LUT indexed by `2 * param_index` (`cos(theta_i)`) /
    /// `2 * param_index + 1` (`sin(theta_i)`). Unlike the old tally-mark
    /// `TrigFactor` design (one direct gather per arena slot), a slot can now
    /// hold `cos^a * sin^b`, so evaluation does up to two gathers plus a
    /// `powi` per slot -- see `SymbolicCoeff::evaluate`.
    fn make_lut(params: &[f64]) -> Vec<f64> {
        params.iter().flat_map(|&t| [t.cos(), t.sin()]).collect()
    }

    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Save to a gzip-compressed binary file.
    ///
    /// Header (little-endian u64): n_params, n_terms, system_size, key_stride.
    /// Per term: key_bytes, overlap (f64le), n_monomials (u64le).
    /// Per monomial: scalar (f64le), n_factors (u64le), factors (u64le each
    /// -- one packed `MonomialUnit` per *distinct parameter*, not per
    /// occurrence).
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
            let n_mono = st.coeff.monomial_count() as u64;
            enc.write_all(&n_mono.to_le_bytes())?;
            for (scalar, factors) in st.coeff.iter_monomials() {
                enc.write_all(&scalar.to_le_bytes())?;
                enc.write_all(&(factors.len() as u64).to_le_bytes())?;
                for f in factors {
                    enc.write_all(&f.raw().to_le_bytes())?;
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

        let n_params = read_u64!() as usize;
        let n_terms  = read_u64!() as usize;
        let system_size = read_u64!();
        let key_stride  = read_u64!() as usize;

        let mut key_buf = vec![0u8; key_stride];
        let mut terms = Vec::with_capacity(n_terms);
        // Reused across monomials: one factor-run staging buffer for the
        // whole load instead of an allocation per monomial.
        let mut run: Vec<MonomialUnit> = Vec::new();

        for _ in 0..n_terms {
            dec.read_exact(&mut key_buf)?;
            let term = M::from_bytes_vec(&key_buf, system_size);
            let overlap = read_f64!();
            let n_mono = read_u64!() as usize;
            let mut coeff = SymbolicCoeff::default();
            coeff.reserve(n_mono, 0);
            for _ in 0..n_mono {
                let scalar = read_f64!();
                let n_factors = read_u64!() as usize;
                run.clear();
                for _ in 0..n_factors {
                    run.push(MonomialUnit::from_raw(read_u64!()));
                }
                coeff.push_monomial(scalar, &run);
            }
            terms.push(SurrogateTerm { term, overlap, coeff });
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
    fn evaluate(&self, py: Python<'_>, params: Vec<f64>) -> PyResult<f64> {
        if params.len() < self.inner.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "params has {} elements but model requires {}",
                params.len(), self.inner.n_params
            )));
        }
        Ok(py.allow_threads(|| self.inner.evaluate(&params)))
    }

    /// Evaluate many parameter assignments at once (parallelized across
    /// assignments); returns one expectation value per assignment.
    fn evaluate_batch(&self, py: Python<'_>, param_sets: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        for (i, params) in param_sets.iter().enumerate() {
            if params.len() < self.inner.n_params {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "param_sets[{i}] has {} elements but model requires {}",
                    params.len(), self.inner.n_params
                )));
            }
        }
        Ok(py.allow_threads(|| self.inner.evaluate_batch(&param_sets)))
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
    fn evaluate(&self, py: Python<'_>, params: Vec<f64>) -> PyResult<f64> {
        if params.len() < self.inner.n_params {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "params has {} elements but model requires {}",
                params.len(), self.inner.n_params
            )));
        }
        Ok(py.allow_threads(|| self.inner.evaluate(&params)))
    }

    /// Evaluate many parameter assignments at once (parallelized across
    /// assignments); returns one expectation value per assignment.
    fn evaluate_batch(&self, py: Python<'_>, param_sets: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        for (i, params) in param_sets.iter().enumerate() {
            if params.len() < self.inner.n_params {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "param_sets[{i}] has {} elements but model requires {}",
                    params.len(), self.inner.n_params
                )));
            }
        }
        Ok(py.allow_threads(|| self.inner.evaluate_batch(&param_sets)))
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
