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

use crate::symcoeff::SymbolicCoeff;

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
    /// `2 * param_index + 1` (`sin(theta_i)`). `SymbolicCoeff::evaluate` walks
    /// each monomial's factor run, reads each parameter index directly from the
    /// factor, and raises the matching `cos`/`sin` to the recorded powers.
    fn make_lut(params: &[f64]) -> Vec<f64> {
        params.iter().flat_map(|&t| [t.cos(), t.sin()]).collect()
    }

    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Save to a sharded, parallel-friendly binary file (see the module-level
    /// format notes on `MAGIC`/`FORMAT_VERSION`).
    ///
    /// Terms are split into ~`current_num_threads()` contiguous shards; each is
    /// serialized and gzip-compressed **independently and in parallel**, then
    /// the header, a shard-length index, and the compressed blobs are written
    /// sequentially. This makes `save` scale with cores instead of walking every
    /// monomial single-threaded. The format is **not** backward compatible with
    /// pre-sharding files — `load` rejects them and the model must be rebuilt.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let first = self.terms.first();
        let key_stride: u64 = first.map_or(0, |t| t.term.to_bytes_vec().len() as u64);
        let system_size: u64 = first.map_or(0, |t| t.term.system_size());

        // Contiguous shards, one per worker (at least one term each). Each shard
        // is serialized + compressed on its own thread; blobs come back in term
        // order because `par_chunks` preserves order.
        let n_terms = self.terms.len();
        let target_shards = rayon::current_num_threads().max(1);
        let chunk = n_terms.div_ceil(target_shards).max(1);
        let blobs: Vec<Vec<u8>> = self
            .terms
            .par_chunks(chunk)
            .map(|shard| {
                let mut raw = Vec::new();
                for st in shard {
                    write_term_into(&mut raw, st);
                }
                gzip_block(&raw)
            })
            .collect::<std::io::Result<_>>()?;

        let file = OpenOptions::new().create(true).write(true).truncate(true).open(path)?;
        let mut w = BufWriter::new(file);
        w.write_all(&MAGIC.to_le_bytes())?;
        w.write_all(&FORMAT_VERSION.to_le_bytes())?;
        w.write_all(&(self.n_params as u64).to_le_bytes())?;
        w.write_all(&system_size.to_le_bytes())?;
        w.write_all(&key_stride.to_le_bytes())?;
        w.write_all(&(blobs.len() as u64).to_le_bytes())?;
        for b in &blobs {
            w.write_all(&(b.len() as u64).to_le_bytes())?;
        }
        for b in &blobs {
            w.write_all(b)?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load from a file produced by `save`. The header + compressed blobs are
    /// read sequentially, then the shards are decompressed and parsed in
    /// parallel and concatenated (in shard/term order). Rejects files that don't
    /// carry the current `MAGIC`/`FORMAT_VERSION` — those predate the sharded
    /// format and their models must be rebuilt.
    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::new(std::fs::File::open(path)?);

        let mut u64_buf = [0u8; 8];
        let mut u32_buf = [0u8; 4];
        macro_rules! read_u64 {
            () => {{ r.read_exact(&mut u64_buf)?; u64::from_le_bytes(u64_buf) }};
        }
        macro_rules! read_u32 {
            () => {{ r.read_exact(&mut u32_buf)?; u32::from_le_bytes(u32_buf) }};
        }

        let magic = read_u32!();
        let version = read_u32!();
        if magic != MAGIC || version != FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unrecognized or outdated surrogate model file (format changed); rebuild the model",
            ));
        }

        let n_params = read_u64!() as usize;
        let system_size = read_u64!();
        let key_stride = read_u64!() as usize;

        let n_shards = read_u64!() as usize;
        let mut shard_lens = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            shard_lens.push(read_u64!() as usize);
        }
        // Read each compressed blob sequentially (I/O is serial), then decode +
        // parse them in parallel.
        let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(n_shards);
        for len in shard_lens {
            let mut blob = vec![0u8; len];
            r.read_exact(&mut blob)?;
            blobs.push(blob);
        }

        let per_shard: Vec<Vec<SurrogateTerm<M>>> = blobs
            .par_iter()
            .map(|blob| parse_shard::<M>(blob, key_stride, system_size))
            .collect::<std::io::Result<_>>()?;

        let mut terms = Vec::with_capacity(per_shard.iter().map(|s| s.len()).sum());
        for shard in per_shard {
            terms.extend(shard);
        }

        Ok(SurrogateModel { terms, n_params })
    }
}

/// Magic bytes and format version stamped at the head of every saved model.
/// A mismatch on load is a hard error — the sharded format deliberately breaks
/// compatibility with pre-sharding files.
const MAGIC: u32 = u32::from_le_bytes(*b"PQSM");
const FORMAT_VERSION: u32 = 3;

/// Serialize one term into `buf` (uncompressed): key bytes, overlap (f64le),
/// monomial count (u64le), then per monomial `scalar` (f64le), factor count
/// (u64le), and that many packed `u32le` parameter-space factors from the
/// coefficient arena.
fn write_term_into<M: AbstractTerm>(buf: &mut Vec<u8>, st: &SurrogateTerm<M>) {
    buf.extend_from_slice(&st.term.to_bytes_vec());
    buf.extend_from_slice(&st.overlap.to_le_bytes());
    buf.extend_from_slice(&(st.coeff.monomial_count() as u64).to_le_bytes());
    for (scalar, run) in st.coeff.iter_monomials() {
        buf.extend_from_slice(&scalar.to_le_bytes());
        buf.extend_from_slice(&(run.len() as u64).to_le_bytes());
        for &f in run {
            buf.extend_from_slice(&f.to_le_bytes());
        }
    }
}

/// gzip a shard's raw bytes into a self-contained compressed blob.
fn gzip_block(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)?;
    enc.finish()
}

/// Decompress and parse one shard blob into its terms (the inverse of
/// `write_term_into`), consuming the whole decompressed buffer.
fn parse_shard<M: AbstractTerm>(
    compressed: &[u8],
    key_stride: usize,
    system_size: u64,
) -> std::io::Result<Vec<SurrogateTerm<M>>> {
    let mut raw = Vec::new();
    GzDecoder::new(compressed).read_to_end(&mut raw)?;

    #[inline]
    fn rd_u64(b: &[u8], pos: &mut usize) -> u64 {
        let v = u64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    }
    #[inline]
    fn rd_f64(b: &[u8], pos: &mut usize) -> f64 {
        let v = f64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    }
    #[inline]
    fn rd_u32(b: &[u8], pos: &mut usize) -> u32 {
        let v = u32::from_le_bytes(b[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        v
    }

    let mut terms = Vec::new();
    let mut factors: Vec<u32> = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let term = M::from_bytes_vec(&raw[pos..pos + key_stride], system_size);
        pos += key_stride;
        let overlap = rd_f64(&raw, &mut pos);
        let n_mono = rd_u64(&raw, &mut pos) as usize;
        let mut coeff = SymbolicCoeff::default();
        coeff.reserve(n_mono, 0);
        for _ in 0..n_mono {
            let scalar = rd_f64(&raw, &mut pos);
            let n_factors = rd_u64(&raw, &mut pos) as usize;
            factors.clear();
            factors.reserve(n_factors);
            for _ in 0..n_factors {
                factors.push(rd_u32(&raw, &mut pos));
            }
            coeff.push_monomial(scalar, &factors);
        }
        terms.push(SurrogateTerm { term, overlap, coeff });
    }
    Ok(terms)
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
