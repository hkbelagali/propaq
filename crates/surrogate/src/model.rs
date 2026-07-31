///
/// Representation of a compiled surrogate model for expectation value calculations.
/// After a symbolic propagation, the resulting object is a mapping
///
/// $f : \theta \mapsto tr(U(\theta)^\dagger H U(\theta) \rho)$
///
/// for some parameters $\theta$.
/// The parameter values are stored in an LUT for fast lookup, and the evaluation
/// of the mapping is parallelized. In order to make the evaluations faster,
/// the terms are structurally pruned to remove zero contributions.
///
/// Surrogate models can be saved to disk and loaded back into memory, allowing for
/// the reuse of the same model for different optimization runs.
///
/// This file contains both the trait definitions for the surrogate model, as well as
/// impls for Pauli and Majorana surrogate models.
///
use std::cell::RefCell;
use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::OpenOptions;

use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::symcoeff::CompiledCoeff;

/// One term of a compiled `SurrogateModel`: an overlap coefficient and the index of its
/// root op in the model's shared `CompiledCoeff` tape.
pub struct SurrogateTerm {
    pub overlap: f64,
    pub root: usize,
}

/// Sentinel `SurrogateTerm::root` value for a structurally-empty coefficient
const EMPTY_ROOT: usize = usize::MAX;

thread_local! {
    static EVAL_SCRATCH: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// Compiled output of a surrogate propagation run.
///
pub struct SurrogateModel {
    pub terms: Vec<SurrogateTerm>,
    pub tape: CompiledCoeff,
    pub n_params: usize,
}

impl SurrogateModel {
    /// Builds a model from its compiled terms, shared coefficient tape, and parameter count.
    pub fn new(terms: Vec<SurrogateTerm>, tape: CompiledCoeff, n_params: usize) -> Self {
        SurrogateModel { terms, tape, n_params }
    }

    /// Evaluate the expectation value for the given parameter angles.
    pub fn evaluate(&self, params: &[f64]) -> f64 {
        let lut = Self::make_lut(params);
        let results = self.tape.evaluate_all(&lut);
        self.terms
            .par_iter()
            .map(|t| if t.root == EMPTY_ROOT { 0.0 } else { t.overlap * results[t.root as usize] })
            .sum()
    }

    /// Evaluate for many parameter assignments at once. Parallelizes across
    /// assignments.
    pub fn evaluate_batch(&self, param_sets: &[Vec<f64>]) -> Vec<f64> {
        param_sets
            .par_iter()
            .map(|params| {
                let lut = Self::make_lut(params);
                EVAL_SCRATCH.with(|cell| {
                    let mut results = cell.borrow_mut();
                    self.tape.evaluate_into(&lut, &mut results);
                    self.terms
                        .iter()
                        .map(|t| if t.root == EMPTY_ROOT { 0.0 } else { t.overlap * results[t.root] })
                        .sum()
                })
            })
            .collect()
    }

    /// Build a lookup table of cos/sin values for the given parameter angles.
    fn make_lut(params: &[f64]) -> Vec<f64> {
        params.iter().flat_map(|&t| [t.cos(), t.sin()]).collect()
    }

    /// Number of compiled terms.
    pub fn n_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total pre-dedup monomial-instance count across every surviving term
    /// (an upper bound, not deduplicated)
    pub fn n_monomials(&self) -> u128 {
        let counts = self.tape.monomial_counts();
        self.terms
            .iter()
            .filter(|t| t.root != EMPTY_ROOT)
            .fold(0u128, |acc, t| acc.saturating_add(counts[t.root]))
    }

    /// Save to a binary file (see the module-level format notes on
    /// `MAGIC`/`FORMAT_VERSION`).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let target_shards = rayon::current_num_threads().max(1);

        let tape_blobs: Vec<Vec<u8>> = self
            .tape
            .serialize_shards_with(target_shards, gzip_block)
            .into_iter()
            .collect::<std::io::Result<_>>()?;

        // Contiguous shards, one per worker (at least one term each). Each shard
        // is serialized + compressed on its own thread; blobs come back in term
        // order because `par_chunks` preserves order.
        let n_terms = self.terms.len();
        let chunk = n_terms.div_ceil(target_shards).max(1);
        let term_blobs: Vec<Vec<u8>> = self
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
        w.write_all(&(tape_blobs.len() as u64).to_le_bytes())?;
        for b in &tape_blobs {
            w.write_all(&(b.len() as u64).to_le_bytes())?;
        }
        for b in &tape_blobs {
            w.write_all(b)?;
        }
        w.write_all(&(term_blobs.len() as u64).to_le_bytes())?;
        for b in &term_blobs {
            w.write_all(&(b.len() as u64).to_le_bytes())?;
        }
        for b in &term_blobs {
            w.write_all(b)?;
        }
        w.flush()?;
        Ok(())
    }

    /// Load from a file produced by `save`. The header and compressed blobs
    /// are read sequentially, then both the tape shards and the term shards
    /// are decompressed/parsed in parallel and reassembled (in shard order).
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

        let n_tape_shards = read_u64!() as usize;
        let mut tape_shard_lens = Vec::with_capacity(n_tape_shards);
        for _ in 0..n_tape_shards {
            tape_shard_lens.push(read_u64!() as usize);
        }
        let mut tape_blobs: Vec<Vec<u8>> = Vec::with_capacity(n_tape_shards);
        for len in tape_shard_lens {
            let mut blob = vec![0u8; len];
            r.read_exact(&mut blob)?;
            tape_blobs.push(blob);
        }
        // Decompress AND parse each shard in one fused step per shard.
        let tape_shards: Vec<CompiledCoeff> = tape_blobs
            .into_par_iter()
            .map(|blob| -> std::io::Result<CompiledCoeff> {
                let mut raw = Vec::new();
                GzDecoder::new(&blob[..]).read_to_end(&mut raw)?;
                let mut pos = 0usize;
                Ok(CompiledCoeff::deserialize(&raw, &mut pos))
            })
            .collect::<std::io::Result<_>>()?;
        let tape = CompiledCoeff::concat(tape_shards);

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

        let per_shard: Vec<Vec<SurrogateTerm>> = blobs
            .par_iter()
            .map(|blob| parse_shard(blob))
            .collect::<std::io::Result<_>>()?;

        let mut terms = Vec::with_capacity(per_shard.iter().map(|s| s.len()).sum());
        for shard in per_shard {
            terms.extend(shard);
        }

        Ok(SurrogateModel { terms, tape, n_params })
    }
}

/// Magic bytes and format version stamped at the head of every saved model.
const MAGIC: u32 = u32::from_le_bytes(*b"PQSM");
const FORMAT_VERSION: u32 = 9;

/// Serialize one term into `buf` (uncompressed)
fn write_term_into(buf: &mut Vec<u8>, st: &SurrogateTerm) {
    buf.extend_from_slice(&st.overlap.to_le_bytes());
    buf.extend_from_slice(&(st.root as u64).to_le_bytes());
}

/// gzip a shard's raw bytes into a self-contained compressed blob.
fn gzip_block(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)?;
    enc.finish()
}

/// Decompress and parse one shard blob into its terms (the inverse of
/// `write_term_into`), consuming the whole decompressed buffer.
fn parse_shard(compressed: &[u8]) -> std::io::Result<Vec<SurrogateTerm>> {
    let mut raw = Vec::new();
    GzDecoder::new(compressed).read_to_end(&mut raw)?;

    #[inline]
    fn rd_f64(b: &[u8], pos: &mut usize) -> f64 {
        let v = f64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v
    }
    #[inline]
    fn rd_root(b: &[u8], pos: &mut usize) -> usize {
        let v = u64::from_le_bytes(b[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        v as usize
    }

    let mut terms = Vec::new();
    let mut pos = 0usize;
    while pos < raw.len() {
        let overlap = rd_f64(&raw, &mut pos);
        let root = rd_root(&raw, &mut pos);
        terms.push(SurrogateTerm { overlap, root });
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
    pub(crate) inner: SurrogateModel,
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
        let inner = SurrogateModel::load(path)
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

    /// Total pre-dedup monomial-instance count across every surviving term
    /// (an upper bound, not deduplicated: `n_terms` alone doesn't say how
    /// much underlying computation a term represents).
    #[getter]
    fn n_monomials(&self) -> u128 {
        self.inner.n_monomials()
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
    pub(crate) inner: SurrogateModel,
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
        let inner = SurrogateModel::load(path)
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

    #[getter]
    fn n_monomials(&self) -> u128 {
        self.inner.n_monomials()
    }

    fn __repr__(&self) -> String {
        format!(
            "MajoranaSurrogateModel(n_terms={}, n_params={})",
            self.inner.n_terms(), self.inner.n_params
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symcoeff::{GateParam, SymbolicCoeff};
    use num_complex::Complex64;
    use propaq_core::coeff::CoeffRepr;

    fn build_shared_model() -> (SurrogateModel, Vec<SymbolicCoeff>, Vec<f64>) {
        let phase = Complex64::new(0.0, -1.0);
        let mut base = SymbolicCoeff::from_scalar(1.0);
        let _ = base.apply_rotation(&GateParam::symbolic(0), phase);

        let overlaps = [1.5f64, -0.5, 2.0];
        let coeffs: Vec<SymbolicCoeff> = overlaps
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let mut c = base.clone();
                let _ = c.apply_rotation(&GateParam::symbolic(1 + i as u32), phase);
                c
            })
            .collect();

        let (tape, roots) = SymbolicCoeff::compile_batch(coeffs.clone());
        let terms: Vec<SurrogateTerm> = overlaps
            .iter()
            .zip(&roots)
            .map(|(&overlap, &root)| SurrogateTerm { overlap, root })
            .collect();

        (SurrogateModel::new(terms, tape, 4), coeffs, overlaps.to_vec())
    }

    #[test]
    fn evaluate_matches_the_old_per_term_compile_algorithm() {
        let (model, coeffs, overlaps) = build_shared_model();
        let params = [0.3, 0.7, 1.1, 1.9];
        let lut = SurrogateModel::make_lut(&params);

        let expected: f64 = overlaps
            .iter()
            .zip(&coeffs)
            .map(|(&overlap, c)| overlap * c.compile().evaluate(&lut))
            .sum();

        let got = model.evaluate(&params);
        assert!((got - expected).abs() < 1e-12, "got {got}, expected {expected}");
    }

    #[test]
    fn n_monomials_matches_the_original_node_count_sum() {
        let (model, coeffs, _overlaps) = build_shared_model();
        let expected: u128 = coeffs.iter().map(|c| c.monomial_count()).sum();
        assert_eq!(model.n_monomials(), expected);
    }

    #[test]
    fn n_monomials_saturates_instead_of_wrapping_when_summing_many_huge_terms() {
        let mut coeffs = Vec::new();
        for i in 0..3 {
            let mut c = SymbolicCoeff::from_scalar(1.0 + i as f64);
            for _ in 0..135 {
                let other = c.clone();
                c.add_assign(other);
            }
            assert_eq!(c.monomial_count(), u128::MAX, "each term must itself be saturated already");
            coeffs.push(c);
        }
        let (tape, roots) = SymbolicCoeff::compile_batch(coeffs);
        let terms: Vec<SurrogateTerm> =
            roots.iter().map(|&root| SurrogateTerm { overlap: 1.0, root }).collect();
        let model = SurrogateModel::new(terms, tape, 1);
        assert_eq!(model.n_monomials(), u128::MAX);
    }

    #[test]
    fn save_load_round_trips_evaluate_output() {
        let (model, _coeffs, _overlaps) = build_shared_model();
        let params = [0.3, 0.7, 1.1, 1.9];
        let before = model.evaluate(&params);

        let path = std::env::temp_dir()
            .join(format!("propaq_surrogate_model_test_{}.bin", std::process::id()));
        let path_str = path.to_str().unwrap();
        model.save(path_str).expect("save should succeed");
        let loaded = SurrogateModel::load(path_str).expect("load should succeed");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.n_params, model.n_params);
        assert_eq!(loaded.n_terms(), model.n_terms());
        let after = loaded.evaluate(&params);
        assert!((after - before).abs() < 1e-12, "round-tripped {after} vs original {before}");
    }
}
