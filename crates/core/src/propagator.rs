///
/// Shared propagator support: file I/O for term maps, the `PropagationResult`
/// pyclass, and tqdm progress-bar helpers used by both the columnar SoA
/// engine (`soa::propagator::SoaPropagator`) and the surrogate propagator
/// (`propaq_surrogate::propagator::SurrogatePropagator`).
///
/// This file used to also hold `AbstractPropagator<M, C>`, a hash-partition/
/// outbox engine: each thread owned a disjoint partition of the term space as
/// a hashmap, plus a set of outboxes for terms belonging to other partitions,
/// with gate application processing both and a parallel-transpose flush
/// merging outboxes into partition maps before truncation. The numerical
/// propagators were rewritten onto the columnar SoA engine (see `soa::mod`'s
/// doc comment) because that model paid a rayon fork/join and a hashmap
/// insert per gate per term; the surrogate propagator was the last remaining
/// consumer and has since been ported onto `SoaTermSum<SymbolicCoeff>` +
/// `soa::kernels` too (interning/reconcile, monomial-budget truncation, and
/// model build all now read/write columnar storage directly), so
/// `AbstractPropagator` and its partition-specific helpers were deleted.
///
use pyo3::prelude::*;
use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::OpenOptions;
use rustc_hash::FxHashMap;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;

use crate::traits::AbstractTerm;

/// Result returned by `expectation_value`: per-gate term counts and the final expectation value.
#[pyclass(module = "propaq._rust_core")]
pub struct PropagationResult {
    #[pyo3(get)]
    pub n_terms: Vec<usize>,
    #[pyo3(get)]
    pub expectation_value: f64,
}

#[pymethods]
impl PropagationResult {
    fn __repr__(&self) -> String {
        format!(
            "PropagationResult(expectation_value={}, n_terms=[{} entries])",
            self.expectation_value,
            self.n_terms.len()
        )
    }
}

/// tqdm progress bar helpers, shared by the hash-partition and SoA engines.
/// These claim and release the GIL since they are called from the main thread.
pub fn make_progress_bar(
    py: Python<'_>,
    enabled: bool,
    total: usize,
) -> PyResult<(Option<Py<PyAny>>, Option<Py<PyAny>>)> {
    if !enabled {
        return Ok((None, None));
    }
    py.import("warnings")?.call_method1(
        "warn",
        ("propaq: the progress bar term count stays stale between truncation \
          flushes. Reduce `truncation_threshold` for more frequent updates.",),
    )?;
    let tqdm = py.import("tqdm.auto")?;
    let postfix = pyo3::types::PyDict::new(py);
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("total", total)?;
    kwargs.set_item("desc", "Propagating through gates")?;
    let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
    Ok((Some(pbar.into()), Some(postfix.into())))
}

pub fn tick_progress_bar(
    py: Python<'_>,
    pbar: &Option<Py<PyAny>>,
    postfix: &Option<Py<PyAny>>,
    n_terms: usize,
) -> PyResult<()> {
    if let (Some(pbar), Some(postfix)) = (pbar, postfix) {
        let pbar = pbar.bind(py);
        let postfix = postfix.bind(py);
        postfix.set_item("terms", n_terms)?;
        pbar.call_method("set_postfix", (), Some(postfix.downcast()?))?;
        pbar.call_method0("update")?;
    }
    Ok(())
}

pub fn close_progress_bar(py: Python<'_>, pbar: &Option<Py<PyAny>>) -> PyResult<()> {
    if let Some(pbar) = pbar {
        pbar.bind(py).call_method0("close")?;
    }
    Ok(())
}

/// Serialize `terms` to a gzip-compressed binary file at `path`.
///
/// Format (all integers little-endian):
///   u64  n_terms
///   u64  key_stride    (bytes per key; 0 when n_terms == 0)
///   u64  system_size   (n_qubits for Pauli, n_modes for Majorana)
///   For each term:
///     [u8; key_stride]  key bytes from AbstractTerm::to_bytes_vec()
///     f64               coefficient (real)
pub fn save_terms_to_file<M: AbstractTerm>(
    terms: &FxHashMap<M, f64>,
    path: &str,
) -> PyResult<()> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());

    let n_terms = terms.len() as u64;
    enc.write_all(&n_terms.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let first = terms.keys().next();
    let key_stride: u64 = first.map_or(0, |t| t.to_bytes_vec().len() as u64);
    let system_size: u64 = first.map_or(0, |t| t.system_size());
    enc.write_all(&key_stride.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    enc.write_all(&system_size.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    for (term, coeff) in terms.iter() {
        enc.write_all(&term.to_bytes_vec())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        enc.write_all(&coeff.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    enc.finish()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

/// Deserialize a term map from a file produced by `save_terms_to_file`.
pub fn load_terms_from_file<M: AbstractTerm>(path: &str) -> PyResult<FxHashMap<M, f64>> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut dec = BufReader::new(GzDecoder::new(file));

    let mut u64_buf = [0u8; 8];
    let mut f64_buf = [0u8; 8];

    let io_err = |e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string());

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let n_terms = u64::from_le_bytes(u64_buf) as usize;

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let key_stride = u64::from_le_bytes(u64_buf) as usize;

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let system_size = u64::from_le_bytes(u64_buf);

    let mut terms = FxHashMap::default();
    terms.reserve(n_terms);
    let mut key_buf = vec![0u8; key_stride];

    for _ in 0..n_terms {
        dec.read_exact(&mut key_buf).map_err(io_err)?;
        dec.read_exact(&mut f64_buf).map_err(io_err)?;
        let coeff = f64::from_le_bytes(f64_buf);
        let term = M::from_bytes_vec(&key_buf, system_size);
        terms.insert(term, coeff);
    }
    Ok(terms)
}
