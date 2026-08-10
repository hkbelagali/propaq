//!
//! Persisting a term map to disk and reading it back.
//!
use std::fs::OpenOptions;
use std::io::{BufReader, BufWriter, Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

use crate::traits::AbstractTerm;

/// Serialize `terms` to a gzip-compressed binary file at `path`.
///
/// Format (all integers little-endian):
///   u64  n_terms
///   u64  key_stride    (bytes per key; 0 when n_terms == 0)
///   u64  system_size   (n_qubits for Pauli, n_modes for Majorana)
///   For each term:
///     [u8; key_stride]  key bytes from AbstractTerm::to_bytes_vec()
///     f64               coefficient (real)
pub fn save_terms_to_file<M: AbstractTerm>(terms: &FxHashMap<M, f64>, path: &str) -> PyResult<()> {
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
