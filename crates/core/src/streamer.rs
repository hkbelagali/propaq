///
/// Lazy loading of terms from a gzipped term file. 
/// 
/// It's common for propagation runs to produce 
/// hundreds of millions of terms. Although these 
/// can be stored in memory in cluster environments, 
/// propaq provides a mechanism to lazily iterate 
/// over terms from a gzipped file for post-processing.
///
use std::fs::File;
use std::io::{BufReader, Read};
use std::marker::PhantomData;

use flate2::read::GzDecoder;
use pyo3::prelude::*;

use crate::traits::AbstractTerm;

pub struct TermStreamer<M: AbstractTerm> {
    reader: BufReader<GzDecoder<File>>,
    system_size: u64,
    remaining: u64,
    key_buf: Vec<u8>,
    _phantom: PhantomData<M>,
}

impl<M: AbstractTerm> TermStreamer<M> {
    pub fn open(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mut reader = BufReader::new(GzDecoder::new(file));

        let io_err = |e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string());
        let mut u64_buf = [0u8; 8];

        reader.read_exact(&mut u64_buf).map_err(io_err)?;
        let n_terms = u64::from_le_bytes(u64_buf);

        reader.read_exact(&mut u64_buf).map_err(io_err)?;
        let key_stride = u64::from_le_bytes(u64_buf) as usize;

        reader.read_exact(&mut u64_buf).map_err(io_err)?;
        let system_size = u64::from_le_bytes(u64_buf);

        Ok(Self {
            reader,
            system_size,
            remaining: n_terms,
            key_buf: vec![0u8; key_stride],
            _phantom: PhantomData,
        })
    }
}

impl<M: AbstractTerm> Iterator for TermStreamer<M> {
    type Item = PyResult<(M, f64)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let io_err = |e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string());
        let mut f64_buf = [0u8; 8];

        if let Err(e) = self.reader.read_exact(&mut self.key_buf).map_err(io_err) {
            return Some(Err(e));
        }
        if let Err(e) = self.reader.read_exact(&mut f64_buf).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string())) {
            return Some(Err(e));
        }
        let coeff = f64::from_le_bytes(f64_buf);

        self.remaining -= 1;
        let term = M::from_bytes_vec(&self.key_buf, self.system_size);
        Some(Ok((term, coeff)))
    }
}
