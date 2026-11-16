///
/// Lazy loading of terms from a gzipped term file.
///
/// It's common for propagation runs to produce
/// hundreds of millions of terms. Although these
/// can be stored in memory in cluster environments,
/// propaq provides a mechanism to lazily iterate
/// over terms from a gzipped file for post-processing.
///
/// The streamer reads v2 files and legacy v0.1.8 files (see `term_io`).
///
use std::marker::PhantomData;

use pyo3::prelude::*;

use crate::basis_module::{open_term_file, BasisCodec};
use crate::term_io::{FileCoeff, TermFileReader};
use crate::traits::AbstractTerm;

pub struct TermStreamer<B: BasisCodec>
where
    B::Term: AbstractTerm,
{
    reader: TermFileReader,
    size: usize,
    _phantom: PhantomData<fn() -> B>,
}

impl<B: BasisCodec> TermStreamer<B>
where
    B::Term: AbstractTerm,
{
    /// Opens `path`. Fails with a clear error if the file is of another basis.
    pub fn open(path: &str) -> PyResult<Self> {
        let (reader, size) = open_term_file::<B>(path)?;
        Ok(Self { reader, size, _phantom: PhantomData })
    }
}

impl<B: BasisCodec> Iterator for TermStreamer<B>
where
    B::Term: AbstractTerm,
{
    type Item = PyResult<(B::Term, FileCoeff)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.reader.next_term() {
            Ok(None) => None,
            Ok(Some((words, coeff))) => Some(Ok((B::from_words(&words, self.size), coeff))),
            Err(e) => Some(Err(e)),
        }
    }
}
