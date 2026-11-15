//! Compile test: `define_basis_streamer!` expands for a minimal term type.

use num_complex::Complex64;
use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::traits::AbstractTerm;

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(from_py_object)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Tm(u64);

impl AbstractTerm for Tm {
    fn weight(&self) -> u32 {
        self.0.count_ones()
    }
    fn commutes_with(&self, other: &Self) -> bool {
        (self.0 & other.0).count_ones().is_multiple_of(2)
    }
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self) {
        (Complex64::new(1.0, 0.0), Tm(self.0 ^ other.0))
    }
    fn trace_with_diag_state(&self, _d: &Bitset) -> f64 {
        0.0
    }
    fn to_bytes_vec(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
    fn partition_key(&self) -> u64 {
        self.0
    }
    fn system_size(&self) -> u64 {
        64
    }
    fn from_bytes_vec(bytes: &[u8], _n: u64) -> Self {
        Tm(u64::from_le_bytes(bytes[..8].try_into().unwrap()))
    }
}

propaq_core::define_basis_streamer!(Tm, TmStreamer);

#[test]
fn the_streamer_class_exists() {
    assert!(std::mem::size_of::<TmStreamer>() > 0);
}
