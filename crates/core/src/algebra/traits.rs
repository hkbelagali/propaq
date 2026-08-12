use num_complex::Complex64;
///
/// An abstract class for \(\sum_i c_i B_i\)
/// where \(B_i\) is a basis element belgonging to
/// an operator basis (usually Pauli or Majorana)
/// and \(c_i\) is a coefficient.
///
use std::hash::Hash;

use crate::bitset::Bitset;

pub trait AbstractTerm: Clone + PartialEq + Eq + Hash + Send + Sync + 'static {
    fn weight(&self) -> u32;
    fn commutes_with(&self, other: &Self) -> bool;
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self);
    /// `fock_state` represents a single bitstring/Slater determinant
    fn trace_with_fock_state(&self, fock_state: &Bitset) -> f64;
    fn to_bytes_vec(&self) -> Vec<u8>;
    fn partition_key(&self) -> u64;
    /// Whether this term preserves particle number.
    fn is_number_preserving(&self) -> bool {
        false
    }
    fn system_size(&self) -> u64;
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self;
}
