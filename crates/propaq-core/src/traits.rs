use std::hash::Hash;
use num_complex::Complex64;

pub trait AbstractTerm: Clone + PartialEq + Eq + Hash + Send + Sync + 'static {
    fn weight(&self) -> u32;
    fn commutes_with(&self, other: &Self) -> bool;
    fn matmul_internal(&self, other: &Self) -> (Complex64, Self);
    fn trace_with_fock_state(&self, fock_state: u64) -> f64;
    fn to_bytes_vec(&self) -> Vec<u8>;
    /// XOR-fold the term's underlying bit representation down to a single u64,
    /// used as input to the multiply-shift partition hash in the propagator.
    fn partition_key(&self) -> u64;
    /// Whether this term preserves particle number.
    /// Default is `false`; `MajoranaMonomial` overrides this field.
    fn is_number_preserving(&self) -> bool { false }
    /// The system size needed to reconstruct this term from bytes
    /// (n_qubits for Pauli, n_modes for Majorana).
    fn system_size(&self) -> u64;
    /// Reconstruct a term from the bytes produced by `to_bytes_vec` and the
    /// system size stored by `system_size`.
    fn from_bytes_vec(bytes: &[u8], system_size: u64) -> Self;
}
