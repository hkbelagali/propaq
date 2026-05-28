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
}
