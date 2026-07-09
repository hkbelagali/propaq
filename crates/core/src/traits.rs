///
/// An abstract class for $\sum_i c_i B_i$ 
/// where $B_i$ is a basis element belgonging to 
/// an operator basis (usually Pauli or Majorana) 
/// and $c_i$ is a coefficient. 
///
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

    /// Compile-time gate for `apply_gate_inplace`'s batched code path. `false`
    /// (the default) means the propagator never touches `GateCtx`/`matmul_batch`
    /// for this term type — its compiled code is textually identical to the
    /// unbatched per-item path. `MajoranaMonomial` overrides this to `true`.
    const SUPPORTS_BATCHING: bool = false;

    /// Per-gate context prepared once from the fixed generator, reused
    /// read-only across every batch for that gate. `()` for term types that
    /// don't override `SUPPORTS_BATCHING`.
    type GateCtx: Send + Sync;
    fn prepare_gate_ctx(&self) -> Self::GateCtx;

    /// Batched `self @ terms[i]`: appends `(i, phase, product)` to `out`
    /// (cleared first) for every `terms[i]` that anticommutes with `self`
    /// (commuting terms are skipped, not appended). `terms: &[&Self]` is
    /// deliberate — batching never clones a whole term, only gathers pointers.
    ///
    /// Default: a plain loop calling `commutes_with`+`matmul_internal` per
    /// item — correct for any implementation, and only actually invoked by
    /// the propagator when `SUPPORTS_BATCHING` is `true`, so leaving this
    /// unoverridden (as `PauliString` does) has zero cost.
    fn matmul_batch(&self, _ctx: &Self::GateCtx, terms: &[&Self], out: &mut Vec<(usize, Complex64, Self)>) {
        out.clear();
        for (i, &term) in terms.iter().enumerate() {
            if !self.commutes_with(term) {
                let (phase, product) = self.matmul_internal(term);
                out.push((i, phase, product));
            }
        }
    }
}
