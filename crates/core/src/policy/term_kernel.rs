//!
//! Custom noise and truncation models that require an
//! individual term's key are implemented as a plugin ABI.
//!

use crate::basis::BasisKind;

/// A wrapper on the term view for the plugin.
pub struct TermView<'a> {
    /// Which algebra the words belong to.
    pub basis_kind: BasisKind,
    /// The term's key, as raw storage words.
    pub words: &'a [u64],
    /// Qubits (Pauli) or modes (Majorana) the operator is sized for.
    pub n_units: usize,
    /// The term's weight, as the basis defines it.
    pub weight: u32,
}

/// Number of terms a batched call covers at once.
pub const KERNEL_BATCH: usize = 1024;

/// A custom noise model that reads each term's key.
pub trait NoiseKernel: Send + Sync {
    /// The multiplicative factor for this term's coefficient.
    fn factor(&self, term: TermView<'_>) -> f64;

    fn factor_batch(
        &self,
        basis_kind: BasisKind,
        words: &[u64],
        stride: usize,
        weights: &[u32],
        n_units: usize,
        out: &mut [f64],
    ) {
        for (i, &weight) in weights.iter().enumerate() {
            out[i] = self.factor(TermView {
                basis_kind,
                words: &words[i * stride..(i + 1) * stride],
                n_units,
                weight,
            });
        }
    }
}

/// A custom truncation predicate that reads each term's key.
pub trait TruncationKernel: Send + Sync {
    /// True if this term belongs in the store.
    fn keep(&self, term: TermView<'_>, coeff_magnitude: f64) -> bool;

    #[allow(clippy::too_many_arguments)]
    fn keep_batch(
        &self,
        _basis_kind: BasisKind,
        _words: &[u64],
        _stride: usize,
        _weights: &[u32],
        _n_units: usize,
        _coeff_magnitudes: &[f64],
        _out: &mut [u8],
    ) -> bool {
        false
    }
}
