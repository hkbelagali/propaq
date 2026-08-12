//!
//! Custom noise and truncation models that require an
//! individual term's key are implemented as a plugin ABI.
//!

use crate::basis::BasisKind;

/// What a plugin reads, and therefore how the engine may evaluate it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Depends(u32);

impl Depends {
    /// A function of term weight alone.
    pub const NONE: Self = Depends(0);
    /// Reads the term's key words.
    pub const KEY: Self = Depends(1 << 0);
    /// Reads the layer index / layer count.
    pub const LAYER: Self = Depends(1 << 1);
    /// Every bit this build understands.
    pub const KNOWN: u32 = 0b11;

    /// The declaration a plugin returned, if every bit is one we understand.
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::KNOWN != 0 {
            return None;
        }
        Some(Depends(bits))
    }

    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// True when the plugin reads the term's key words.
    #[inline]
    pub const fn key(self) -> bool {
        self.contains(Self::KEY)
    }

    /// True when the plugin reads the layer index.
    #[inline]
    pub const fn layer(self) -> bool {
        self.contains(Self::LAYER)
    }
}

impl std::ops::BitOr for Depends {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Depends(self.0 | rhs.0)
    }
}

/// Where in the circuit a call is being made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LayerContext {
    /// Zero-based index of the layer being applied.
    pub index: u32,
    /// Layers in the circuit.
    pub total: u32,
}

impl LayerContext {
    #[inline]
    pub const fn new(index: u32, total: u32) -> Self {
        LayerContext { index, total }
    }
}

/// A wrapper on the term view for the plugin.
pub struct TermView<'a> {
    /// Which algebra the words belong to.
    pub basis_kind: BasisKind,
    /// The term's key, as raw storage words. Empty when the caller is a
    /// weight-only path, which only happens for a plugin that declared no
    /// [`Depends::KEY`].
    pub words: &'a [u64],
    /// Qubits (Pauli) or modes (Majorana) the operator is sized for.
    pub n_units: usize,
    /// The term's weight, as the basis defines it.
    pub weight: u32,
    /// Where in the circuit this call is happening.
    pub layer: LayerContext,
}

/// Number of terms a batched call covers at once.
pub const KERNEL_BATCH: usize = 1024;

/// A custom noise model that reads each term's key.
pub trait NoiseKernel: Send + Sync {
    /// What this model reads, which decides how the engine evaluates it.
    fn depends(&self) -> Depends {
        Depends::NONE
    }

    /// The multiplicative factor for this term's coefficient.
    fn factor(&self, term: TermView<'_>) -> f64;

    #[allow(clippy::too_many_arguments)]
    fn factor_batch(
        &self,
        basis_kind: BasisKind,
        words: &[u64],
        stride: usize,
        weights: &[u32],
        n_units: usize,
        layer: LayerContext,
        out: &mut [f64],
    ) {
        for (i, &weight) in weights.iter().enumerate() {
            out[i] = self.factor(TermView {
                basis_kind,
                words: &words[i * stride..(i + 1) * stride],
                n_units,
                weight,
                layer,
            });
        }
    }
}

/// A custom truncation predicate that reads each term's key.
pub trait TruncationKernel: Send + Sync {
    /// What this model reads, which decides how the engine evaluates it.
    fn depends(&self) -> Depends {
        Depends::NONE
    }

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
        _layer: LayerContext,
        _out: &mut [u8],
    ) -> bool {
        false
    }
}
