use num_complex::Complex64;
use pyo3::prelude::*;
use smallvec::SmallVec;

use propaq_core::coeff::CoeffRepr;

/// Packed trig factor: bit 0 = is_sin, bits 1–31 = param_index.
/// Supports up to 2^31 ≈ 2 billion distinct parameters.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrigFactor(pub u32);

impl TrigFactor {
    #[inline]
    pub fn cos(idx: u32) -> Self {
        TrigFactor(idx << 1)
    }

    #[inline]
    pub fn sin(idx: u32) -> Self {
        TrigFactor((idx << 1) | 1)
    }

    #[inline]
    pub fn param_index(self) -> u32 {
        self.0 >> 1
    }

    #[inline]
    pub fn is_sin(self) -> bool {
        self.0 & 1 == 1
    }
}

/// A single term in a symbolic polynomial: `scalar * product(factors)`.
///
/// Uses `SmallVec<[TrigFactor; 8]>` for 8 inline factors (~32 bytes),
/// keeping the total struct size ~56 bytes and avoiding heap allocation
/// for circuits with ≤8 non-commuting rotations per term.
#[derive(Clone)]
pub struct Monomial {
    pub scalar: Complex64,
    pub factors: SmallVec<[TrigFactor; 8]>,
}

impl Monomial {
    fn new(scalar: Complex64) -> Self {
        Monomial { scalar, factors: SmallVec::new() }
    }
}

/// A sum of monomials: represents a symbolic coefficient accumulated
/// during surrogate propagation.
///
/// `add_assign` simply appends monomials; call `deduplicate` to merge
/// identical factor patterns and drop near-zero terms before evaluation.
#[derive(Clone, Default)]
pub struct SymbolicCoeff {
    pub monomials: Vec<Monomial>,
}

impl SymbolicCoeff {
    /// Single scalar monomial with no trig factors (used to seed from observable).
    pub fn from_scalar(c: Complex64) -> Self {
        SymbolicCoeff { monomials: vec![Monomial::new(c)] }
    }

    /// Push a cos(param_idx) factor onto every existing monomial.
    pub fn multiply_cos(&mut self, idx: u32) {
        for m in &mut self.monomials {
            m.factors.push(TrigFactor::cos(idx));
        }
    }

    /// Clone self, multiply each scalar by `phase`, push sin(param_idx).
    pub fn branch_sin(&self, idx: u32, phase: Complex64) -> Self {
        let monomials = self.monomials.iter().map(|m| {
            let mut factors = m.factors.clone();
            factors.push(TrigFactor::sin(idx));
            Monomial { scalar: m.scalar * phase, factors }
        }).collect();
        SymbolicCoeff { monomials }
    }

    /// Multiply all scalars by a real factor (for noise damping).
    pub fn scale(&mut self, factor: f64) {
        for m in &mut self.monomials {
            m.scalar *= factor;
        }
    }

    /// Drop monomials with frequency (factor count) > max_freq.
    pub fn trim_high_frequency(&mut self, max_freq: usize) {
        self.monomials.retain(|m| m.factors.len() <= max_freq);
    }

    /// Sort factors within each monomial, sort monomials lexicographically,
    /// merge adjacent terms with identical factor patterns, and drop near-zero.
    pub fn deduplicate(&mut self) {
        if self.monomials.len() <= 1 {
            return;
        }
        for m in &mut self.monomials {
            m.factors.sort_unstable();
        }
        self.monomials.sort_unstable_by(|a, b| a.factors.cmp(&b.factors));

        let mut out: Vec<Monomial> = Vec::with_capacity(self.monomials.len());
        for m in self.monomials.drain(..) {
            if let Some(last) = out.last_mut() {
                if last.factors == m.factors {
                    last.scalar += m.scalar;
                    continue;
                }
            }
            out.push(m);
        }
        out.retain(|m| m.scalar.norm() > 1e-15);
        self.monomials = out;
    }

    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Evaluate at the given (cos, sin) lookup table indexed by param_index.
    pub fn evaluate(&self, cos_sin: &[(f64, f64)]) -> Complex64 {
        self.monomials.iter().map(|m| {
            let prod: f64 = m.factors.iter().map(|&f| {
                let (c, s) = cos_sin[f.param_index() as usize];
                if f.is_sin() { s } else { c }
            }).product();
            m.scalar * prod
        }).sum()
    }

    /// Maximum factor count across all monomials; 0 if empty.
    pub fn max_frequency(&self) -> usize {
        self.monomials.iter().map(|m| m.factors.len()).max().unwrap_or(0)
    }
}

impl CoeffRepr for SymbolicCoeff {
    /// Gate parameter is a parameter index (u32).
    type GateParam = u32;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        SymbolicCoeff::from_scalar(c)
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.monomials.extend(other.monomials);
    }

    #[inline]
    fn apply_rotation(&mut self, idx: &u32, phase: Complex64) -> Self {
        // sin branch: clone * (i * phase), push sin factor
        let sin_branch = self.branch_sin(*idx, Complex64::new(0.0, 1.0) * phase);
        // cos branch (self): push cos factor in-place
        self.multiply_cos(*idx);
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        self.scale(factor);
    }

    /// L1 norm is undefined for symbolic; return 0 to skip coeff-based truncation.
    #[inline]
    fn l1_norm(&self) -> f64 {
        0.0
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<u32> {
        obj.getattr("param_index")?.extract()
    }
}
