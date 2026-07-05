///
/// Coefficient representation trait and implementations for Pauli/Majorana strings. 
/// propaq currently supports two coefficient types - numerical and symbolic. 
/// For generalizability and performance, the coefficient representation is 
/// abstract into a trait `CoeffRepr`, which is implemented concretely for
/// `f64` (numerical) and `SymbolicCoeff` (symbolic). Doing so
/// allows one to easily extend the library to a new basis if and when required!
///
/// Numerical coefficients are represented as `f64`: propaq only propagates
/// Hermitian operators, so coefficients remain real throughout a run. The
/// Pauli/Majorana product *phase* fed into `apply_rotation` is still a genuine
/// 4th root of unity (`±1, ±i`) and stays `Complex64`; `i * phase` collapses to
/// a real number for the anticommuting terms that branch.
///
/// As many of the operations on coefficients are on the hot path, 
/// as many methods as possible should be `#[inline]`-able for performance. 
 
use pyo3::prelude::*;
use num_complex::Complex64;

/// Coefficient type carried by Pauli/Majorana terms during propagation.
///
/// `Default` must be the additive identity (zero).
///
pub trait CoeffRepr: Clone + Send + Sync + Default + 'static {
    /// Gate parameter type: `f64` (angle) for numerical mode,
    /// `u32` (parameter index) for surrogate mode.
    type GateParam: Clone + Send + Sync;

    /// Additive identity. Matches `Default::default()` by convention.
    #[inline]
    fn zero() -> Self {
        Default::default()
    }

    /// Convert a real coefficient into this internal representation.
    /// For `f64`, this just returns itself, and for
    /// `SymbolicCoeff`,  this wraps it in a monomial's scalar field.
    fn from_real(c: f64) -> Self;

    /// Additive merge: `self += other`. Used when inserting from inboxes into
    /// the thread map and two entries arrive at the same Pauli string.
    fn add_assign(&mut self, other: Self);

    /// Called after `add_assign` during a periodic (non-truncating) outbox
    /// merge, so a coefficient can collapse any structure that just became
    /// mergeable while doing so is still cheap. No-op by default.
    // 
    // `f64` is already merged exactly by `add_assign`. `SymbolicCoeff` overrides
    /// this to call `deduplicate()`. Without it, monomials whose masks
    /// happen to coincide (e.g. every monomial produced by a purely-numeric
    /// gate history shares the same empty mask) would otherwise pile up as
    /// separate entries until the next full truncation flush.
    #[inline]
    fn post_merge(&mut self) {}

    /// Apply a non-commuting Pauli rotation.
    ///
    /// Modifies `self` in-place for the cos branch and returns a new value for
    /// the sin branch. `phase` is the Pauli product phase from
    /// `AbstractTerm::matmul_internal`.
    fn apply_rotation(&mut self, param: &Self::GateParam, phase: Complex64) -> Self;

    /// Multiply all scalar components by a real noise damping factor.
    fn scale_real(&mut self, factor: f64);

    /// L1 norm for discard statistics in verbose logging.
    /// Returns `0.0` for symbolic representations where the norm is undefined.
    fn l1_norm(&self) -> f64;

    /// How many indivisible units this coefficient represents. 
    // `1` for scalar representations (the default)
    // `SymbolicCoeff` overrides this with its monomial count,
    /// as this is the primary driver of memory usage for surrogate propagation.
    #[inline]
    fn size_hint(&self) -> usize {
        1
    }

    /// Prefetch any out-of-line storage this coefficient owns. Called by the
    /// flush merge loop a few entries ahead of use, so buffer cache misses
    /// overlap with the current entry's work. No-op by default (scalar
    /// coefficients have no out-of-line data).
    /// 
    /// `SymbolicCoeff` overrides this to prefetch its monomial vector.
    #[inline]
    fn prefetch_read(&self) {}

    /// Extract the gate parameter from a Python rotation object.
    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<Self::GateParam>;
}

/// Implementation for the numerical coefficient representation.
impl CoeffRepr for f64 {
    type GateParam = f64;

    #[inline]
    fn from_real(c: f64) -> Self {
        c
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }

    #[inline]
    fn apply_rotation(&mut self, angle: &f64, phase: Complex64) -> Self {
        let (sin_t, cos_t) = angle.sin_cos();
        // For real coefficients the rotation phase is always ±i (the
        // anticommuting terms that branch); `i * phase` is then real and equals
        // `-phase.im`. A phase of ±1 would produce an imaginary contribution,
        // which cannot occur for Hermitian generators.
        debug_assert!(phase.re.abs() < 1e-9, "rotation phase must be ±i for real coefficients");
        let sin_branch = *self * sin_t * (-phase.im);
        *self *= cos_t;
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        *self *= factor;
    }

    #[inline]
    fn l1_norm(&self) -> f64 {
        self.abs()
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
        obj.getattr("angle")?.extract()
    }
}
