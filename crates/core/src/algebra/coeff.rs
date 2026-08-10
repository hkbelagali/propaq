//!
//! Coefficient representation trait and implementations for basis strings.
//! A generic trait `CoeffRepr` represents numerical coefficients of varying
//! precisions, as well as symbolic coefficients. Many functions are
//! inlined to invoke zero-cost abstractions for performance.
//!

use num_complex::Complex64;
use pyo3::prelude::*;

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

    /// Additive merge: `self += other`. Used when duplicate terms fold into
    /// the same store row.
    fn add_assign(&mut self, other: Self);

    /// Apply a non-commuting rotation.
    ///
    /// Modifies `self` in-place for the cos branch and returns a new value for
    /// the sin branch. `phase` is the product phase from `AbstractTerm::matmul_internal`.
    fn apply_rotation(&mut self, param: &Self::GateParam, phase: Complex64) -> Self;

    /// `(sin(theta), cos(theta))` for this gate. This avoids recomputing
    /// the same trig functions for every term in the pool.
    /// Returns `None` if the representation does not support this optimization,
    /// which is the default behavior.
    #[inline]
    fn rotation_factors(_param: &Self::GateParam) -> Option<(f64, f64)> {
        None
    }

    /// The magnitude the sine branch will carry, without forming it.
    #[inline]
    fn sin_branch_magnitude(&self, _sin: f64) -> f64 {
        f64::INFINITY
    }

    /// Apply a rotation with memoized trig factors.
    #[inline]
    fn apply_rotation_with(
        &mut self,
        param: &Self::GateParam,
        _factors: Option<(f64, f64)>,
        phase: Complex64,
    ) -> Self {
        self.apply_rotation(param, phase)
    }

    /// Multiply all scalar components by a real noise damping factor.
    fn scale_real(&mut self, factor: f64);

    #[inline]
    fn size_hint(&self) -> u128 {
        1
    }

    #[inline]
    fn prefetch_read(&self) {}

    #[inline]
    fn passes_coeff_cutoff(&self, _cutoff: f64) -> bool {
        true
    }

    /// The coefficient's absolute value.
    #[inline]
    fn magnitude(&self) -> f64 {
        0.0
    }

    /// Widen this coefficient to its real f64 value
    #[inline]
    fn to_f64(&self) -> f64 {
        self.magnitude()
    }

    /// Whether or not we can use the Clifford fast path.
    #[inline]
    fn is_clifford_param(_param: &Self::GateParam, _eps: f64) -> bool {
        false
    }

    #[inline]
    fn phase_only_scale(_param: &Self::GateParam, _eps: f64) -> Option<f64> {
        None
    }

    #[inline]
    fn clifford_branch_sign(_param: &Self::GateParam, _phase: Complex64) -> Option<f64> {
        None
    }

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
        debug_assert!(
            phase.re.abs() < 1e-9,
            "rotation phase must be +- i for real coefficients"
        );
        let sin_branch = *self * sin_t * (-phase.im);
        *self *= cos_t;
        sin_branch
    }

    #[inline]
    fn rotation_factors(param: &f64) -> Option<(f64, f64)> {
        Some(param.sin_cos())
    }

    #[inline]
    fn sin_branch_magnitude(&self, sin: f64) -> f64 {
        (*self * sin).abs()
    }

    #[inline]
    fn apply_rotation_with(
        &mut self,
        param: &f64,
        factors: Option<(f64, f64)>,
        phase: Complex64,
    ) -> Self {
        let (sin_t, cos_t) = match factors {
            Some(f) => f,
            None => param.sin_cos(),
        };
        debug_assert!(
            phase.re.abs() < 1e-9,
            "rotation phase must be +- i for real coefficients"
        );
        let sin_branch = *self * sin_t * (-phase.im);
        *self *= cos_t;
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        *self *= factor;
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
        obj.getattr("angle")?.extract()
    }

    #[inline]
    fn passes_coeff_cutoff(&self, cutoff: f64) -> bool {
        self.abs() >= cutoff
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        self.abs()
    }

    #[inline]
    fn to_f64(&self) -> f64 {
        *self
    }

    #[inline]
    fn is_clifford_param(angle: &f64, eps: f64) -> bool {
        angle.cos().abs() < eps
    }

    #[inline]
    fn phase_only_scale(angle: &f64, eps: f64) -> Option<f64> {
        let (sin_t, cos_t) = angle.sin_cos();
        (sin_t.abs() < eps).then_some(cos_t)
    }

    #[inline]
    fn clifford_branch_sign(angle: &f64, phase: Complex64) -> Option<f64> {
        Some(angle.sin() * (-phase.im))
    }
}

/// Single-precision numerical coefficient.
impl CoeffRepr for f32 {
    type GateParam = f64;

    #[inline]
    fn from_real(c: f64) -> Self {
        c as f32
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }

    #[inline]
    fn apply_rotation(&mut self, angle: &f64, phase: Complex64) -> Self {
        let (sin_t, cos_t) = angle.sin_cos();
        debug_assert!(
            phase.re.abs() < 1e-9,
            "rotation phase must be +- i for real coefficients"
        );
        let sin_branch = *self * (sin_t as f32) * (-phase.im as f32);
        *self *= cos_t as f32;
        sin_branch
    }

    #[inline]
    fn rotation_factors(param: &f64) -> Option<(f64, f64)> {
        Some(param.sin_cos())
    }

    #[inline]
    fn sin_branch_magnitude(&self, sin: f64) -> f64 {
        // Narrowed and multiplied exactly where the branch is, then widened.
        ((*self * (sin as f32)) as f64).abs()
    }

    #[inline]
    fn apply_rotation_with(
        &mut self,
        param: &f64,
        factors: Option<(f64, f64)>,
        phase: Complex64,
    ) -> Self {
        let (sin_t, cos_t) = match factors {
            Some(f) => f,
            None => param.sin_cos(),
        };
        debug_assert!(
            phase.re.abs() < 1e-9,
            "rotation phase must be +- i for real coefficients"
        );
        // Narrowed exactly where `apply_rotation` narrows, so the two paths
        // round identically and the precheck stays exact.
        let sin_branch = *self * (sin_t as f32) * (-phase.im as f32);
        *self *= cos_t as f32;
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        *self *= factor as f32;
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
        obj.getattr("angle")?.extract()
    }

    #[inline]
    fn passes_coeff_cutoff(&self, cutoff: f64) -> bool {
        (*self as f64).abs() >= cutoff
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        (*self as f64).abs()
    }

    #[inline]
    fn to_f64(&self) -> f64 {
        *self as f64
    }

    #[inline]
    fn is_clifford_param(angle: &f64, eps: f64) -> bool {
        angle.cos().abs() < eps
    }

    #[inline]
    fn phase_only_scale(angle: &f64, eps: f64) -> Option<f64> {
        let (sin_t, cos_t) = angle.sin_cos();
        (sin_t.abs() < eps).then_some(cos_t)
    }

    #[inline]
    fn clifford_branch_sign(angle: &f64, phase: Complex64) -> Option<f64> {
        Some(angle.sin() * (-phase.im))
    }
}
