///
/// Coefficient representation trait and implementations for Pauli/Majorana strings.
/// propaq currently supports numerical (f64, f32) and symbolic coefficient types.
/// For generalizability and performance, the coefficient representation is
/// abstract into a trait `CoeffRepr`, which is implemented concretely for
/// `f64` and `f32` (numerical) and `SymbolicCoeff` (symbolic). Doing so
/// allows one to easily extend the library to a new basis if and when required!
///
/// Numerical coefficients are real: propaq only propagates Hermitian
/// operators, so coefficients remain real throughout a run.
///
/// As many of the operations on coefficients are on the hot path, 
/// as many methods as possible should be `#[inline]`-able for performance. 
///
use pyo3::prelude::*;
use num_complex::Complex64;

///
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
    #[inline]
    fn post_merge(&mut self) {}

    /// Apply a non-commuting rotation.
    ///
    /// Modifies `self` in-place for the cos branch and returns a new value for
    /// the sin branch. `phase` is the product phase from `AbstractTerm::matmul_internal`.
    fn apply_rotation(&mut self, param: &Self::GateParam, phase: Complex64) -> Self;

    /// Multiply all scalar components by a real noise damping factor.
    fn scale_real(&mut self, factor: f64);

    #[inline]
    fn size_hint(&self) -> u128 {
        1
    }

    #[inline]
    fn prefetch_read(&self) {}

    #[inline]
    fn passes_coeff_cutoff(&self, _cutoff: f64) -> bool { true }

    #[inline]
    fn magnitude(&self) -> f64 { 0.0 }

    /// Widen this coefficient to its real f64 value (signed, not abs).
    /// Only meaningful for numerical reprs; defaults to magnitude.
    #[inline]
    fn to_f64(&self) -> f64 { self.magnitude() }

    #[inline]
    fn is_clifford_param(_param: &Self::GateParam, _eps: f64) -> bool { false }

    /// The other half of the Clifford-angle condition. A Pauli rotation `exp(-i*theta*G)` is
    /// Clifford exactly when `theta` is a multiple of `pi/2`; `is_clifford_param` covers
    /// `cos(theta) == 0` (theta = pi/2 mod pi, where the rotation maps an anticommuting term
    /// wholly onto `G*term`), and this covers `sin(theta) == 0` (theta = 0 mod pi), where the
    /// rotation leaves every term's Pauli content alone and merely scales anticommuting terms
    /// by `cos(theta)`, which is `+-1`.
    ///
    /// Returns `Some(cos(theta))` in that case, `None` otherwise. This matters far more than it
    /// looks: without it, a `theta = pi` rotation takes the generic splitting path, which
    /// counts the anticommuting rows, appends a brand-new row for every one of them, writes a
    /// `sin(pi) == 0` coefficient into each, and then relies on the next truncation to delete
    /// them all. Qiskit's transpiler emits exactly such rotations routinely (a Hadamard lowers
    /// to one `cos == 0` rotation plus one `theta = pi` rotation), so this is a hot case, not a
    /// corner case.
    ///
    /// Defaults to `None` so symbolic/surrogate coefficients, whose `GateParam` is a parameter
    /// index with no numeric angle to test, keep the generic path unchanged.
    #[inline]
    fn phase_only_scale(_param: &Self::GateParam, _eps: f64) -> Option<f64> { None }

    /// The real scalar by which an exactly-Clifford rotation (`cos(theta) ~ 0`) multiplies an
    /// *anticommuting* term's coefficient: `sin(theta) * (-phase.im)`. Used to build the
    /// Clifford lookup table, whose whole point is to derive per-label constants once per gate
    /// instead of calling `apply_rotation` per row.
    ///
    /// `None` means "this representation cannot express that factor as a plain real scalar",
    /// which disables the lookup-table fast path and falls back to the generic per-row
    /// `apply_rotation`. That fallback is the *safe* answer and must stay reachable: the table
    /// previously obtained this factor as
    /// `C::from_real(1.0).apply_rotation(param, phase).to_f64()`, which silently yields `0.0`
    /// for any representation that does not override `to_f64`/`magnitude`. `SymbolicCoeff`
    /// does not, so every surrogate Clifford gate with a concrete angle was zeroing the entire
    /// observable. Deriving the scalar from `param`/`phase` directly, and refusing to guess
    /// when it is not available, is what makes that failure mode unrepresentable.
    #[inline]
    fn clifford_branch_sign(_param: &Self::GateParam, _phase: Complex64) -> Option<f64> { None }

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
        debug_assert!(phase.re.abs() < 1e-9, "rotation phase must be +- i for real coefficients");
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
        debug_assert!(phase.re.abs() < 1e-9, "rotation phase must be +- i for real coefficients");
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
        *self as f64
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
