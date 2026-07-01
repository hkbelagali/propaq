use pyo3::prelude::*;
use num_complex::Complex64;

/// Coefficient type carried by Pauli/Majorana terms during propagation.
///
/// `Default` must be the additive identity (zero).
/// All methods are expected to be `#[inline]`-able; they sit on the hot path.
pub trait CoeffRepr: Clone + Send + Sync + Default + 'static {
    /// Gate parameter type: `f64` (angle) for numerical mode,
    /// `u32` (parameter index) for surrogate mode.
    type GateParam: Clone + Send + Sync;

    /// Additive identity. Matches `Default::default()` by convention.
    #[inline]
    fn zero() -> Self {
        Default::default()
    }

    /// Convert a numerical initial coefficient (from the input observable) into
    /// this representation. For `Complex64` this is the identity. For
    /// `SymbolicCoeff` it wraps the value in a single scalar monomial.
    fn from_complex(c: Complex64) -> Self;

    /// Additive merge: `self += other`. Used when inserting from inboxes into
    /// the thread map and two entries arrive at the same Pauli string.
    fn add_assign(&mut self, other: Self);

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

    /// Extract the gate parameter from a Python rotation object.
    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<Self::GateParam>;
}

impl CoeffRepr for Complex64 {
    type GateParam = f64;

    #[inline]
    fn from_complex(c: Complex64) -> Self {
        c
    }

    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }

    #[inline]
    fn apply_rotation(&mut self, angle: &f64, phase: Complex64) -> Self {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        let sin_branch = *self * Complex64::new(0.0, sin_t) * phase;
        *self *= cos_t;
        sin_branch
    }

    #[inline]
    fn scale_real(&mut self, factor: f64) {
        *self *= factor;
    }

    #[inline]
    fn l1_norm(&self) -> f64 {
        self.norm()
    }

    fn extract_gate_param(obj: &Bound<'_, PyAny>) -> PyResult<f64> {
        obj.getattr("angle")?.extract()
    }
}
