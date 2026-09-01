//!
//! Truncation policy: the composable truncators the propagators run, the flat
//! config they resolve into, and the legacy single-policy `TruncationPolicy`
//! that decomposes into them.
//!
use pyo3::prelude::*;

use crate::native_truncator::NativeTruncator;

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncator {
    #[pyo3(get, set)]
    pub frequency: Option<usize>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl FrequencyTruncator {
    #[new]
    #[pyo3(signature = (frequency=None))]
    pub fn new(frequency: Option<usize>) -> Self {
        FrequencyTruncator { frequency }
    }
    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncator(frequency={})",
            self.frequency
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct CoefficientTruncator {
    #[pyo3(get, set)]
    pub coefficient: Option<f64>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl CoefficientTruncator {
    #[new]
    #[pyo3(signature = (coefficient=None))]
    pub fn new(coefficient: Option<f64>) -> Self {
        CoefficientTruncator { coefficient }
    }
    fn __repr__(&self) -> String {
        format!(
            "CoefficientTruncator(coefficient={})",
            self.coefficient
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

/// Drop whole Pauli/Majorana terms whose operator weight exceeds `weight`.
/// Applies to both propagators.
///
/// A term with weight `w` is exponentially unlikely in `w` to
/// contribute to the final state, which is why this is a useful
/// truncation criterion for larger circuits!
///
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct WeightTruncator {
    #[pyo3(get, set)]
    pub weight: Option<u32>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl WeightTruncator {
    #[new]
    #[pyo3(signature = (weight=None))]
    pub fn new(weight: Option<u32>) -> Self {
        WeightTruncator { weight }
    }
    fn __repr__(&self) -> String {
        format!(
            "WeightTruncator(weight={})",
            self.weight
                .map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

/// Term-count floor, below which truncation is suppressed.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TermBudget {
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl TermBudget {
    #[new]
    #[pyo3(signature = (min_terms=None))]
    pub fn new(min_terms: Option<usize>) -> Self {
        TermBudget { min_terms }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<usize>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!("TermBudget(min_terms={})", f(self.min_terms))
    }
}

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct Simplify {
    #[pyo3(get, set)]
    pub enabled: bool,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl Simplify {
    #[new]
    #[pyo3(signature = (enabled=true))]
    pub fn new(enabled: bool) -> Self {
        Simplify { enabled }
    }
    fn __repr__(&self) -> String {
        format!("Simplify(enabled={})", self.enabled)
    }
}

#[derive(Clone, FromPyObject)]
pub enum Truncator {
    Frequency(FrequencyTruncator),
    Coefficient(CoefficientTruncator),
    Weight(WeightTruncator),
    TermBudget(TermBudget),
    Simplify(Simplify),
    Native(NativeTruncator),
}

impl Truncator {
    pub fn to_object(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        match self {
            Truncator::Frequency(t) => t.clone().into_py_any(py),
            Truncator::Coefficient(t) => t.clone().into_py_any(py),
            Truncator::Weight(t) => t.clone().into_py_any(py),
            Truncator::TermBudget(t) => t.clone().into_py_any(py),
            Truncator::Simplify(t) => t.clone().into_py_any(py),
            Truncator::Native(t) => t.clone().into_py_any(py),
        }
    }

    pub fn is_surrogate_only(&self) -> bool {
        matches!(self, Truncator::Frequency(_) | Truncator::Simplify(_))
    }

    pub fn is_numerical_only(&self) -> bool {
        matches!(self, Truncator::Native(_))
    }
}

pub fn reject_surrogate_only(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(Truncator::is_surrogate_only) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "FrequencyTruncator/Simplify only apply to surrogate propagation, \
             use WeightTruncator / CoefficientTruncator / TermBudget with the numerical propagator",
        ));
    }
    Ok(())
}

pub fn reject_numerical_only(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(Truncator::is_numerical_only) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "NativeTruncator only applies to numerical propagation, use it with PauliPropagator/MajoranaPropagator instead",
        ));
    }
    Ok(())
}

#[derive(Default, Clone)]
pub struct ResolvedConfig {
    pub frequency: Option<usize>,
    pub coefficient: Option<f64>,
    pub weight: Option<u32>,
    pub min_terms: Option<usize>,
    pub simplify: bool,
    /// When set, fully replaces the weight/coefficient cutoff comparison
    /// in `kernels::truncate` with the plugin's own per-term decision.
    pub native: Option<NativeTruncator>,
}

/// Collapse a truncator pipeline into a flat config.
pub fn resolve_config(truncators: &[Truncator]) -> ResolvedConfig {
    let mut r = ResolvedConfig::default();
    for t in truncators {
        match t {
            Truncator::Frequency(x) => r.frequency = x.frequency,
            Truncator::Coefficient(x) => r.coefficient = x.coefficient,
            Truncator::Weight(x) => r.weight = x.weight,
            Truncator::TermBudget(x) => r.min_terms = x.min_terms,
            Truncator::Simplify(x) => r.simplify = x.enabled,
            Truncator::Native(x) => r.native = Some(x.clone()),
        }
    }
    r
}

/// Controls when and how terms are discarded during propagation.
///
/// Arguments:
///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
///         None disables weight-based truncation.
///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
///     min_terms: Live-term floor below which truncation is suppressed.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TruncationPolicy {
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    #[pyo3(get, set)]
    pub coeff_cutoff: f64,
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl TruncationPolicy {
    /// Initialize the truncation policy.
    ///
    /// Arguments:
    ///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
    ///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
    ///     min_terms: Live-term floor below which truncation is suppressed.
    #[new]
    #[pyo3(signature = (weight_cutoff=None, coeff_cutoff=0.0, min_terms=None))]
    fn new(weight_cutoff: Option<u32>, coeff_cutoff: f64, min_terms: Option<usize>) -> Self {
        TruncationPolicy {
            weight_cutoff,
            coeff_cutoff,
            min_terms,
        }
    }

    /// Return True if a term with *weight* and |coefficient| *abs_coeff* should be discarded.
    fn should_truncate(&self, weight: u32, abs_coeff: f64) -> bool {
        self.weight_cutoff.is_some_and(|wc| weight > wc) || abs_coeff < self.coeff_cutoff
    }
}

impl TruncationPolicy {
    /// Decompose the legacy single-policy form into the composable
    /// truncator list the propagators run internally.
    ///
    /// `weight_cutoff` maps to `WeightTruncator`, a positive `coeff_cutoff`
    /// maps to `CoefficientTruncator`, and `min_terms` maps to `TermBudget`.
    pub fn decompose(&self) -> Vec<crate::truncators::Truncator> {
        use crate::truncators::{CoefficientTruncator, TermBudget, Truncator, WeightTruncator};
        let mut ops = Vec::new();
        if let Some(weight) = self.weight_cutoff {
            ops.push(Truncator::Weight(WeightTruncator {
                weight: Some(weight),
            }));
        }
        if self.coeff_cutoff > 0.0 {
            ops.push(Truncator::Coefficient(CoefficientTruncator {
                coefficient: Some(self.coeff_cutoff),
            }));
        }
        if self.min_terms.is_some() {
            ops.push(Truncator::TermBudget(TermBudget {
                min_terms: self.min_terms,
            }));
        }
        ops
    }
}

/// Resolve the flexible `truncation` constructor argument into a truncator list.
///
/// Recognizes: a list of truncators, a single truncator, a legacy
/// `TruncationPolicy` (decomposed), and `None`. Surrogate-specific
/// forms (`FrequencyTruncationPolicy`) are handled by the surrogate crate before
/// delegating here.
pub fn resolve_truncation(truncation: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Truncator>> {
    let Some(obj) = truncation else {
        return Ok(Vec::new());
    };
    if let Ok(legacy) = obj.extract::<PyRef<TruncationPolicy>>() {
        return Ok(legacy.decompose());
    }
    if let Ok(ops) = obj.extract::<Vec<Truncator>>() {
        return Ok(ops);
    }
    if let Ok(one) = obj.extract::<Truncator>() {
        return Ok(vec![one]);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "truncation must be a truncator (FrequencyTruncator/CoefficientTruncator/\
         WeightTruncator/TermBudget/Simplify/NativeTruncator), a list of truncators, a \
         TruncationPolicy, or None",
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/policy/truncators.rs"]
mod tests;
