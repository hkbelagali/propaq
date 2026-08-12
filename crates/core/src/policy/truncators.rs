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

/// Term-count budget: `min_terms` is the count below which lossy operators are
/// suppressed; `max_terms` triggers a truncation pass once the live term count
/// reaches it. Applies to both propagators. Either field `None` disables that
/// bound.
///
/// The two are keyword-only.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TermBudget {
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
    #[pyo3(get, set)]
    pub max_terms: Option<usize>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl TermBudget {
    #[new]
    #[pyo3(signature = (*, min_terms=None, max_terms=None))]
    pub fn new(min_terms: Option<usize>, max_terms: Option<usize>) -> Self {
        TermBudget {
            min_terms,
            max_terms,
        }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<usize>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!(
            "TermBudget(min_terms={}, max_terms={})",
            f(self.min_terms),
            f(self.max_terms)
        )
    }
}

#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct MonomialBudget {
    #[pyo3(get, set)]
    pub min_monomials: Option<u128>,
    #[pyo3(get, set)]
    pub max_monomials: Option<u128>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl MonomialBudget {
    #[new]
    #[pyo3(signature = (*, min_monomials=None, max_monomials=None))]
    pub fn new(min_monomials: Option<u128>, max_monomials: Option<u128>) -> Self {
        MonomialBudget {
            min_monomials,
            max_monomials,
        }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<u128>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!(
            "MonomialBudget(min_monomials={}, max_monomials={})",
            f(self.min_monomials),
            f(self.max_monomials),
        )
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
    MonomialBudget(MonomialBudget),
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
            Truncator::MonomialBudget(t) => t.clone().into_py_any(py),
            Truncator::Simplify(t) => t.clone().into_py_any(py),
            Truncator::Native(t) => t.clone().into_py_any(py),
        }
    }

    pub fn is_surrogate_only(&self) -> bool {
        matches!(
            self,
            Truncator::Frequency(_) | Truncator::MonomialBudget(_) | Truncator::Simplify(_)
        )
    }

    pub fn is_numerical_only(&self) -> bool {
        matches!(self, Truncator::Native(_))
    }
}

pub fn reject_surrogate_only(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(Truncator::is_surrogate_only) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "FrequencyTruncator/MonomialBudget/Simplify only apply to surrogate propagation; \
             use WeightTruncator / CoefficientTruncator / TermBudget with the numerical propagator",
        ));
    }
    Ok(())
}

pub fn reject_numerical_only(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(Truncator::is_numerical_only) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "NativeTruncator only applies to numerical propagation (it decides per-term based on a \
             concrete coefficient magnitude, which the surrogate's symbolic coefficients don't have \
             during build); use it with PauliPropagator/MajoranaPropagator instead",
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
    pub max_terms: Option<usize>,
    pub min_monomials: Option<u128>,
    pub max_monomials: Option<u128>,
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
            Truncator::TermBudget(x) => {
                r.min_terms = x.min_terms;
                r.max_terms = x.max_terms;
            }
            Truncator::MonomialBudget(x) => {
                r.min_monomials = x.min_monomials;
                r.max_monomials = x.max_monomials;
            }
            Truncator::Simplify(x) => r.simplify = x.enabled,
            Truncator::Native(x) => r.native = Some(x.clone()),
        }
    }
    r
}

const DEFAULT_MAX_TERMS: usize = 10_000_000;

/// Controls when and how terms are discarded during propagation.
///
/// Arguments:
///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
///         None disables weight-based truncation.
///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
///     truncation_range: (min_terms, max_terms) pair. Truncation fires when the term
///         count reaches max_terms and will not reduce it below min_terms.
///         Defaults to (None, \(10^7\)).
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TruncationPolicy {
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    #[pyo3(get, set)]
    pub coeff_cutoff: f64,
    pub truncation_range: (Option<usize>, Option<usize>),
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl TruncationPolicy {
    /// Initialize the truncation policy.
    ///
    /// Arguments:
    ///     weight_cutoff: Discard terms with Pauli weight strictly greater than this value.
    ///     coeff_cutoff: Discard terms with |coefficient| strictly less than this value.
    ///     truncation_range: (min_terms, max_terms) pair. Truncation is triggered when the
    ///         term count reaches max_terms and will not reduce it below min_terms.
    #[new]
    #[pyo3(signature = (weight_cutoff=None, coeff_cutoff=0.0, truncation_range=None))]
    fn new(
        weight_cutoff: Option<u32>,
        coeff_cutoff: f64,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
    ) -> Self {
        TruncationPolicy {
            weight_cutoff,
            coeff_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
        }
    }

    /// The (min_terms, max_terms) pair controlling when and how aggressively truncation fires.
    #[getter]
    fn truncation_range(&self) -> (Option<usize>, Option<usize>) {
        self.truncation_range
    }

    #[setter]
    fn set_truncation_range(&mut self, value: (Option<usize>, Option<usize>)) {
        self.truncation_range = value;
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
    /// maps to `CoefficientTruncator`, and `truncation_range` maps to
    /// `TermBudget`.
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
        if self.truncation_range.0.is_some() || self.truncation_range.1.is_some() {
            ops.push(Truncator::TermBudget(TermBudget {
                min_terms: self.truncation_range.0,
                max_terms: self.truncation_range.1,
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
         WeightTruncator/TermBudget/MonomialBudget/Simplify/NativeTruncator), a list of truncators, a \
         TruncationPolicy, or None",
    ))
}

#[cfg(test)]
#[path = "../../tests/unit/policy/truncators.rs"]
mod tests;
