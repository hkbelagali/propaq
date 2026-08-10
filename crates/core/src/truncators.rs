/// Composable truncation and flush policies shared by the numerical and surrogate
/// propagators.
///
/// Since we're doing a merging-BFS, we provide the functionality to control 
/// when to flush the outboxes, and when to truncate the live terms. The 
/// latter will always require a flush, but the former can be done independently 
/// in a lossless manner. 
///
use pyo3::prelude::*;

use crate::native_truncator::NativeTruncator;
use crate::truncation::TruncationPolicy;

pub const DEFAULT_MERGE_MAX_TERMS: usize = 1;

/// When to do the finer lossless merge.
///
/// `merge_max_terms`: once this many terms accumulate in the outboxes since the
/// last flush, collapse duplicate Pauli/Majorana strings into the maps without
/// truncating. `None` disables the finer cadence (merging then happens only at
/// truncation flushes).
///
/// **This has no effect on the partitioned engine**, which folds a duplicate
/// into its store the moment the term is emitted and so has no outbox to flush
/// and no merge cadence to schedule. Nothing reads it any more: the surrogate
/// moved onto the same engine. The setting is accepted and ignored rather than
/// rejected, so that existing scripts keep running.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FlushSchedule {
    #[pyo3(get, set)]
    pub merge_max_terms: Option<usize>,
}

#[pymethods]
impl FlushSchedule {
    #[new]
    #[pyo3(signature = (merge_max_terms=None))]
    pub fn new(merge_max_terms: Option<usize>) -> Self {
        // Default-on. Pass/assign `merge_max_terms=None` to disable.
        FlushSchedule { merge_max_terms: merge_max_terms.or(Some(DEFAULT_MERGE_MAX_TERMS)) }
    }

    fn __repr__(&self) -> String {
        format!(
            "FlushSchedule(merge_max_terms={})",
            self.merge_max_terms.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

impl FlushSchedule {
    pub fn none() -> Self {
        FlushSchedule { merge_max_terms: None }
    }
}

impl Default for FlushSchedule {
    fn default() -> Self {
        FlushSchedule::new(None)
    }
}

#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncator {
    #[pyo3(get, set)]
    pub frequency: Option<usize>,
}

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
            self.frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct CoefficientTruncator {
    #[pyo3(get, set)]
    pub coefficient: Option<f64>,
}

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
            self.coefficient.map_or_else(|| "None".to_string(), |v| v.to_string()),
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
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct WeightTruncator {
    #[pyo3(get, set)]
    pub weight: Option<u32>,
}

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
            self.weight.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

/// Term-count budget: `max_terms` triggers a flush-and-truncate once the live
/// term count reaches it; `min_terms` is the count below which the lossy
/// operators are suppressed (only lossless dedup/merge runs). Applies to both
/// propagators. Either field `None` disables that bound.
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct TermBudget {
    #[pyo3(get, set)]
    pub max_terms: Option<usize>,
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
}

#[pymethods]
impl TermBudget {
    #[new]
    #[pyo3(signature = (max_terms=None, min_terms=None))]
    pub fn new(max_terms: Option<usize>, min_terms: Option<usize>) -> Self {
        TermBudget { max_terms, min_terms }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<usize>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!("TermBudget(max_terms={}, min_terms={})", f(self.max_terms), f(self.min_terms))
    }
}

#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct MonomialBudget {
    #[pyo3(get, set)]
    pub max_monomials: Option<u128>,
    #[pyo3(get, set)]
    pub min_monomials: Option<u128>,
}

#[pymethods]
impl MonomialBudget {
    #[new]
    #[pyo3(signature = (max_monomials=None, min_monomials=None))]
    pub fn new(max_monomials: Option<u128>, min_monomials: Option<u128>) -> Self {
        MonomialBudget { max_monomials, min_monomials }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<u128>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!(
            "MonomialBudget(max_monomials={}, min_monomials={})",
            f(self.max_monomials), f(self.min_monomials),
        )
    }
}

#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct Simplify {
    #[pyo3(get, set)]
    pub enabled: bool,
}

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
    pub fn to_object(&self, py: Python<'_>) -> PyResult<PyObject> {
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
        matches!(self, Truncator::Frequency(_) | Truncator::MonomialBudget(_) | Truncator::Simplify(_))
    }

    /// `NativeTruncator` operates on a concrete per-term coefficient
    /// magnitude (`CoeffRepr::magnitude`), which the surrogate's
    /// `SymbolicCoeff` doesn't have during build (structural pruning
    /// there uses cached per-node bounds instead).
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

/// Not `Copy` (unlike before `Native` was added): `NativeTruncator` holds
/// an `Arc`-shared plugin handle, so every consumer now gets its own
/// (cheap, refcount-bumping) `.clone()` rather than an implicit bitwise
/// copy. Every call site already binds this to a single local and reads
/// through a reference or moves it once, so this required no changes
/// beyond dropping the derive.
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

/// Collapse a truncator pipeline into a flat config (last-wins per field).
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

/// Resolve the flexible `truncation` constructor argument together with an
/// optional explicit `schedule` into `(FlushSchedule, [Truncator])`.
///
/// Recognizes: a list of truncators, a single truncator, a legacy
/// `TruncationPolicy` (decomposed; an explicit `schedule` overrides), and `None`
/// ("flush only at the end" unless a schedule is given). Surrogate-specific
/// forms (`FrequencyTruncationPolicy`) are handled by the surrogate crate before
/// delegating here.
pub fn resolve_truncation(
    truncation: Option<&Bound<'_, PyAny>>,
    schedule: Option<FlushSchedule>,
) -> PyResult<(FlushSchedule, Vec<Truncator>)> {
    let Some(obj) = truncation else {
        return Ok((schedule.unwrap_or_default(), Vec::new()));
    };
    if let Ok(legacy) = obj.extract::<PyRef<TruncationPolicy>>() {
        let (decomposed, ops) = legacy.decompose();
        return Ok((schedule.unwrap_or(decomposed), ops));
    }
    if let Ok(ops) = obj.extract::<Vec<Truncator>>() {
        return Ok((schedule.unwrap_or_default(), ops));
    }
    if let Ok(one) = obj.extract::<Truncator>() {
        return Ok((schedule.unwrap_or_default(), vec![one]));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "truncation must be a truncator (FrequencyTruncator/CoefficientTruncator/\
         WeightTruncator/TermBudget/MonomialBudget/Simplify/NativeTruncator), a list of truncators, a \
         TruncationPolicy, or None",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_config_last_wins_and_none_disables() {
        let cfg = resolve_config(&[
            Truncator::Frequency(FrequencyTruncator { frequency: Some(9) }),
            Truncator::Frequency(FrequencyTruncator { frequency: Some(5) }), // last wins
            Truncator::Coefficient(CoefficientTruncator { coefficient: Some(1e-8) }),
            Truncator::Weight(WeightTruncator { weight: Some(12) }),
            Truncator::Weight(WeightTruncator { weight: None }), // None disables
            Truncator::TermBudget(TermBudget { min_terms: Some(1), max_terms: Some(100) }),
        ]);
        assert_eq!(cfg.frequency, Some(5));
        assert_eq!(cfg.coefficient, Some(1e-8));
        assert_eq!(cfg.weight, None);
        assert_eq!((cfg.min_terms, cfg.max_terms), (Some(1), Some(100)));
    }

    #[test]
    fn resolve_config_monomial_budget_last_wins_and_none_disables() {
        let cfg = resolve_config(&[
            Truncator::MonomialBudget(MonomialBudget { min_monomials: Some(1), max_monomials: Some(1_000) }),
            Truncator::MonomialBudget(MonomialBudget { min_monomials: None, max_monomials: Some(500) }), // last wins wholesale
        ]);
        assert_eq!((cfg.min_monomials, cfg.max_monomials), (None, Some(500)));
    }

    #[test]
    fn resolve_config_empty_is_all_none() {
        let cfg = resolve_config(&[]);
        assert_eq!(cfg.frequency, None);
        assert_eq!(cfg.weight, None);
        assert_eq!(cfg.max_terms, None);
        assert_eq!(cfg.min_monomials, None);
        assert_eq!(cfg.max_monomials, None);
    }

    #[test]
    fn is_surrogate_only_flags_frequency() {
        assert!(Truncator::Frequency(FrequencyTruncator { frequency: Some(3) }).is_surrogate_only());
        assert!(!Truncator::Weight(WeightTruncator { weight: Some(2) }).is_surrogate_only());
        assert!(!Truncator::TermBudget(TermBudget { min_terms: None, max_terms: Some(9) }).is_surrogate_only());
        assert!(!Truncator::Coefficient(CoefficientTruncator { coefficient: Some(1e-3) }).is_surrogate_only());
    }

    #[test]
    fn is_surrogate_only_flags_monomial_budget() {
        assert!(
            Truncator::MonomialBudget(MonomialBudget { min_monomials: None, max_monomials: Some(9) })
                .is_surrogate_only()
        );
    }

    #[test]
    fn is_surrogate_only_flags_simplify() {
        assert!(Truncator::Simplify(Simplify { enabled: true }).is_surrogate_only());
    }

    #[test]
    fn resolve_config_simplify_last_wins_and_defaults_to_false() {
        let cfg = resolve_config(&[]);
        assert!(!cfg.simplify, "simplify must default to false when no Simplify truncator is present");

        let cfg = resolve_config(&[
            Truncator::Simplify(Simplify { enabled: true }),
            Truncator::Simplify(Simplify { enabled: false }), // last wins
        ]);
        assert!(!cfg.simplify);

        let cfg = resolve_config(&[Truncator::Simplify(Simplify { enabled: true })]);
        assert!(cfg.simplify);
    }
}
