/// Composable truncation and flush policies shared by the numerical and surrogate
/// propagators.
///
/// Since we're doing a merging-BFS, we provide the functionality to control 
/// when to flush the outboxes, and when to truncate the live terms. The 
/// latter will always require a flush, but the former can be done independently 
/// in a lossless manner. Despite the parallel transpose, the flush operations 
/// are still the dominant source of walltime in the propagators,
/// so a finer cadence will reduce peak memory usage at the expense of time. 
///
use pyo3::prelude::*;

use crate::truncation::TruncationPolicy;

/// Default finer merge cadence (see `FlushSchedule`). Merging is O(1) per
/// term regardless of prior history under the DAG symbolic-coefficient
/// representation (and was already cheap for the numerical propagator), so
/// the default is to merge after every gate that adds a term, keeping live
/// term count minimal at all times rather than drifting toward path count
/// within a flush window.
pub const DEFAULT_MERGE_MAX_TERMS: usize = 1;

/// When to do the finer lossless merge. 
///
/// `merge_max_terms`: once this many terms accumulate in the outboxes since the
/// last flush, collapse duplicate Pauli/Majorana strings into the maps without
/// truncating. `None` disables the finer cadence (merging then happens only at
/// truncation flushes).
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

/// Drop monomials whose symbolic branch count (frequency) exceeds `frequency`.
/// The numerical propagator rejects it. A monomial with `l`
/// trig factors has expected squared magnitude `(1/2)^l` over uniform random
/// angles, so this bounds the approximation order.
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

/// Drop contributions whose coefficient magnitude is below `coefficient`. For
/// the numerical propagator this drops whole terms with `|coeff| < coefficient`;
/// for the surrogate it drops monomials with `|scalar| < coefficient`
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
    pub min_terms: Option<usize>,
    #[pyo3(get, set)]
    pub max_terms: Option<usize>,
}

#[pymethods]
impl TermBudget {
    #[new]
    #[pyo3(signature = (max_terms=None, min_terms=None))]
    pub fn new(max_terms: Option<usize>, min_terms: Option<usize>) -> Self {
        TermBudget { min_terms, max_terms }
    }
    fn __repr__(&self) -> String {
        let f = |v: Option<usize>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!("TermBudget(min_terms={}, max_terms={})", f(self.min_terms), f(self.max_terms))
    }
}

/// One entry in a truncation pipeline. Extracted from a Python list of the
/// individual truncator objects (`FromPyObject` tries each variant in turn); a
/// Python wrapper subclass instance extracts as its Rust base.
#[derive(Clone, FromPyObject)]
pub enum Truncator {
    Frequency(FrequencyTruncator),
    Coefficient(CoefficientTruncator),
    Weight(WeightTruncator),
    TermBudget(TermBudget),
}

impl Truncator {
    /// Re-materialize this operator as its Python truncator object (for the
    /// propagators' `truncators` getter).
    pub fn to_object(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::IntoPyObjectExt;
        match self {
            Truncator::Frequency(t) => t.clone().into_py_any(py),
            Truncator::Coefficient(t) => t.clone().into_py_any(py),
            Truncator::Weight(t) => t.clone().into_py_any(py),
            Truncator::TermBudget(t) => t.clone().into_py_any(py),
        }
    }

    /// Whether this operator is only meaningful for surrogate (symbolic)
    /// propagation; the numerical propagator rejects these.
    pub fn is_surrogate_only(&self) -> bool {
        matches!(self, Truncator::Frequency(_))
    }
}

pub fn reject_surrogate_only(truncators: &[Truncator]) -> PyResult<()> {
    if truncators.iter().any(Truncator::is_surrogate_only) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "FrequencyTruncator only applies to surrogate propagation; \
             use WeightTruncator / CoefficientTruncator / TermBudget with the numerical propagator",
        ));
    }
    Ok(())
}

/// The distinct truncation operations resolved from a pipeline. The list is
/// collapsed into at most one of each kind. `None` disables. The pure filters commute,
/// and the budgets are always applied by the propagator at the appropriate stage, so
/// list order is immaterial.
#[derive(Default, Clone, Copy)]
pub struct ResolvedConfig {
    pub frequency: Option<usize>,
    pub coefficient: Option<f64>,
    pub weight: Option<u32>,
    pub min_terms: Option<usize>,
    pub max_terms: Option<usize>,
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
        return Ok((schedule.unwrap_or_else(FlushSchedule::none), Vec::new()));
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
         WeightTruncator/TermBudget), a list of truncators, a \
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
    fn resolve_config_empty_is_all_none() {
        let cfg = resolve_config(&[]);
        assert_eq!(cfg.frequency, None);
        assert_eq!(cfg.weight, None);
        assert_eq!(cfg.max_terms, None);
    }

    #[test]
    fn is_surrogate_only_flags_frequency() {
        assert!(Truncator::Frequency(FrequencyTruncator { frequency: Some(3) }).is_surrogate_only());
        assert!(!Truncator::Weight(WeightTruncator { weight: Some(2) }).is_surrogate_only());
        assert!(!Truncator::TermBudget(TermBudget { min_terms: None, max_terms: Some(9) }).is_surrogate_only());
        assert!(!Truncator::Coefficient(CoefficientTruncator { coefficient: Some(1e-3) }).is_surrogate_only());
    }
}
