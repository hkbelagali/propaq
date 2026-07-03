use pyo3::prelude::*;

const DEFAULT_MAX_TERMS: usize = 10_000_000;
const DEFAULT_MAX_MONOMIALS: usize = 10_000_000;
const DEFAULT_MIN_MONOMIALS: usize = 5_000_000;
/// Default finer merge cadence: once this many terms accumulate in the outboxes
/// since the last flush, do a lossless merge (dedup duplicate Pauli strings into
/// the maps) without truncating. Smaller than the truncation window (default
/// `DEFAULT_MAX_TERMS` = 10M) so several merges happen per truncation, keeping
/// within-window peak near the unique-term count instead of the path count.
///
/// The mechanism was validated in `cluster_bench` (28-thread Xeon, 22 qubits,
/// monomial-only truncation): enabling the finer cadence cut peak RSS ~3×
/// (≈1.54 GB → ≈0.53 GB) and wall time ~30% at identical monomial count, since
/// the collapsed duplicate Pauli strings never blow up into the path count.
/// 2M is calibrated for the design-scale 10M window; assign
/// `policy.merge_max_terms = None` to disable.
const DEFAULT_MERGE_MAX_TERMS: usize = 2_000_000;

/// Truncation policy for surrogate propagation.
///
/// Frequency truncation drops monomials whose trig factor count exceeds
/// `max_frequency`. A monomial with `l` factors has expected squared magnitude
/// `(1/2)^l` over uniform random angles, so this controls the approximation order.
///
/// `weight_cutoff` mirrors the numerical propagator's Pauli/Majorana weight cutoff.
///
/// `truncation_range` mirrors the numerical propagator's `TruncationPolicy`:
/// a `(min_terms, max_terms)` pair. A flush is triggered once the live term
/// count reaches `max_terms`, and the lossy `max_frequency`/`weight_cutoff`
/// filtering is skipped (only lossless deduplication runs) while the term
/// count is below `min_terms`. Defaults to `(None, 10_000_000)`.
///
/// `monomial_range` is a *second*, independent `(min_monomials, max_monomials)`
/// pair, on its own axis: term count is a poor proxy for a symbolic
/// coefficient's actual size — a handful of terms can carry the overwhelming
/// majority of monomials while term count barely moves, so
/// `truncation_range`'s term-count trigger alone can fail to fire until
/// memory has already exploded. A flush's monomial-level (frequency)
/// truncation isn't triggered until the live monomial count exceeds
/// `max_monomials`; once triggered, it removes monomials (highest frequency
/// first, on top of whatever `max_frequency` alone already trimmed) down to
/// `max_monomials` — the target it aims to land on, not `min_monomials`.
/// `min_monomials` is only a floor: since removal happens in whole
/// highest-frequency buckets, a bucket bigger than what's needed to reach
/// `max_monomials` gets a partial removal rather than being discarded
/// entirely, so in practice truncation lands at or just above
/// `max_monomials`, not somewhere down near `min_monomials`.
/// Defaults to `(5_000_000, 10_000_000)`.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncationPolicy {
    /// Drop monomials with more than this many trig factors (None = no limit).
    #[pyo3(get, set)]
    pub max_frequency: Option<usize>,
    /// Drop Pauli/Majorana terms with weight exceeding this value (None = no limit).
    #[pyo3(get, set)]
    pub weight_cutoff: Option<u32>,
    pub truncation_range: (Option<usize>, Option<usize>),
    pub monomial_range: (Option<usize>, Option<usize>),
    /// Finer lossless merge cadence: when this many terms accumulate in the
    /// outboxes since the last flush, collapse duplicate Pauli strings into the
    /// partition maps (no truncation). Decoupled from — and finer than — the
    /// truncation window so path-count blowup within a window is curbed early.
    /// `None` disables the finer cadence (merging then happens only at
    /// truncation flushes, the pre-decoupling behavior).
    #[pyo3(get, set)]
    pub merge_max_terms: Option<usize>,
}

#[pymethods]
impl FrequencyTruncationPolicy {
    /// `monomial_range` defaults to `(Some(5_000_000), Some(10_000_000))` when
    /// omitted. To disable monomial-range-driven truncation entirely, set
    /// `policy.monomial_range = (None, None)` after construction.
    #[new]
    #[pyo3(signature = (max_frequency=None, weight_cutoff=None, truncation_range=None, monomial_range=None, merge_max_terms=None))]
    pub fn new(
        max_frequency: Option<usize>,
        weight_cutoff: Option<u32>,
        truncation_range: Option<(Option<usize>, Option<usize>)>,
        monomial_range: Option<(Option<usize>, Option<usize>)>,
        merge_max_terms: Option<usize>,
    ) -> Self {
        FrequencyTruncationPolicy {
            max_frequency,
            weight_cutoff,
            truncation_range: truncation_range.unwrap_or((None, Some(DEFAULT_MAX_TERMS))),
            monomial_range: monomial_range
                .unwrap_or((Some(DEFAULT_MIN_MONOMIALS), Some(DEFAULT_MAX_MONOMIALS))),
            // Default-on. Assign `policy.merge_max_terms = None` after
            // construction to disable the finer cadence.
            merge_max_terms: merge_max_terms.or(Some(DEFAULT_MERGE_MAX_TERMS)),
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

    /// The (min_monomials, max_monomials) pair controlling when a flush's
    /// monomial-level truncation fires (once live count exceeds
    /// `max_monomials`) and how far it reduces the count once it does
    /// (down to `max_monomials`; `min_monomials` is only a floor against a
    /// single oversized top-frequency bucket removal overshooting further
    /// than necessary).
    #[getter]
    fn monomial_range(&self) -> (Option<usize>, Option<usize>) {
        self.monomial_range
    }

    #[setter]
    fn set_monomial_range(&mut self, value: (Option<usize>, Option<usize>)) {
        self.monomial_range = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "FrequencyTruncationPolicy(max_frequency={}, weight_cutoff={}, truncation_range=({}, {}), monomial_range=({}, {}), merge_max_terms={})",
            self.max_frequency.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.weight_cutoff.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.truncation_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.0.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.monomial_range.1.map_or_else(|| "None".to_string(), |v| v.to_string()),
            self.merge_max_terms.map_or_else(|| "None".to_string(), |v| v.to_string()),
        )
    }
}

impl FrequencyTruncationPolicy {
    /// Decompose the legacy all-in-one policy into the new `(FlushSchedule,
    /// [Truncator])` shape: scheduling knobs (term/monomial/merge triggers and
    /// the lossy `min_terms` gate) go to the schedule, and each configured cutoff
    /// becomes its own operator. Lets existing `FrequencyTruncationPolicy` calls
    /// keep working while the propagator runs the composable pipeline internally.
    pub fn decompose(&self) -> (FlushSchedule, Vec<Truncator>) {
        let schedule = FlushSchedule {
            max_terms: self.truncation_range.1,
            min_terms: self.truncation_range.0,
            max_monomials: self.monomial_range.1,
            merge_max_terms: self.merge_max_terms,
        };
        let mut ops = Vec::new();
        if let Some(max_frequency) = self.max_frequency {
            ops.push(Truncator::Frequency(FrequencyTruncator { max_frequency }));
        }
        if let Some(weight_cutoff) = self.weight_cutoff {
            ops.push(Truncator::Weight(WeightTruncator { weight_cutoff }));
        }
        if let (Some(min_monomials), Some(max_monomials)) = self.monomial_range {
            ops.push(Truncator::MonomialBudget(MonomialBudget { min_monomials, max_monomials }));
        }
        (schedule, ops)
    }
}

/// When to flush (transpose outboxes → maps and run the truncator pipeline) and
/// when to do the finer lossless merge — the *scheduling* half of truncation,
/// orthogonal to *what* gets removed (that is the `[Truncator]` list).
///
/// - `max_terms` / `max_monomials`: a flush+truncate fires once the live count
///   (plus pending) reaches either ceiling. `None` disables that trigger.
/// - `merge_max_terms`: finer lossless merge cadence (see the legacy policy docs
///   and `DEFAULT_MERGE_MAX_TERMS`). `None` disables it.
/// - `min_terms`: below this many live terms, a flush runs only the lossless
///   dedup and skips the lossy operators (frequency/weight/coefficient). The
///   monomial-budget operator is *not* gated by it — a monomial explosion with
///   few terms still needs to be cut.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FlushSchedule {
    #[pyo3(get, set)]
    pub max_terms: Option<usize>,
    #[pyo3(get, set)]
    pub max_monomials: Option<usize>,
    #[pyo3(get, set)]
    pub merge_max_terms: Option<usize>,
    #[pyo3(get, set)]
    pub min_terms: Option<usize>,
}

#[pymethods]
impl FlushSchedule {
    #[new]
    #[pyo3(signature = (max_terms=None, max_monomials=None, merge_max_terms=None, min_terms=None))]
    pub fn new(
        max_terms: Option<usize>,
        max_monomials: Option<usize>,
        merge_max_terms: Option<usize>,
        min_terms: Option<usize>,
    ) -> Self {
        FlushSchedule {
            max_terms: max_terms.or(Some(DEFAULT_MAX_TERMS)),
            max_monomials: max_monomials.or(Some(DEFAULT_MAX_MONOMIALS)),
            merge_max_terms: merge_max_terms.or(Some(DEFAULT_MERGE_MAX_TERMS)),
            min_terms,
        }
    }

    fn __repr__(&self) -> String {
        let f = |v: Option<usize>| v.map_or_else(|| "None".to_string(), |x| x.to_string());
        format!(
            "FlushSchedule(max_terms={}, max_monomials={}, merge_max_terms={}, min_terms={})",
            f(self.max_terms), f(self.max_monomials), f(self.merge_max_terms), f(self.min_terms),
        )
    }
}

impl FlushSchedule {
    /// A schedule with every trigger disabled — the "no scheduled flushing"
    /// baseline used when neither a schedule nor any truncator is supplied
    /// (propagation then flushes only once, at the end).
    pub fn none() -> Self {
        FlushSchedule { max_terms: None, max_monomials: None, merge_max_terms: None, min_terms: None }
    }
}

impl Default for FlushSchedule {
    fn default() -> Self {
        FlushSchedule::new(None, None, None, None)
    }
}

/// Drop monomials whose symbolic branch count (frequency) exceeds
/// `max_frequency`. A monomial with `l` trig factors has expected squared
/// magnitude `(1/2)^l` over uniform random angles, so this bounds the
/// approximation order. Targets the *symbolic* side of the propagation.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct FrequencyTruncator {
    #[pyo3(get, set)]
    pub max_frequency: usize,
}

#[pymethods]
impl FrequencyTruncator {
    #[new]
    pub fn new(max_frequency: usize) -> Self {
        FrequencyTruncator { max_frequency }
    }
    fn __repr__(&self) -> String {
        format!("FrequencyTruncator(max_frequency={})", self.max_frequency)
    }
}

/// Drop monomials whose (post-merge) scalar prefactor has magnitude below
/// `min_abs_scalar`. Because the symbolic trig product is bounded by 1 in
/// magnitude, `|scalar|` upper-bounds a monomial's contribution for *any*
/// parameter assignment, so this is a valid small-coefficient truncation.
/// Targets the *numerical* side — e.g. the shrinking prefactors left by
/// small-angle numeric-baked gates. Applied after dedup, so it sees merged
/// (possibly cancelled) scalars.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct CoefficientTruncator {
    #[pyo3(get, set)]
    pub min_abs_scalar: f64,
}

#[pymethods]
impl CoefficientTruncator {
    #[new]
    pub fn new(min_abs_scalar: f64) -> Self {
        CoefficientTruncator { min_abs_scalar }
    }
    fn __repr__(&self) -> String {
        format!("CoefficientTruncator(min_abs_scalar={})", self.min_abs_scalar)
    }
}

/// Drop whole Pauli/Majorana terms whose operator weight exceeds
/// `weight_cutoff` (mirrors the numerical propagator's weight cutoff).
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct WeightTruncator {
    #[pyo3(get, set)]
    pub weight_cutoff: u32,
}

#[pymethods]
impl WeightTruncator {
    #[new]
    pub fn new(weight_cutoff: u32) -> Self {
        WeightTruncator { weight_cutoff }
    }
    fn __repr__(&self) -> String {
        format!("WeightTruncator(weight_cutoff={})", self.weight_cutoff)
    }
}

/// Importance-ranked monomial budget: once the live monomial count exceeds
/// `max_monomials`, remove monomials by rank `(frequency desc, |scalar| asc)`
/// down to `max_monomials`. `min_monomials` is only a floor guarding against a
/// single oversized top bucket overshooting. This is the surrogate's memory
/// backstop; unlike the other operators it rebalances globally across terms.
#[pyclass(module = "propaq._rust_core")]
#[derive(Clone)]
pub struct MonomialBudget {
    #[pyo3(get, set)]
    pub min_monomials: usize,
    #[pyo3(get, set)]
    pub max_monomials: usize,
}

#[pymethods]
impl MonomialBudget {
    #[new]
    #[pyo3(signature = (max_monomials=DEFAULT_MAX_MONOMIALS, min_monomials=DEFAULT_MIN_MONOMIALS))]
    pub fn new(max_monomials: usize, min_monomials: usize) -> Self {
        MonomialBudget { min_monomials, max_monomials }
    }
    fn __repr__(&self) -> String {
        format!("MonomialBudget(min_monomials={}, max_monomials={})", self.min_monomials, self.max_monomials)
    }
}

/// One entry in a truncation pipeline. Extracted from a Python list of the
/// individual truncator objects (`FromPyObject` tries each variant in turn).
#[derive(Clone, FromPyObject)]
pub enum Truncator {
    Frequency(FrequencyTruncator),
    Coefficient(CoefficientTruncator),
    Weight(WeightTruncator),
    MonomialBudget(MonomialBudget),
}

impl Truncator {
    /// Re-materialize this operator as its Python truncator object (for the
    /// propagator's `truncators` getter).
    pub fn to_object(&self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::IntoPyObjectExt;
        match self {
            Truncator::Frequency(t) => t.clone().into_py_any(py),
            Truncator::Coefficient(t) => t.clone().into_py_any(py),
            Truncator::Weight(t) => t.clone().into_py_any(py),
            Truncator::MonomialBudget(t) => t.clone().into_py_any(py),
        }
    }
}
