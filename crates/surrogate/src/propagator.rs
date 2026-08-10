///
/// impl for the surrogate/symbolic propagator!
///
/// This propagator runs on the same partitioned engine as the numerical ones:
/// the propagation itself lives in [`crate::engine`], over a
/// `PartitionedOperator<SymbolicCoeff, _, _>`. Only the truncation differs, and
/// deliberately so: see that module for why it stays post-accumulation.
///
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::logger::Logger;
use propaq_core::store::TermBasis;
use propaq_core::traits::AbstractTerm;

use crate::truncation::FrequencyTruncationPolicy;
use crate::model::{MajoranaSurrogateModel, PauliSurrogateModel};
use propaq_core::truncators::{
    reject_numerical_only, resolve_config, resolve_truncation as core_resolve_truncation, FlushSchedule,
    Truncator,
};

/// Resolve the flexible `truncation` constructor argument into `(FlushSchedule,
/// [Truncator])`. The surrogate additionally accepts the legacy
/// `FrequencyTruncationPolicy` (decomposed here); everything else, such as a list, a
/// single truncator, a core `TruncationPolicy`, or `None`, is delegated to the
/// shared `propaq_core` resolver. Every truncator the resolved list can
/// produce is honored by `apply_truncation_policy` below, with no rejection step.
fn resolve_truncation(
    truncation: Option<&Bound<'_, PyAny>>,
    schedule: Option<FlushSchedule>,
) -> PyResult<(FlushSchedule, Vec<Truncator>)> {
    if let Some(obj) = truncation {
        if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
            let (decomposed, ops) = legacy.decompose();
            return Ok((schedule.unwrap_or(decomposed), ops));
        }
    }
    core_resolve_truncation(truncation, schedule)
}

/// Configuration for a surrogate build.
///
/// The propagation itself lives in [`crate::engine`]; what remains here is the
/// worker pool and the settings a build reads, which the Python class owns
/// across calls.
pub struct SurrogatePropagator<B: TermBasis> {
    pub pool: Arc<rayon::ThreadPool>,
    /// Retained for API compatibility. The partitioned engine folds duplicates
    /// on insert, so there is no merge cadence to schedule; see `FlushSchedule`.
    pub schedule: FlushSchedule,
    /// The truncation pipeline, applied between gates in list order.
    pub truncators: Vec<Truncator>,
    pub log_filename: Option<String>,
    pub log_every: usize,
    /// Kept because the constructor accepts it. The partitioned engine runs its
    /// gate loop with the GIL released and cannot drive a tqdm bar from there,
    /// so this is currently inert; see the numerical propagators, which have the
    /// same gap.
    pub progress_bar: bool,
    _marker: PhantomData<B>,
}

impl<B: TermBasis> SurrogatePropagator<B>
where
    B::Term: AbstractTerm + for<'py> FromPyObject<'py>,
{
    /// Builds a propagator with its own rayon thread pool, flush schedule, truncator chain,
    /// and optional verbose logger.
    pub fn new(
        schedule: FlushSchedule,
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        let pool = Arc::new(
            builder
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        let (log_filename, log_every) = match &logger {
            Some(obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(SurrogatePropagator {
            pool,
            schedule,
            truncators,
            progress_bar,
            log_filename,
            log_every,
            _marker: PhantomData,
        })
    }

}

use propaq_pauli::string::PauliBasis;
use propaq_pauli::termsum::PauliTermSum;
use propaq_majorana::monomial::MajoranaBasis;
use propaq_majorana::termsum::MajoranaTermSum;


/// Width dispatch for a surrogate build, one arm per monomial storage width.
///
/// `W = ceil(2 * n_units / 64)`, and the position type is the narrowest that can
/// address that width's bits, exactly as the numerical engines choose them.

/// Replays the build's collected truncation passes into the verbose log.
///
/// Written after the build rather than during it: the build releases the GIL
/// and holds no file handle, so the events are buffered and emitted here in the
/// order they happened.
fn write_flush_log(
    log_filename: Option<&str>,
    flushes: &[crate::engine::FlushRecord],
) -> PyResult<()> {
    let Some(filename) = log_filename else {
        return Ok(());
    };
    let f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(filename)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut log = BufWriter::new(f);
    let log = &mut log;
    for f in flushes {
        let discarded = f.terms_before - f.terms_after;
        let mono_discarded = f.monomials_before.saturating_sub(f.monomials_after);
        let _ = writeln!(
            log,
            r#"{{"event":"surrogate_flush","gate_idx":{},"layer_idx":{},"qiskit_gate_idx":null,"trigger":"{}","terms_before":{},"terms_after":{},"terms_discarded":{},"monomials_before":{},"monomials_after":{},"monomials_discarded":{},"frequency":null,"weight":null,"coefficient":null,"elapsed_ms":0.0e0}}"#,
            f.gate_idx, f.layer_idx, f.trigger, f.terms_before, f.terms_after,
            discarded, f.monomials_before, f.monomials_after, mono_discarded,
        );
    }
    Ok(())
}

use propaq_majorana::algebra::to_monomial as to_monomial_majorana;
use propaq_pauli::algebra::to_monomial as to_monomial_pauli;

macro_rules! surrogate_build {
    ($algebra:ty, $to_mono:ident, $obs:expr, $layers:expr, $n_units:expr,
     $partitions:expr, $cfg:expr, $fock:expr, $n_params:expr) => {{
        let n = $n_units;
        let inline = crate::engine::INITIAL_INLINE_POSITIONS.min(2 * n.max(1));
        macro_rules! arm {
            ($w:expr, $pos:ty) => {
                crate::engine::build::<$algebra, $pos, _, $w>(
                    $obs, $layers, $to_mono::<$w>, n, $partitions, inline,
                    $cfg, $fock, $n_params,
                )
            };
        }
        if n <= 32 {
            arm!(1, u8)
        } else if n <= 64 {
            arm!(2, u8)
        } else if n <= 128 {
            arm!(4, u16)
        } else if n <= 256 {
            arm!(8, u16)
        } else if n <= 512 {
            arm!(16, u16)
        } else if n <= 1024 {
            arm!(32, u16)
        } else {
            arm!(64, u16)
        }
    }};
}

/// Back-propagates Pauli observables symbolically, producing a compiled model
/// that can be re-evaluated for any parameter assignment.
///
/// Arguments:
///     truncation: A list of truncator objects
///         (FrequencyTruncator, CoefficientTruncator, WeightTruncator,
///         TermBudget) applied at each flush, a single such truncator, a
///         legacy FrequencyTruncationPolicy (decomposed automatically), or None.
///     schedule: Optional FlushSchedule controlling flush/merge cadence. Omitted
///         -> sensible defaults when any truncator is given, or "flush only at the
///         end" when truncation is also None. A legacy policy supplies its own
///         schedule unless one is passed explicitly here.
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliBasis>,
}

#[pymethods]
impl PauliSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        reject_numerical_only(&truncators)?;
        Ok(PauliSurrogatePropagator {
            inner: SurrogatePropagator::new(schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    /// Compile the observable back-propagated through the circuit into a SurrogateModel.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A SurrogatePauliCircuit.
    ///     initial_state: Fock state as a bitstring integer (default 0).
    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PauliSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.n_units())?,
            None => Bitset::zero(),
        };
        let n_units = observable.n_units();
        let obs: Vec<(propaq_pauli::string::PauliString, f64)> =
            propaq_pauli::termsum::materialize(&observable.as_f64()).into_iter().collect();
        let layers = crate::engine::extract_layers(py, circuit)?;
        let cfg = resolve_config(&self.inner.truncators);
        let pool = std::sync::Arc::clone(&self.inner.pool);
        let partitions = pool.current_num_threads().max(1);
        let model = py.allow_threads(|| {
            pool.install(|| {
                surrogate_build!(
                    propaq_pauli::algebra::PauliAlgebra,
                    to_monomial_pauli,
                    &obs, &layers, n_units, partitions, &cfg, initial_state.as_words(), n_params
                )
            })
        })?;
        let (model, flushes) = model;
        write_flush_log(self.inner.log_filename.as_deref(), &flushes)?;
        Ok(PauliSurrogateModel { inner: model })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
    }

    /// The flush/merge schedule.
    #[getter]
    fn schedule(&self) -> FlushSchedule {
        self.inner.schedule.clone()
    }

    #[setter]
    fn set_schedule(&mut self, schedule: FlushSchedule) {
        self.inner.schedule = schedule;
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`). The current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        reject_numerical_only(&truncators)?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}

/// Back-propagates Majorana observables symbolically.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogatePropagator {
    inner: SurrogatePropagator<MajoranaBasis>,
}

#[pymethods]
impl MajoranaSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, schedule=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        schedule: Option<FlushSchedule>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        let (schedule, truncators) = resolve_truncation(truncation.as_ref(), schedule)?;
        reject_numerical_only(&truncators)?;
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(schedule, truncators, n_threads, progress_bar, logger)?,
        })
    }

    #[pyo3(signature = (observable, circuit, initial_state=None))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<MajoranaSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let initial_state = match initial_state {
            Some(v) => pyint_to_bitset(v, observable.n_units())?,
            None => Bitset::zero(),
        };
        // A unit here is a fermionic site: the monomial carries two bits per
        // unit and a Majorana monomial two modes per site.
        let n_modes = observable.n_units();
        let n_units = n_modes / 2;
        let obs: Vec<(propaq_majorana::MajoranaMonomial, f64)> =
            propaq_majorana::termsum::materialize(&observable.as_f64()).into_iter().collect();
        let layers = crate::engine::extract_layers(py, circuit)?;
        let cfg = resolve_config(&self.inner.truncators);
        let pool = std::sync::Arc::clone(&self.inner.pool);
        let partitions = pool.current_num_threads().max(1);
        let model = py.allow_threads(|| {
            pool.install(|| {
                surrogate_build!(
                    propaq_majorana::algebra::MajoranaAlgebra,
                    to_monomial_majorana,
                    &obs, &layers, n_units, partitions, &cfg, initial_state.as_words(), n_params
                )
            })
        })?;
        let (model, flushes) = model;
        write_flush_log(self.inner.log_filename.as_deref(), &flushes)?;
        Ok(MajoranaSurrogateModel { inner: model })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        self.inner.truncators.iter().map(|t| t.to_object(py)).collect()
    }

    /// The flush/merge schedule.
    #[getter]
    fn schedule(&self) -> FlushSchedule {
        self.inner.schedule.clone()
    }

    #[setter]
    fn set_schedule(&mut self, schedule: FlushSchedule) {
        self.inner.schedule = schedule;
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`). The current schedule is preserved.
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let (schedule, truncators) =
            resolve_truncation(truncation.as_ref(), Some(self.inner.schedule.clone()))?;
        reject_numerical_only(&truncators)?;
        self.inner.schedule = schedule;
        self.inner.truncators = truncators;
        Ok(())
    }
}
