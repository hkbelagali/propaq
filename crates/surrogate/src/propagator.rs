//!
//! impl for the surrogate/symbolic propagator! This uses the same
//! partitioned engine as the numerical propagators, with a symbolic
//! coefficient type.
//!
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::sync::Arc;

use pyo3::prelude::*;

use propaq_core::bitset::Bitset;
use propaq_core::helpers::pyint_to_bitset;
use propaq_core::logger::Logger;
use propaq_core::progress::Progress;
use propaq_core::store::TermBasis;
use propaq_core::traits::AbstractTerm;

use crate::model::{MajoranaSurrogateModel, PauliSurrogateModel};
use crate::truncation::FrequencyTruncationPolicy;
use propaq_core::truncators::{
    reject_numerical_only, resolve_config, resolve_truncation as core_resolve_truncation, Truncator,
};

fn resolve_truncation(truncation: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Truncator>> {
    if let Some(obj) = truncation {
        if let Ok(legacy) = obj.extract::<PyRef<FrequencyTruncationPolicy>>() {
            return Ok(legacy.decompose());
        }
    }
    core_resolve_truncation(truncation)
}

/// Configuration for a surrogate build.
pub struct SurrogatePropagator<B: TermBasis> {
    pub pool: Arc<rayon::ThreadPool>,

    pub truncators: Vec<Truncator>,
    pub log_filename: Option<String>,
    pub log_every: usize,
    /// Draw a tqdm bar over the build's gate loop.
    pub progress_bar: bool,
    /// Gates between bar ticks.
    pub progress_every: usize,
    _marker: PhantomData<B>,
}

impl<B: TermBasis> SurrogatePropagator<B>
where
    B::Term: AbstractTerm + for<'a, 'py> FromPyObject<'a, 'py>,
{
    pub fn new(
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
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
            Some(obj) => Python::attach(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(SurrogatePropagator {
            pool,
            truncators,
            log_filename,
            log_every,
            progress_bar,
            progress_every: progress_every.max(1),
            _marker: PhantomData,
        })
    }
}

use propaq_majorana::monomial::MajoranaBasis;
use propaq_majorana::termsum::MajoranaTermSum;
use propaq_pauli::string::PauliBasis;
use propaq_pauli::termsum::PauliTermSum;

/// Replays the build's collected truncation passes into the verbose log.
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
            f.gate_idx,
            f.layer_idx,
            f.trigger,
            f.terms_before,
            f.terms_after,
            discarded,
            f.monomials_before,
            f.monomials_after,
            mono_discarded,
        );
    }
    Ok(())
}

use propaq_majorana::algebra::to_basis_string as to_basis_string_majorana;
use propaq_pauli::algebra::to_basis_string as to_basis_string_pauli;

/// Width dispatch for a surrogate build, one arm per basis-string storage width.
macro_rules! surrogate_build {
    ($algebra:ty, $to_basis:ident, $obs:expr, $layers:expr, $n_units:expr,
     $partitions:expr, $cfg:expr, $fock:expr, $n_params:expr, $progress:expr) => {{
        let n = $n_units;
        let inline = crate::engine::INITIAL_INLINE_POSITIONS.min(2 * n.max(1));
        macro_rules! arm {
            ($w:expr, $pos:ty) => {
                crate::engine::build::<$algebra, $pos, _, $w>(
                    $obs,
                    $layers,
                    $to_basis::<$w>,
                    n,
                    $partitions,
                    inline,
                    $cfg,
                    $fock,
                    $n_params,
                    $progress,
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
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     logger: Optional Logger for verbose JSON Lines event logging.
///     progress_bar: Draw a tqdm bar over the build's gate loop.
///     progress_every: Gates between progress bar ticks. Defaults to 1.
///
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliBasis>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl PauliSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, logger=None, progress_bar=false, progress_every=1))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        Ok(PauliSurrogatePropagator {
            inner: SurrogatePropagator::new(
                truncators,
                n_threads,
                logger,
                progress_bar,
                progress_every,
            )?,
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
            propaq_pauli::termsum::materialize(&observable.as_f64())
                .into_iter()
                .collect();
        let layers = crate::engine::extract_layers(py, circuit)?;
        let cfg = resolve_config(&self.inner.truncators);
        let pool = std::sync::Arc::clone(&self.inner.pool);
        let partitions = pool.current_num_threads().max(1);
        let total_gates = layers.iter().map(Vec::len).sum();
        let progress = Progress::new(
            py,
            self.inner.progress_bar,
            total_gates,
            self.inner.progress_every,
        )?;
        let model = py.detach(|| {
            pool.install(|| {
                surrogate_build!(
                    propaq_pauli::algebra::PauliAlgebra,
                    to_basis_string_pauli,
                    &obs,
                    &layers,
                    n_units,
                    partitions,
                    &cfg,
                    initial_state.as_words(),
                    n_params,
                    progress.as_ref()
                )
            })
        });

        if let Some(p) = progress.as_ref() {
            p.close();
        }
        let (model, flushes) = model?;
        write_flush_log(self.inner.log_filename.as_deref(), &flushes)?;
        Ok(PauliSurrogateModel { inner: model })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .truncators
            .iter()
            .map(|t| t.to_object(py))
            .collect()
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`).
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        self.inner.truncators = truncators;
        Ok(())
    }
}

/// Back-propagates Majorana observables symbolically.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogatePropagator {
    inner: SurrogatePropagator<MajoranaBasis>,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl MajoranaSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, logger=None, progress_bar=false, progress_every=1))]
    fn new(
        truncation: Option<Bound<'_, PyAny>>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
        progress_bar: bool,
        progress_every: usize,
    ) -> PyResult<Self> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(
                truncators,
                n_threads,
                logger,
                progress_bar,
                progress_every,
            )?,
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

        let n_modes = observable.n_units();
        let n_units = n_modes / 2;
        let obs: Vec<(propaq_majorana::MajoranaMonomial, f64)> =
            propaq_majorana::termsum::materialize(&observable.as_f64())
                .into_iter()
                .collect();
        let layers = crate::engine::extract_layers(py, circuit)?;
        let cfg = resolve_config(&self.inner.truncators);
        let pool = std::sync::Arc::clone(&self.inner.pool);
        let partitions = pool.current_num_threads().max(1);
        let total_gates = layers.iter().map(Vec::len).sum();
        let progress = Progress::new(
            py,
            self.inner.progress_bar,
            total_gates,
            self.inner.progress_every,
        )?;
        let model = py.detach(|| {
            pool.install(|| {
                surrogate_build!(
                    propaq_majorana::algebra::MajoranaAlgebra,
                    to_basis_string_majorana,
                    &obs,
                    &layers,
                    n_units,
                    partitions,
                    &cfg,
                    initial_state.as_words(),
                    n_params,
                    progress.as_ref()
                )
            })
        });

        if let Some(p) = progress.as_ref() {
            p.close();
        }
        let (model, flushes) = model?;
        write_flush_log(self.inner.log_filename.as_deref(), &flushes)?;
        Ok(MajoranaSurrogateModel { inner: model })
    }

    /// The active truncation pipeline as a list of truncator objects.
    #[getter]
    fn truncators(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .truncators
            .iter()
            .map(|t| t.to_object(py))
            .collect()
    }

    /// Replace the truncation pipeline (accepts the same forms as the
    /// constructor's `truncation`).
    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let truncators = resolve_truncation(truncation.as_ref())?;
        reject_numerical_only(&truncators)?;
        self.inner.truncators = truncators;
        Ok(())
    }
}
