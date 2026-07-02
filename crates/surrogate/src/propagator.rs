use std::io::{BufWriter, Write};
use std::fs::OpenOptions;

use pyo3::prelude::*;

use propaq_core::propagator::AbstractPropagator;
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::SymbolicCoeff;
use crate::truncation::FrequencyTruncationPolicy;
use crate::model::{SurrogateModel, SurrogateTerm, PauliSurrogateModel, MajoranaSurrogateModel};

/// Generic surrogate propagator wrapping `AbstractPropagator<M, SymbolicCoeff>`.
///
/// Cannot write `impl<M> AbstractPropagator<M, SymbolicCoeff>` here because
/// `AbstractPropagator` is a foreign type; instead we wrap and delegate.
pub struct SurrogatePropagator<M: AbstractTerm> {
    pub inner: AbstractPropagator<M, SymbolicCoeff>,
    pub truncation: Option<FrequencyTruncationPolicy>,
    verbose_log: Option<BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
    current_qiskit_gate_idx: Option<usize>,
    /// Total monomial count across all live coefficients. Like `total_terms`,
    /// this is only refreshed at flush points (recomputing it every gate would
    /// require a full O(total_terms) pass, unlike the O(1) term-count read).
    total_monomials: usize,
}

impl<M: AbstractTerm + for<'py> FromPyObject<'py>> SurrogatePropagator<M> {
    pub fn new(
        truncation: Option<FrequencyTruncationPolicy>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        use propaq_core::logger::Logger;
        let (log_filename, log_every) = match &logger {
            Some(obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        // Inner propagator carries no noise/truncation — surrogate manages its own.
        let inner = AbstractPropagator::new(None, None, n_threads, progress_bar, logger)?;
        Ok(SurrogatePropagator {
            inner,
            truncation,
            verbose_log: None,
            log_filename,
            log_every,
            last_log_instant: None,
            last_log_gate_idx: 0,
            current_qiskit_gate_idx: None,
            total_monomials: 0,
        })
    }

    fn open_log(&mut self) -> PyResult<()> {
        if let Some(ref filename) = self.log_filename {
            let f = OpenOptions::new()
                .create(true).write(true).truncate(true)
                .open(filename)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.verbose_log = Some(BufWriter::new(f));
        }
        self.last_log_instant = None;
        self.last_log_gate_idx = 0;
        self.current_qiskit_gate_idx = None;
        Ok(())
    }

    fn flush_and_maybe_truncate(
        &mut self,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();

        self.inner.flush_outboxes_to_maps();
        let total_before = self.inner.total_terms();
        // Only needed for the verbose log line below; skip the O(total_terms)
        // pass entirely when logging is off.
        let monomials_before = if self.verbose_log.is_some() {
            self.inner.sum_coeffs(|c| c.monomials.len())
        } else {
            0
        };

        let (max_freq, weight_cutoff, min_terms) = match &self.truncation {
            Some(tp) => (tp.max_frequency, tp.weight_cutoff, tp.truncation_range.0.unwrap_or(0)),
            None => (None, None, 0),
        };

        // Deferred like the numerical propagator's TruncationPolicy: below
        // min_terms, skip the lossy max_frequency/weight_cutoff filtering and
        // only run the lossless dedup (merge identical monomials, drop zeros).
        let apply_lossy = total_before >= min_terms;

        self.inner.map_coeffs_inplace(|_, c| {
            if apply_lossy {
                if let Some(mf) = max_freq {
                    c.trim_high_frequency(mf);
                }
            }
            c.deduplicate();
        });

        self.inner.retain_maps_with(|t, c| {
            let weight_ok = !apply_lossy || weight_cutoff.map_or(true, |w| t.weight() <= w);
            weight_ok && !c.is_empty()
        });

        let total_after = self.inner.total_terms();
        let monomials_after = self.inner.sum_coeffs(|c| c.monomials.len());
        self.total_monomials = monomials_after;

        if self.verbose_log.is_some() {
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let qki = match self.current_qiskit_gate_idx {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            };
            let mf_str = max_freq.map_or_else(|| "null".to_string(), |v| v.to_string());
            let wc_str = weight_cutoff.map_or_else(|| "null".to_string(), |v| v.to_string());
            let terms_discarded = total_before - total_after;
            let monomials_discarded = monomials_before - monomials_after;
            if let Some(ref mut log) = self.verbose_log {
                let _ = writeln!(
                    log,
                    r#"{{"event":"surrogate_flush","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{terms_discarded},"monomials_before":{monomials_before},"monomials_after":{monomials_after},"monomials_discarded":{monomials_discarded},"max_frequency":{mf_str},"weight_cutoff":{wc_str},"elapsed_ms":{elapsed_ms:.3e}}}"#
                );
            }
        }
    }

    /// Run surrogate propagation and return the compiled model.
    ///
    /// `evolved` is the observable map (contains the initial coefficients);
    /// `circuit` is a `SurrogatePauliCircuit` / `SurrogateMajoranaCircuit` Python object;
    /// `initial_state` is the Fock state for structural filtering;
    /// `n_params` is the total parameter count (determines lut size at evaluate time).
    pub fn run_build(
        &mut self,
        py: Python<'_>,
        evolved: &propaq_core::termsum::AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
        n_params: usize,
    ) -> PyResult<SurrogateModel<M>> {
        self.open_log()?;

        // Extract circuit data from Python: read `param_index` (u32) instead of angle.
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let circuit_data: Vec<Vec<(M, u32, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, u32, bool, Option<usize>)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let param_index: u32 = rot.getattr("param_index")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    Ok((generator, param_index, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Uniform noise support only (symbolic coefficients can carry damping as scalar).
        let damping = self.inner.uniform_damping(py);

        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.inner.make_progress_bar(py, total_rotations)?;

        self.inner.initialize_from(evolved);

        let max_terms: Option<usize> = self.truncation.as_ref().and_then(|p| p.truncation_range.1);
        let max_monomials: Option<usize> = self.truncation.as_ref().and_then(|p| p.max_monomials);

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let mut pending_monomials: usize = 0;

        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            // Apply uniform noise before the layer (mirrors numerical propagator order).
            if let Some(d) = damping {
                py.allow_threads(|| self.inner.apply_uniform_noise_inplace(d));
            }

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (generator, param_index, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let (added, added_monomials) = py.allow_threads(|| self.inner.apply_gate_inplace(generator, *param_index));
                pending += added;
                pending_monomials += added_monomials;

                self.current_qiskit_gate_idx = *qiskit_gate_idx;

                if self.verbose_log.is_some() && gate_idx % self.log_every == 0 {
                    let now = std::time::Instant::now();
                    let avg_ms_str = match self.last_log_instant {
                        Some(last) => {
                            let gates = (gate_idx - self.last_log_gate_idx).max(1);
                            format!("{:.6e}", last.elapsed().as_secs_f64() * 1000.0 / gates as f64)
                        }
                        None => "null".to_string(),
                    };
                    self.last_log_instant = Some(now);
                    self.last_log_gate_idx = gate_idx;
                    let outbox_terms = self.inner.n_outbox_terms();
                    let map_terms = self.inner.total_terms();
                    // Stale between flushes, like map_terms: cheap O(1) read, not
                    // recomputed every gate (that would require an O(total_terms) pass).
                    let monomials = self.total_monomials;
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{map_terms},"outbox_terms":{outbox_terms},"monomials":{monomials},"avg_ms_per_gate":{avg_ms_str}}}"#
                        );
                    }
                }

                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni, _)| *ni);
                let terms_trigger = max_terms.map_or(false, |max| self.inner.total_terms() + pending >= max);
                // Term count is a poor proxy for a symbolic coefficient's actual
                // size: a handful of terms can carry the overwhelming majority
                // of monomials while term count barely moves. Watch monomial
                // count directly so a flush still fires in that case.
                let monomials_trigger = max_monomials
                    .map_or(false, |max| self.total_monomials + pending_monomials >= max);
                if !next_is_intermediate && (terms_trigger || monomials_trigger) {
                    let trigger = if monomials_trigger && !terms_trigger {
                        "monomial_threshold"
                    } else {
                        "threshold"
                    };
                    py.allow_threads(|| self.flush_and_maybe_truncate(gate_idx, layer_idx, trigger));
                    pending = 0;
                    pending_monomials = 0;
                }

                if let Some(ref pf) = postfix {
                    // Cached at flush time, like total_terms; cheap O(1) read here.
                    pf.bind(py).set_item("monomials", self.total_monomials)?;
                }
                AbstractPropagator::<M, SymbolicCoeff>::tick_progress_bar(
                    py, &pbar, &postfix, self.inner.total_terms(),
                )?;
                gate_idx += 1;
            }
        }

        AbstractPropagator::<M, SymbolicCoeff>::close_progress_bar(py, &pbar)?;

        py.allow_threads(|| self.flush_and_maybe_truncate(gate_idx, circuit_data.len(), "final"));

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        // Compile: collect terms with nonzero structural overlap.
        let raw: Vec<SurrogateTerm<M>> = self.inner.collect_terms(|term, coeff| {
            let overlap = term.trace_with_fock_state(initial_state);
            if overlap.abs() > 1e-15 {
                let mut c = coeff.clone();
                c.deduplicate();
                Some(SurrogateTerm { term: term.clone(), overlap, coeff: c })
            } else {
                None
            }
        });

        Ok(SurrogateModel::new(raw, n_params))
    }
}

use propaq_pauli::string::PauliString;
use propaq_pauli::termsum::PauliTermSum;
use propaq_majorana::monomial::MajoranaMonomial;
use propaq_majorana::termsum::MajoranaTermSum;

/// Back-propagates Pauli observables symbolically, producing a compiled model
/// that can be re-evaluated for any parameter assignment.
///
/// Arguments:
///     truncation: Optional FrequencyTruncationPolicy (frequency + weight cutoffs).
///     n_threads: Number of worker threads. Defaults to the system thread count.
///     progress_bar: Display a tqdm progress bar during propagation.
///     logger: Optional Logger for verbose JSON Lines event logging.
#[pyclass(module = "propaq._rust_core")]
pub struct PauliSurrogatePropagator {
    inner: SurrogatePropagator<PauliString>,
}

#[pymethods]
impl PauliSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<FrequencyTruncationPolicy>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        Ok(PauliSurrogatePropagator {
            inner: SurrogatePropagator::new(truncation, n_threads, progress_bar, logger)?,
        })
    }

    /// Compile the observable back-propagated through the circuit into a SurrogateModel.
    ///
    /// Arguments:
    ///     observable: The Pauli observable to back-propagate.
    ///     circuit: A SurrogatePauliCircuit.
    ///     initial_state: Fock state as a bitstring integer (default 0).
    #[pyo3(signature = (observable, circuit, initial_state=0))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &PauliTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
    ) -> PyResult<PauliSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let evolved = observable.inner.copy();
        let model = self.inner.run_build(py, &evolved, circuit, initial_state, n_params)?;
        Ok(PauliSurrogateModel { inner: model })
    }

    #[getter]
    fn truncation(&self) -> Option<FrequencyTruncationPolicy> {
        self.inner.truncation.clone()
    }

    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<FrequencyTruncationPolicy>) {
        self.inner.truncation = truncation;
    }
}

/// Back-propagates Majorana observables symbolically.
#[pyclass(module = "propaq._rust_core")]
pub struct MajoranaSurrogatePropagator {
    inner: SurrogatePropagator<MajoranaMonomial>,
}

#[pymethods]
impl MajoranaSurrogatePropagator {
    #[new]
    #[pyo3(signature = (truncation=None, n_threads=None, progress_bar=false, logger=None))]
    fn new(
        truncation: Option<FrequencyTruncationPolicy>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
    ) -> PyResult<Self> {
        Ok(MajoranaSurrogatePropagator {
            inner: SurrogatePropagator::new(truncation, n_threads, progress_bar, logger)?,
        })
    }

    #[pyo3(signature = (observable, circuit, initial_state=0))]
    fn build(
        &mut self,
        py: Python<'_>,
        observable: &MajoranaTermSum,
        circuit: &Bound<'_, PyAny>,
        initial_state: u64,
    ) -> PyResult<MajoranaSurrogateModel> {
        let n_params: usize = circuit.getattr("n_params")?.extract()?;
        let evolved = observable.inner.copy();
        let model = self.inner.run_build(py, &evolved, circuit, initial_state, n_params)?;
        Ok(MajoranaSurrogateModel { inner: model })
    }

    #[getter]
    fn truncation(&self) -> Option<FrequencyTruncationPolicy> {
        self.inner.truncation.clone()
    }

    #[pyo3(signature = (truncation=None))]
    fn set_truncation(&mut self, truncation: Option<FrequencyTruncationPolicy>) {
        self.inner.truncation = truncation;
    }
}
