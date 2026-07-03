use std::io::{BufWriter, Write};
use std::fs::OpenOptions;

use pyo3::prelude::*;

use propaq_core::coeff::CoeffRepr;
use propaq_core::propagator::AbstractPropagator;
use propaq_core::traits::AbstractTerm;

use crate::symcoeff::{GateParam, SymbolicCoeff};
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
        // Only needed for the verbose log line below; skip the O(total_terms)
        // pass entirely when logging is off.
        let monomials_before = if self.verbose_log.is_some() {
            self.inner.sum_coeffs(|c| c.monomial_count())
        } else {
            0
        };

        let outcome = apply_truncation_policy(&mut self.inner, self.truncation.as_ref());
        self.total_monomials = outcome.monomials_after;

        if self.verbose_log.is_some() {
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let qki = match self.current_qiskit_gate_idx {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            };
            let mf_str = outcome.max_frequency.map_or_else(|| "null".to_string(), |v| v.to_string());
            let wc_str = outcome.weight_cutoff.map_or_else(|| "null".to_string(), |v| v.to_string());
            let terms_discarded = outcome.total_before - outcome.total_after;
            let monomials_discarded = monomials_before - outcome.monomials_after;
            let (total_before, total_after, monomials_after) =
                (outcome.total_before, outcome.total_after, outcome.monomials_after);
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

        // Extract circuit data from Python: each rotation is either
        // symbolic (`param_index`) or numeric (`angle`) — see `GateParam`.
        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let circuit_data: Vec<Vec<(M, GateParam, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, GateParam, bool, Option<usize>)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let param = SymbolicCoeff::extract_gate_param(rot)?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    Ok((generator, param, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Uniform noise support only (symbolic coefficients can carry damping as scalar).
        let damping = self.inner.uniform_damping(py);

        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.inner.make_progress_bar(py, total_rotations)?;

        self.inner.initialize_from(evolved);

        let max_terms: Option<usize> = self.truncation.as_ref().and_then(|p| p.truncation_range.1);
        let max_monomials: Option<usize> = self.truncation.as_ref().and_then(|p| p.monomial_range.1);

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let mut pending_monomials: usize = 0;

        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            // Apply uniform noise before the layer (mirrors numerical propagator order).
            if let Some(d) = damping {
                py.allow_threads(|| self.inner.apply_uniform_noise_inplace(d));
            }

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (generator, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let (added, added_monomials) = py.allow_threads(|| self.inner.apply_gate_inplace(generator, *param));
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

        // Compile: collect terms with nonzero structural overlap. Drains
        // `self.inner`'s partition maps rather than cloning out of them —
        // `self.inner` is re-initialized from scratch on the next `build()`
        // call, so nothing is lost by moving instead of copying here.
        let raw: Vec<SurrogateTerm<M>> = self.inner.drain_collect_terms(|term, mut coeff| {
            let overlap = term.trace_with_fock_state(initial_state);
            if overlap.abs() > 1e-15 {
                coeff.deduplicate();
                Some(SurrogateTerm { term, overlap, coeff })
            } else {
                None
            }
        });

        Ok(SurrogateModel::new(raw, n_params))
    }
}

/// Result of one `apply_truncation_policy` call, for logging/reporting.
pub struct TruncationOutcome {
    pub total_before: usize,
    pub total_after: usize,
    pub monomials_after: usize,
    pub max_frequency: Option<usize>,
    pub weight_cutoff: Option<u32>,
}

/// Apply `policy`'s truncation rules to `propagator`'s current live state.
/// Assumes the caller has already flushed outboxes into partition maps
/// (`AbstractPropagator::flush_outboxes_to_maps`) if that's needed — this
/// only touches what's already live in `propagator`'s maps.
///
/// Two independent stages:
///
/// 1. Dedup, plus (once term count reaches `truncation_range.0`) an
///    optional `max_frequency` trim and `weight_cutoff` term retain.
/// 2. Independently, if a `monomial_range` is configured and the live
///    monomial count still exceeds `monomial_range.1` (`max`) after stage 1,
///    remove monomials at the single highest frequency currently present
///    — never anything lower. The *target* is `max`, not `monomial_range.0`
///    (`min`): the top-frequency bucket is removed in full only if doing so
///    doesn't remove more than needed to reach `max`; otherwise only enough
///    of it is removed (an arbitrary subset of the tied top frequency,
///    since frequency alone — not scalar magnitude — is this crate's
///    existing pruning signal; see `trim_high_frequency`) to land exactly
///    at `max`. `min` is not a target here — see `monomial_removal_budget`
///    — it's a floor that only matters on a misconfigured policy
///    (`min > max`). If the whole top-frequency bucket is itself smaller
///    than what's needed to reach `max`, it's still removed in full
///    (falling short of `max`, left to a subsequent flush — see below). Not
///    gated behind stage 1's term-count floor: monomial count can explode
///    with comparatively few live terms, so it needs its own trigger.
///
///    This deliberately only ever erodes one frequency level per call
///    rather than searching for a cutoff that reaches the floor in one
///    shot: a fast-growing run may need several consecutive flushes to
///    fully erode a deep distribution, each cheap and predictable, rather
///    than one large adaptive cut that might remove a lot of only
///    moderately-high-frequency data along with the truly extreme end.
///
/// Extracted as a standalone function (rather than inlined in
/// `SurrogatePropagator::flush_and_maybe_truncate`) so tooling that drives
/// an `AbstractPropagator` directly — without going through the
/// PyO3-circuit-driven `build()` entrypoint, e.g. `bin/cluster_bench` —
/// replicates the exact same flush behavior instead of a hand-rolled copy
/// that could drift out of sync with it.
///
/// Maximum number of monomials the monomial-range stage may remove from the
/// current top-frequency bucket this call, given `monomials_after > max`.
///
/// Targets landing at `max` (`monomials_after - max`), not `min`: a
/// top-frequency bucket bigger than that amount only ever gets a partial,
/// budgeted removal (see the caller's `budget >= n_top` branch), never
/// discarded in full, so truncation stops at `max` instead of continuing to
/// erode down toward `min`. `min` only becomes the binding constraint (via
/// `want.min(floor)` below) on a misconfigured policy with `min > max`;
/// under a sane one (`max >= min`), `monomials_after - max` is always `<=
/// monomials_after - min`, so this always reduces to `monomials_after -
/// max` and `min` has no effect. Kept as an explicit floor regardless,
/// rather than relying on callers never passing `min > max`.
#[inline]
fn monomial_removal_budget(monomials_after: usize, min: usize, max: usize) -> usize {
    let want = monomials_after.saturating_sub(max);
    let floor = monomials_after.saturating_sub(min);
    want.min(floor)
}

pub fn apply_truncation_policy<M: AbstractTerm>(
    propagator: &mut AbstractPropagator<M, SymbolicCoeff>,
    policy: Option<&FrequencyTruncationPolicy>,
) -> TruncationOutcome {
    let total_before = propagator.total_terms();

    let (max_freq, weight_cutoff, min_terms) = match policy {
        Some(tp) => (tp.max_frequency, tp.weight_cutoff, tp.truncation_range.0.unwrap_or(0)),
        None => (None, None, 0),
    };
    let (monomial_min, monomial_max) = policy.map_or((None, None), |tp| tp.monomial_range);

    // Deferred like the numerical propagator's TruncationPolicy: below
    // min_terms, skip the lossy max_frequency/weight_cutoff filtering and
    // only run the lossless dedup (merge identical monomials, drop zeros).
    let apply_lossy = total_before >= min_terms;

    let mut monomials_after = propagator.map_and_retain_coeffs_inplace(
        |_, c: &mut SymbolicCoeff| {
            if apply_lossy {
                if let Some(mf) = max_freq {
                    c.trim_high_frequency(mf);
                }
            }
            c.deduplicate();
        },
        |t: &M, c: &SymbolicCoeff| {
            let weight_ok = !apply_lossy || weight_cutoff.map_or(true, |w| t.weight() <= w);
            weight_ok && !c.is_empty()
        },
    );

    if let (Some(min), Some(max)) = (monomial_min, monomial_max) {
        if monomials_after > max {
            // Find the single highest frequency present and how many
            // monomials sit at exactly that frequency, in one parallel
            // pass — cheaper than a full histogram (plain int comparisons,
            // no hashmap) since only the top bucket is ever needed here.
            // The per-coefficient scan is itself parallel (see
            // `top_frequency_and_count`), so one giant coefficient doesn't
            // serialize its partition's whole pass.
            let (target_freq, n_top): (usize, usize) = propagator.fold_coeffs(
                || (0usize, 0usize),
                |acc, c: &SymbolicCoeff| {
                    SymbolicCoeff::combine_top_frequency(acc, c.top_frequency_and_count())
                },
                SymbolicCoeff::combine_top_frequency,
            );

            let budget = monomial_removal_budget(monomials_after, min, max);

            if n_top > 0 && budget > 0 {
                if budget >= n_top {
                    // The whole top-frequency bucket fits within budget
                    // (it's not more than what's needed to reach `max`, or
                    // `min` is the binding constraint instead): every
                    // coefficient can just drop its own hits
                    // unconditionally, no cross-coefficient coordination
                    // needed since we already know globally it's safe.
                    monomials_after = propagator.map_and_retain_coeffs_inplace(
                        |_, c: &mut SymbolicCoeff| {
                            c.remove_at_frequency(target_freq);
                        },
                        |_, c: &SymbolicCoeff| !c.is_empty(),
                    );
                } else {
                    // The whole bucket is more than needed to reach `max`
                    // (or would breach `min`): claim exactly `budget`
                    // removals across coefficients via a shared atomic
                    // counter, landing exactly at the binding target.
                    let remaining = std::sync::atomic::AtomicUsize::new(budget);
                    monomials_after = propagator.map_and_retain_coeffs_inplace(
                        |_, c: &mut SymbolicCoeff| {
                            c.remove_at_frequency_budgeted(target_freq, &remaining);
                        },
                        |_, c: &SymbolicCoeff| !c.is_empty(),
                    );
                }
            }
        }
    }

    let total_after = propagator.total_terms();
    TruncationOutcome { total_before, total_after, monomials_after, max_frequency: max_freq, weight_cutoff }
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

#[cfg(test)]
mod monomial_removal_budget_tests {
    use super::monomial_removal_budget;

    #[test]
    fn targets_max_when_bucket_would_be_more_than_enough() {
        // A bucket of any size > 10 should only ever be allowed to remove
        // 10 (landing exactly at max=90), never eroded further toward
        // min=50 just because a bigger bucket happens to be available.
        assert_eq!(monomial_removal_budget(100, 50, 90), 10);
    }

    #[test]
    fn sane_policy_ignores_min_entirely() {
        // want = after - max is always <= floor = after - min when
        // max >= min, so min should never change the result.
        for (after, min, max) in [(100, 50, 90), (100, 0, 99), (1_000_000, 1, 999_999), (11, 10, 10)] {
            assert_eq!(monomial_removal_budget(after, min, max), after - max);
        }
    }

    #[test]
    fn misconfigured_min_greater_than_max_clamps_to_the_min_floor() {
        // min > max: the floor (after - min) is tighter than the max-based
        // want (after - max), so it's the binding constraint.
        assert_eq!(monomial_removal_budget(100, 90, 50), 10);
    }

    #[test]
    fn exactly_one_over_max_wants_a_budget_of_one() {
        assert_eq!(monomial_removal_budget(91, 50, 90), 1);
    }

    #[test]
    fn zero_min_and_max_equal_to_after_minus_one() {
        assert_eq!(monomial_removal_budget(1000, 0, 999), 1);
    }
}
