///
/// The SoA propagator: applies a circuit's Pauli/Majorana rotations to a
/// `SoaTermSum` in place, using the flag/prefix-sum/scatter kernels.
///
use pyo3::prelude::*;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::coeff::CoeffRepr;
use crate::logger::Logger;
use crate::native_noise::{NativeNoiseHandle, NativeNoiseModel};
use crate::noise::UniformNoiseModel;
use crate::propagator::{close_progress_bar, make_progress_bar, tick_progress_bar, PropagationResult};
use crate::soa::kernels;
use crate::soa::{SoaBasis, SoaTermSum};
use crate::truncators::{resolve_config, FlushSchedule, ResolvedConfig, Truncator};

const EXP_LUT_SIZE: usize = 4096;

/// How a single-qubit noise channel is resolved and applied inside the propagation loop.
#[derive(Clone, Copy)]
pub enum NoiseDispatch {
    /// Built-in uniform damping, applied via a precomputed exponential lookup table.
    Uniform(f64),
    /// A dynamically loaded native noise plugin.
    Native(NativeNoiseHandle),
    /// A user-supplied Python noise model, called back into per gate.
    Python,
}

/// Tolerance for treating `cos(theta)` or `sin(theta)` as zero when classifying a rotation as
/// Clifford. See `CoeffRepr::is_clifford_param` and `CoeffRepr::phase_only_scale`.
pub const CLIFFORD_COS_EPS: f64 = 1e-9;

/// What to do with one rotation, once consecutive Clifford rotations on a shared one- or
/// two-qubit support have been collapsed. The plan keeps one entry per rotation so gate
/// indices, progress ticks, and the verbose gate log are completely unaffected by fusion; only
/// the amount of work changes.
enum GateAction {
    /// Apply this rotation normally.
    Normal,
    /// Already folded into a later rotation's fused table; do nothing at all.
    Skip,
    /// Apply this fused conjugation, which stands in for this rotation together with every
    /// immediately preceding `Skip`.
    Fused(kernels::CliffordOp),
}

/// The one- or two-qubit support of a rotation that is eligible to be fused, as
/// `(stride word, bit mask within that word)`. `None` means "not fusable": either it is not a
/// Clifford rotation, its generator spans more than one stride word, or it touches more than
/// two qubits (which would need a table bigger than the 16 entries `CliffordOp` carries).
///
/// Bases without a `local_word` (Majorana) always return `None` here and are simply never fused.
fn clifford_support<B: SoaBasis, C: CoeffRepr>(
    gen0: &[u64],
    gen1: &[u64],
    param: &C::GateParam,
) -> Option<(usize, u64)> {
    let is_clifford = C::is_clifford_param(param, CLIFFORD_COS_EPS)
        || C::phase_only_scale(param, CLIFFORD_COS_EPS).is_some();
    if !is_clifford {
        return None;
    }
    let w = B::local_word([gen0, gen1])?;
    let mask = gen0[w] | gen1[w];
    if mask == 0 || mask.count_ones() > 2 {
        return None;
    }
    Some((w, mask))
}

/// The (at most two) set bit positions of `mask`. When only one bit is set both entries are that
/// bit, which is what `CliffordOp` expects for its single-qubit form.
fn support_bits(mask: u64) -> [u32; 2] {
    let b0 = mask.trailing_zeros();
    let rest = mask & !(1u64 << b0);
    [b0, if rest == 0 { b0 } else { rest.trailing_zeros() }]
}

/// Collapses maximal runs of consecutive Clifford rotations that share a one- or two-qubit
/// support into a single conjugation table each.
///
/// `applied` must be in the order the rotations are actually applied. propaq has no gate
/// primitives (everything is `exp(-i*theta*G)` for one Pauli generator), so a single gate can
/// lower to several rotations (measured: `h` becomes 2, `cz` becomes 3, `swap` becomes 3, `cx`
/// becomes 7), each of which would otherwise cost its own full in-place traversal of the term
/// array. Fusing the run costs `4^k` table folds once and one traversal.
///
/// Runs of length 1 are deliberately left as `Normal`: there is nothing to save, and
/// `apply_rotation`'s existing single-qubit lookup table already handles that case.
fn plan_clifford_fusion<B: SoaBasis, C: CoeffRepr>(
    applied: &[&(Vec<u64>, Vec<u64>, C::GateParam, bool, Option<usize>)],
) -> Vec<GateAction> {
    let mut actions: Vec<GateAction> = applied.iter().map(|_| GateAction::Normal).collect();
    let mut i = 0usize;
    while i < applied.len() {
        let Some((word, first_mask)) = clifford_support::<B, C>(&applied[i].0, &applied[i].1, &applied[i].2)
        else {
            i += 1;
            continue;
        };
        // Extend while the run stays Clifford, stays inside the same stride word, and its
        // cumulative qubit support stays within two qubits.
        let mut mask = first_mask;
        let mut j = i + 1;
        while j < applied.len() {
            let Some((w, m)) = clifford_support::<B, C>(&applied[j].0, &applied[j].1, &applied[j].2) else {
                break;
            };
            if w != word || (mask | m).count_ones() > 2 {
                break;
            }
            mask |= m;
            j += 1;
        }
        if j - i >= 2 {
            let group: Vec<([u64; 2], C::GateParam)> = applied[i..j]
                .iter()
                .map(|(g0, g1, param, _, _)| ([g0[word], g1[word]], param.clone()))
                .collect();
            // `None` here means some rotation in the run declined to fold (e.g. a symbolic
            // parameter with no real branch factor); leaving every entry `Normal` applies the
            // run exactly as it was before fusion existed.
            if let Some(op) = kernels::build_fused_clifford::<B, C>(
                word,
                support_bits(mask),
                mask.count_ones() as usize,
                &group,
                CLIFFORD_COS_EPS,
            ) {
                for action in actions[i..j - 1].iter_mut() {
                    *action = GateAction::Skip;
                }
                actions[j - 1] = GateAction::Fused(op);
            }
        }
        i = j.max(i + 1);
    }
    actions
}

/// Drives a full circuit propagation over a `SoaTermSum`: applies every rotation, dispatches
/// noise, and flushes (merge/truncate) on the configured schedule. One instance owns its own
/// rayon thread pool and optional verbose-log file handle.
pub struct SoaPropagator<B: SoaBasis> {
    pub noise: Option<PyObject>,
    pub schedule: FlushSchedule,
    pub truncators: Vec<Truncator>,
    pub pool: Arc<rayon::ThreadPool>,
    pub progress_bar: bool,
    verbose_log: Option<std::io::BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
    current_qiskit_gate_idx: Option<usize>,
    _marker: PhantomData<B>,
}

impl<B: SoaBasis> SoaPropagator<B> {
    /// Builds a propagator with its own rayon thread pool (`n_threads` workers, or the rayon
    /// default when `None`), plus a noise model, flush schedule, truncator chain, and optional
    /// verbose logger.
    pub fn new(
        noise: Option<PyObject>,
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
        let (log_filename, log_every) = match logger {
            Some(ref obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(SoaPropagator {
            noise,
            schedule,
            truncators,
            pool,
            progress_bar,
            verbose_log: None,
            log_filename,
            log_every,
            last_log_instant: None,
            last_log_gate_idx: 0,
            current_qiskit_gate_idx: None,
            _marker: PhantomData,
        })
    }

    fn open_log(&mut self) -> PyResult<()> {
        if let Some(ref filename) = self.log_filename {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(filename)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.verbose_log = Some(std::io::BufWriter::new(f));
        }
        self.last_log_instant = None;
        self.last_log_gate_idx = 0;
        self.current_qiskit_gate_idx = None;
        Ok(())
    }

    /// Classifies `self.noise` into a `NoiseDispatch`: the built-in uniform model, a native
    /// plugin, or a plain Python callback, in that priority order.
    pub fn resolve_noise_dispatch(&self, py: Python<'_>) -> NoiseDispatch {
        if let Some(ref noise_obj) = self.noise {
            let noise = noise_obj.bind(py);
            if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
                return NoiseDispatch::Uniform(unm.damping);
            }
            if let Ok(nnm) = noise.extract::<PyRef<NativeNoiseModel>>() {
                return NoiseDispatch::Native(*nnm.handle());
            }
        }
        NoiseDispatch::Python
    }

    /// Runs `kernels::merge_and_truncate` on `evolved`, then writes a verbose-log line if a
    /// logger is configured. `trigger` names why this flush fired ("threshold", "merge",
    /// "noise", or "final") for that log line.
    ///
    /// The discarded-coefficient stats computed here are approximate: they run on the
    /// pre-dedup state, since `merge_and_truncate` does dedup, cutoff, and compaction in one
    /// pass with no clean post-merge/pre-truncate checkpoint to measure from exactly. This can
    /// over-report slightly what ends up discarded once same-key coefficients are summed.
    fn flush_and_maybe_truncate<C: CoeffRepr>(
        &mut self,
        evolved: &mut SoaTermSum<C>,
        cfg: Option<&ResolvedConfig>,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();
        let pool = Arc::clone(&self.pool);

        // Verbose logging is opt-in; only compute the (mildly expensive) discard stats when
        // a logger is actually attached.
        let active_cfg = cfg.filter(|c| c.weight.is_some() || c.coefficient.is_some() || c.native.is_some());
        let (disc_l1, disc_max) = match (active_cfg, &self.verbose_log) {
            (Some(cfg), Some(_)) => discarded_coeff_stats::<B, C>(evolved, cfg),
            _ => (0.0, 0.0),
        };

        let (total_before, total_after) = pool.install(|| kernels::merge_and_truncate::<B, C>(evolved, cfg));

        if let Some(ref mut log) = self.verbose_log {
            let actual_discarded = total_before - total_after;
            let wc_str = active_cfg
                .and_then(|c| c.weight)
                .map_or_else(|| "null".to_string(), |w| w.to_string());
            let cc = active_cfg.and_then(|c| c.coefficient).unwrap_or(0.0);
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let qki = match self.current_qiskit_gate_idx {
                Some(v) => v.to_string(),
                None => "null".to_string(),
            };
            let _ = writeln!(
                log,
                r#"{{"event":"truncation","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{actual_discarded},"discarded_coeff_l1":{disc_l1:.6e},"discarded_coeff_max":{disc_max:.6e},"weight_cutoff":{wc_str},"coeff_cutoff":{cc:.6e},"elapsed_ms":{elapsed_ms:.3e}}}"#
            );
        }
    }

    fn apply_layer_noise<C: CoeffRepr>(
        &mut self,
        py: Python<'_>,
        evolved: &mut SoaTermSum<C>,
        dispatch: NoiseDispatch,
        gate_idx: usize,
        layer_idx: usize,
    ) -> PyResult<()> {
        if self.noise.is_none() {
            return Ok(());
        }
        match dispatch {
            NoiseDispatch::Uniform(d) => {
                let exp_lut: Vec<f64> = (0..=EXP_LUT_SIZE).map(|w| (-d * w as f64).exp()).collect();
                let pool = Arc::clone(&self.pool);
                py.allow_threads(|| pool.install(|| kernels::apply_noise_inplace::<B, C>(evolved, &exp_lut)));
            }
            NoiseDispatch::Native(handle) => {
                let pool = Arc::clone(&self.pool);
                py.allow_threads(|| pool.install(|| kernels::apply_noise_native::<B, C>(evolved, &handle)));
            }
            NoiseDispatch::Python => {
                py.allow_threads(|| self.flush_and_maybe_truncate(evolved, None, gate_idx, layer_idx, "noise"));
                let noise = self.noise.as_ref().unwrap().bind(py);
                let n = evolved.len();
                let stride = evolved.stride;
                for i in 0..n {
                    let s = i * stride;
                    let w = B::weight([&evolved.planes[0][s..s + stride], &evolved.planes[1][s..s + stride]], evolved.n_units);
                    let factor: f64 = noise.call_method1("damping_factor", (w, 0u32))?.extract()?;
                    evolved.coeffs[i].scale_real(factor);
                }
            }
        }
        Ok(())
    }

    /// Core run loop, shared by `propagate` and `expectation_value`: applies
    /// every rotation in `circuit.layers`, reversed (Heisenberg
    /// back-propagation), flushing/truncating on the same threshold and
    /// `merge_max_terms` cadence as the hash-partition engine.
    fn run_propagation_inner<C: CoeffRepr>(
        &mut self,
        py: Python<'_>,
        evolved: &mut SoaTermSum<C>,
        circuit: &Bound<'_, PyAny>,
        collect_n_terms: bool,
    ) -> PyResult<Vec<usize>>
    where
        B::Term: for<'py> FromPyObject<'py>,
    {
        self.open_log()?;

        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;
        let n_units = evolved.n_units;
        let stride = evolved.stride;

        // (generator plane words, angle, is_intermediate, qiskit_gate_idx).
        // `C::extract_gate_param` reads the angle (numerical) or parameter
        // index (surrogate) off the same Python rotation object.
        let circuit_data: Vec<Vec<(Vec<u64>, Vec<u64>, C::GateParam, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<_> {
                    let rot = rot_obj.bind(py);
                    let generator: B::Term = rot.getattr("generator")?.extract()?;
                    let param = C::extract_gate_param(rot)?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    let mut gen0 = vec![0u64; stride];
                    let mut gen1 = vec![0u64; stride];
                    B::term_into_planes(&generator, n_units, [&mut gen0, &mut gen1]);
                    Ok((gen0, gen1, param, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        let cfg = resolve_config(&self.truncators);
        let max_terms: Option<usize> = cfg.max_terms;
        let merge_max_terms: Option<usize> = self.schedule.merge_max_terms;

        let mut n_terms: Vec<usize> = Vec::new();
        let dispatch = self.resolve_noise_dispatch(py);
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = make_progress_bar(py, self.progress_bar, total_rotations)?;

        // Deduplicated count as of the last merge, for the gate log's
        // `map_terms`/`outbox_terms` split (there's no physical partition
        // between the two in a flat SoA array, only "merged" vs "appended
        // since the last merge").
        let mut merged_len = evolved.len();

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let pool = Arc::clone(&self.pool);
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            self.apply_layer_noise(py, evolved, dispatch, gate_idx, layer_idx)?;

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            // One entry per rotation, so everything index-based below (gate_idx, the gate log,
            // progress ticks, the n_terms trace) sees exactly the sequence it always did.
            let fusion_plan = plan_clifford_fusion::<B, C>(&reversed_layer);
            for (idx, (gen0, gen1, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let gen = [gen0.as_slice(), gen1.as_slice()];
                let clifford_inplace = C::is_clifford_param(param, CLIFFORD_COS_EPS);
                let added = match &fusion_plan[idx] {
                    // Folded into a later rotation's table; its effect is applied there.
                    GateAction::Skip => 0,
                    // Stands in for this rotation and the preceding `Skip`s. Contributes 0 for
                    // the same injectivity reason every Clifford path does.
                    GateAction::Fused(op) => {
                        py.allow_threads(|| pool.install(|| kernels::apply_clifford_op::<B, C>(evolved, op)));
                        0
                    }
                    GateAction::Normal => py.allow_threads(|| {
                        pool.install(|| kernels::apply_rotation::<B, C>(evolved, gen, param, clifford_inplace))
                    }),
                };
                pending += added;

                self.current_qiskit_gate_idx = *qiskit_gate_idx;

                if self.verbose_log.is_some() && gate_idx % self.log_every == 0 {
                    let now = std::time::Instant::now();
                    let avg_ms_per_gate_str = match self.last_log_instant {
                        Some(last) => {
                            let gates = (gate_idx - self.last_log_gate_idx).max(1);
                            format!("{:.6e}", last.elapsed().as_secs_f64() * 1000.0 / gates as f64)
                        }
                        None => "null".to_string(),
                    };
                    self.last_log_instant = Some(now);
                    self.last_log_gate_idx = gate_idx;
                    let outbox_terms = evolved.len() - merged_len;
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{merged_len},"outbox_terms":{outbox_terms},"avg_ms_per_gate":{avg_ms_per_gate_str}}}"#
                        );
                    }
                }

                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, _, ni, _)| *ni);
                if !next_is_intermediate && max_terms.map_or(false, |max| evolved.len() >= max) {
                    py.allow_threads(|| self.flush_and_maybe_truncate(evolved, Some(&cfg), gate_idx, layer_idx, "threshold"));
                    pending = 0;
                    merged_len = evolved.len();
                } else if !next_is_intermediate && merge_max_terms.map_or(false, |m| pending >= m) {
                    py.allow_threads(|| self.flush_and_maybe_truncate(evolved, Some(&cfg), gate_idx, layer_idx, "merge"));
                    pending = 0;
                    merged_len = evolved.len();
                }

                if collect_n_terms {
                    n_terms.push(evolved.len());
                }
                tick_progress_bar(py, &pbar, &postfix, evolved.len())?;
                gate_idx += 1;
            }
        }

        close_progress_bar(py, &pbar)?;

        py.allow_threads(|| self.flush_and_maybe_truncate(evolved, Some(&cfg), gate_idx, circuit_data.len(), "final"));

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        Ok(n_terms)
    }

    /// Propagates `evolved` through `circuit` in place, discarding the intermediate term-count
    /// trace. Use `run_expectation_value` if that trace or the final expectation value is
    /// needed.
    pub fn run_propagate<C: CoeffRepr>(
        &mut self,
        py: Python<'_>,
        evolved: &mut SoaTermSum<C>,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<()>
    where
        B::Term: for<'py> FromPyObject<'py>,
    {
        self.run_propagation_inner(py, evolved, circuit, false)?;
        Ok(())
    }

    /// Propagates `evolved` through `circuit`, then computes its expectation value against
    /// `fock_state`. Returns the final expectation value together with the term-count trace
    /// collected after every rotation.
    pub fn run_expectation_value<C: CoeffRepr>(
        &mut self,
        py: Python<'_>,
        evolved: &mut SoaTermSum<C>,
        circuit: &Bound<'_, PyAny>,
        fock_state: &[u64],
    ) -> PyResult<PropagationResult>
    where
        B::Term: for<'py> FromPyObject<'py>,
    {
        let n_terms = self.run_propagation_inner(py, evolved, circuit, true)?;
        let pool = Arc::clone(&self.pool);
        let total = pool.install(|| kernels::expectation::<B, C>(evolved, fock_state));
        Ok(PropagationResult { n_terms, expectation_value: total })
    }
}

fn discarded_coeff_stats<B: SoaBasis, C: CoeffRepr>(
    terms: &SoaTermSum<C>,
    cfg: &ResolvedConfig,
) -> (f64, f64) {
    let n = terms.len();
    let stride = terms.stride;
    let cc = cfg.coefficient.unwrap_or(0.0);
    let mut l1 = 0.0f64;
    let mut max = 0.0f64;
    for i in 0..n {
        let s = i * stride;
        let term = [&terms.planes[0][s..s + stride], &terms.planes[1][s..s + stride]];
        let kept = if let Some(nt) = &cfg.native {
            let w = B::weight(term, terms.n_units);
            nt.keep(w, terms.coeffs[i].magnitude(), 0)
        } else {
            let weight_ok = cfg.weight.is_none_or(|w| B::weight(term, terms.n_units) <= w);
            weight_ok && terms.coeffs[i].passes_coeff_cutoff(cc)
        };
        if !kept {
            let mag = terms.coeffs[i].magnitude();
            l1 += mag;
            max = max.max(mag);
        }
    }
    (l1, max)
}
