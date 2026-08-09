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
/// Clifford.
pub const CLIFFORD_COS_EPS: f64 = 1e-9;

enum GateAction {
    /// Apply this rotation normally.
    Normal,
    /// Already folded into a later rotation's fused table
    Skip,
    /// Apply this fused conjugation
    Fused(kernels::CliffordOp),
}

/// The one- or two-qubit support of a rotation that is eligible to be fused
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

/// The (at most two) set bit positions of `mask`.
fn support_bits(mask: u64) -> [u32; 2] {
    let b0 = mask.trailing_zeros();
    let rest = mask & !(1u64 << b0);
    [b0, if rest == 0 { b0 } else { rest.trailing_zeros() }]
}

/// Collapses maximal runs of consecutive Clifford rotations that share a one- or two-qubit
/// support into a single conjugation table each.
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

/// Drives a full circuit propagation over a `SoaTermSum`
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
    /// Builds a propagator with its own rayon thread pool
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

    /// Runs `kernels::merge_and_truncate` on `evolved`
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

        // Verbose logging is opt-in
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
                let plane_span = evolved.plane_span();
                for i in 0..n {
                    let w = B::weight_sparse(evolved.row_positions(i), plane_span, evolved.n_units);
                    let factor: f64 = noise.call_method1("damping_factor", (w, 0u32))?.extract()?;
                    evolved.coeffs[i].scale_real(factor);
                }
            }
        }
        Ok(())
    }

    /// Core run loop, shared by `propagate` and `expectation_value`
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
        // So the reported peak belongs to this run, not to whatever ran before it.
        crate::soa::reset_workspace_peak();

        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;
        let n_units = evolved.n_units;
        let stride = evolved.stride;

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

        // Deduplicated count as of the last merge
        let mut merged_len = evolved.len();

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        let pool = Arc::clone(&self.pool);
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            self.apply_layer_noise(py, evolved, dispatch, gate_idx, layer_idx)?;

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            // One entry per rotation
            let fusion_plan = plan_clifford_fusion::<B, C>(&reversed_layer);
            for (idx, (gen0, gen1, param, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let gen = [gen0.as_slice(), gen1.as_slice()];
                let clifford_inplace = C::is_clifford_param(param, CLIFFORD_COS_EPS);
                let added = match &fusion_plan[idx] {
                    // Folded into a later rotation's table.
                    GateAction::Skip => 0,
                    // Stands in for this rotation and the preceding `Skip`s.
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
    /// trace.
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
    /// `fock_state`.
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
        Ok(PropagationResult {
            n_terms,
            expectation_value: total,
            sparse_key_bytes: evolved.sparse_key_bytes(),
            workspace_peak_bytes: crate::soa::workspace_peak_bytes(),
        })
    }
}

fn discarded_coeff_stats<B: SoaBasis, C: CoeffRepr>(
    terms: &SoaTermSum<C>,
    cfg: &ResolvedConfig,
) -> (f64, f64) {
    let n = terms.len();
    let plane_span = terms.plane_span();
    let cc = cfg.coefficient.unwrap_or(0.0);
    let mut l1 = 0.0f64;
    let mut max = 0.0f64;
    for i in 0..n {
        let weight_of = || B::weight_sparse(terms.row_positions(i), plane_span, terms.n_units);
        let kept = if let Some(nt) = &cfg.native {
            nt.keep(weight_of(), terms.coeffs[i].magnitude(), 0)
        } else {
            let weight_ok = cfg.weight.is_none_or(|w| weight_of() <= w);
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
