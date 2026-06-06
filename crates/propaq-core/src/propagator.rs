use pyo3::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;
use std::marker::PhantomData;
use std::sync::Arc;
use std::io::{BufWriter, Write};
use std::fs::OpenOptions;
use rustc_hash::FxHashMap;

use crate::termsum::AbstractTermSum;
use crate::noise::UniformNoiseModel;
use crate::truncation::TruncationPolicy;
use crate::traits::AbstractTerm;
use crate::logger::Logger;

/// Fibonacci hashing multiplier for multiply-shift uniform partition distribution.
const PARTITION_HASH_MUL: u64 = 0x517cc1b727220a95;

#[pyclass]
pub struct PropagationResult {
    #[pyo3(get)]
    pub n_terms: Vec<usize>,
    #[pyo3(get)]
    pub expectation_value: f64,
}

#[pymethods]
impl PropagationResult {
    fn __repr__(&self) -> String {
        format!(
            "PropagationResult(expectation_value={}, n_terms=[{} entries])",
            self.expectation_value,
            self.n_terms.len()
        )
    }
}

/// Multiply-shift partition hash: XOR-folds the term's bits, then maps
/// uniformly into [0, n_partitions) where n_partitions is a power of two.
#[inline]
fn owner_of<M: AbstractTerm>(term: &M, log2_n: u32) -> usize {
    if log2_n == 0 {
        return 0;
    }
    (term.partition_key().wrapping_mul(PARTITION_HASH_MUL) >> (64 - log2_n)) as usize
}

pub struct AbstractPropagator<M: AbstractTerm> {
    pub noise: Option<PyObject>,
    pub truncation: Option<PyObject>,
    pub pool: Arc<rayon::ThreadPool>,
    pub progress_bar: bool,
    n_partitions: usize,
    log2_n: u32,
    thread_maps: Vec<FxHashMap<M, Complex64>>,
    // outboxes[src][dst]: terms produced by partition src that belong to partition dst
    outboxes: Vec<Vec<Vec<(M, Complex64)>>>,
    total_terms: usize,
    scratch_new_terms: Vec<Vec<(usize, M, Complex64)>>,
    scratch_snap: Vec<Vec<usize>>,
    scratch_inboxes: Vec<Vec<(M, Complex64)>>,
    verbose_log: Option<BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    _marker: PhantomData<M>,
}

impl<M: AbstractTerm> AbstractPropagator<M> {
    pub fn new(
        noise: Option<PyObject>,
        truncation: Option<PyObject>,
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
        let n_threads_actual = pool.current_num_threads();
        let n_partitions = n_threads_actual.next_power_of_two().max(1);
        let log2_n = n_partitions.trailing_zeros();
        let thread_maps = (0..n_partitions).map(|_| FxHashMap::default()).collect();
        let outboxes = (0..n_partitions)
            .map(|_| (0..n_partitions).map(|_| Vec::new()).collect())
            .collect();
        let scratch_new_terms = (0..n_partitions).map(|_| Vec::new()).collect();
        let scratch_snap = (0..n_partitions).map(|_| Vec::new()).collect();
        let scratch_inboxes = (0..n_partitions).map(|_| Vec::new()).collect();
        let (log_filename, log_every) = match logger {
            Some(ref obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<pyo3::PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(AbstractPropagator {
            noise,
            truncation,
            pool,
            progress_bar,
            n_partitions,
            log2_n,
            thread_maps,
            outboxes,
            total_terms: 0,
            scratch_new_terms,
            scratch_snap,
            scratch_inboxes,
            verbose_log: None,
            log_filename,
            log_every,
            _marker: PhantomData,
        })
    }

    /// Partition `evolved.terms` into per-partition maps at the start of a run.
    fn initialize_from(&mut self, evolved: &AbstractTermSum<M>) {
        for m in &mut self.thread_maps {
            m.clear();
        }
        for (term, coeff) in &evolved.terms {
            let p = owner_of(term, self.log2_n);
            self.thread_maps[p].insert(term.clone(), *coeff);
        }
        self.total_terms = evolved.terms.len();
    }

    /// Reassemble `evolved.terms` from per-partition maps at the end of a run.
    fn finalize_to(&self, evolved: &mut AbstractTermSum<M>) {
        evolved.terms.clear();
        evolved.terms.reserve(self.total_terms);
        for map in &self.thread_maps {
            evolved.terms.extend(map.iter().map(|(k, v)| (k.clone(), *v)));
        }
    }

    fn apply_gate_inplace(&mut self, generator: &M, angle: f64) -> usize {
        let cos_t = angle.cos();
        let sin_t = angle.sin();
        let log2_n = self.log2_n;
        let n = self.n_partitions;
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        let outboxes = &mut self.outboxes;
        let scratch_new_terms = &mut self.scratch_new_terms;
        let scratch_snap = &mut self.scratch_snap;

        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(outboxes.par_iter_mut())
                .zip(scratch_new_terms.par_iter_mut())
                .zip(scratch_snap.par_iter_mut())
                .map(|(((local_map, outbox_row), new_terms), snap)| {
                    new_terms.clear();

                    // Apply gate to thread_map entries.
                    for (term, coeff) in local_map.iter_mut() {
                        if !term.commutes_with(generator) {
                            let (phase, new_term) = generator.matmul_internal(term);
                            let new_coeff = *coeff * Complex64::new(0.0, sin_t) * phase;
                            *coeff *= cos_t;
                            let dst = owner_of(&new_term, log2_n);
                            new_terms.push((dst, new_term, new_coeff));
                        }
                    }

                    // Apply gate to existing outbox items. Snapshot lengths first so
                    // items appended in this same pass are not re-processed.
                    snap.clear();
                    snap.extend(outbox_row.iter().map(|v| v.len()));
                    for dst in 0..n {
                        for i in 0..snap[dst] {
                            let (term, coeff) = &mut outbox_row[dst][i];
                            if !term.commutes_with(generator) {
                                let (phase, new_term) = generator.matmul_internal(term);
                                let new_coeff = *coeff * Complex64::new(0.0, sin_t) * phase;
                                *coeff *= cos_t;
                                let new_dst = owner_of(&new_term, log2_n);
                                new_terms.push((new_dst, new_term, new_coeff));
                            }
                        }
                    }

                    let count = new_terms.len();
                    for (dst, term, coeff) in new_terms.drain(..) {
                        outbox_row[dst].push((term, coeff));
                    }
                    count
                })
                .sum()
        })
    }

    /// Drain scratch_inboxes into thread_maps in parallel.
    fn insert_from_inboxes(&mut self) {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        let scratch_inboxes = &mut self.scratch_inboxes;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(scratch_inboxes.par_iter_mut())
                .for_each(|(map, inbox)| {
                    for (term, coeff) in inbox.drain(..) {
                        *map.entry(term).or_insert(Complex64::new(0.0, 0.0)) += coeff;
                    }
                });
        });
    }

    /// Remove terms from thread_maps that fail the truncation policy cutoffs.
    fn retain_by_policy(&mut self, tp: &TruncationPolicy) {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps.par_iter_mut().for_each(|map| {
                map.retain(|t, c| {
                    wc.map_or(true, |w| t.weight() <= w) && c.norm() >= cc
                });
            });
        });
    }

    /// Read-only pass over thread_maps to compute discard statistics before truncation.
    /// Returns (discarded_count, discarded_coeff_l1, discarded_coeff_max).
    fn collect_discard_stats(&self, tp: &TruncationPolicy) -> (usize, f64, f64) {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        let pool = Arc::clone(&self.pool);
        pool.install(|| {
            self.thread_maps
                .par_iter()
                .map(|map| {
                    let (mut dc, mut dl1, mut dmax) = (0usize, 0.0f64, 0.0f64);
                    for (t, c) in map.iter() {
                        if !wc.map_or(true, |w| t.weight() <= w) || c.norm() < cc {
                            dc += 1;
                            let norm = c.norm();
                            dl1 += norm;
                            dmax = dmax.max(norm);
                        }
                    }
                    (dc, dl1, dmax)
                })
                .reduce(
                    || (0, 0.0, 0.0f64),
                    |(a0, a1, a2), (b0, b1, b2)| (a0 + b0, a1 + b1, a2.max(b2)),
                )
        })
    }

    fn flush_and_maybe_truncate(
        &mut self,
        tp: Option<&TruncationPolicy>,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let n = self.n_partitions;

        // Phase 1: serial transpose — reuse scratch_inboxes to avoid allocation
        for inbox in &mut self.scratch_inboxes {
            inbox.clear();
        }
        for src in 0..n {
            for dst in 0..n {
                self.scratch_inboxes[dst].extend(self.outboxes[src][dst].drain(..));
            }
        }

        self.insert_from_inboxes();
        let total_before: usize = self.thread_maps.iter().map(|m| m.len()).sum();
        self.total_terms = total_before;

        if let Some(tp) = tp {
            let min_terms = tp.truncation_range.0.unwrap_or(0);
            if total_before >= min_terms {
                let (disc_count, disc_l1, disc_max) = if self.verbose_log.is_some() {
                    self.collect_discard_stats(tp)
                } else {
                    (0, 0.0, 0.0)
                };

                self.retain_by_policy(tp);
                let total_after: usize = self.thread_maps.iter().map(|m| m.len()).sum();
                self.total_terms = total_after;

                if let Some(ref mut log) = self.verbose_log {
                    let wc_str = tp.weight_cutoff
                        .map_or_else(|| "null".to_string(), |w| w.to_string());
                    let cc = tp.coeff_cutoff;
                    let _ = writeln!(
                        log,
                        r#"{{"event":"truncation","gate_idx":{gate_idx},"layer_idx":{layer_idx},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{disc_count},"discarded_coeff_l1":{disc_l1:.6e},"discarded_coeff_max":{disc_max:.6e},"weight_cutoff":{wc_str},"coeff_cutoff":{cc:.6e}}}"#
                    );
                }
            }
        }
    }

    /// Apply per-term damping to all live terms: thread_maps and outboxes alike.
    fn apply_layer_noise(
        &mut self,
        py: Python<'_>,
        pool: &rayon::ThreadPool,
        damping: Option<f64>,
        gate_idx: usize,
        layer_idx: usize,
    ) -> PyResult<()> {
        if self.noise.is_none() {
            return Ok(());
        }

        if let Some(d) = damping {
            py.allow_threads(|| {
                pool.install(|| {
                    self.thread_maps
                        .par_iter_mut()
                        .zip(self.outboxes.par_iter_mut())
                        .for_each(|(map, outbox_row)| {
                            map.iter_mut().for_each(|(term, coeff)| {
                                *coeff *= (-d * term.weight() as f64).exp();
                            });
                            for outbox in outbox_row.iter_mut() {
                                for (term, coeff) in outbox.iter_mut() {
                                    *coeff *= (-d * term.weight() as f64).exp();
                                }
                            }
                        });
                });
            });
        } else {
            // Generic Python noise: flush first, apply via Python callback, re-partition.
            py.allow_threads(|| self.flush_and_maybe_truncate(None, gate_idx, layer_idx, "noise"));
            let noise = self.noise.as_ref().unwrap().bind(py);
            let mut tmp = AbstractTermSum::new();
            self.finalize_to(&mut tmp);
            tmp.apply_damping(noise, 0)?;
            self.initialize_from(&tmp);
        }
        Ok(())
    }


    fn run_propagation_inner(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        collect_n_terms: bool,
    ) -> PyResult<Vec<usize>>
    where
        M: for<'py> FromPyObject<'py>,
    {
        self.open_log()?;

        let layers: Vec<Vec<PyObject>> = circuit.getattr("layers")?.extract()?;

        let circuit_data: Vec<Vec<(M, f64, bool)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let angle: f64 = rot.getattr("angle")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    Ok((generator, angle, is_intermediate))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Extract TruncationPolicy once (requires GIL).
        let tp: Option<TruncationPolicy> = self.truncation.as_ref().and_then(|t| {
            t.bind(py).extract::<PyRef<TruncationPolicy>>().ok().map(|p| p.clone())
        });
        let max_terms: Option<usize> = tp.as_ref().and_then(|p| p.truncation_range.1);

        let mut n_terms: Vec<usize> = Vec::new();
        let damping = self.uniform_damping(py);
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.make_progress_bar(py, total_rotations)?;
        let pool = Arc::clone(&self.pool);

        self.initialize_from(evolved);

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            self.apply_layer_noise(py, &pool, damping, gate_idx, layer_idx)?;

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (generator, angle, _is_intermediate)) in reversed_layer.iter().enumerate() {
                let added = py.allow_threads(|| self.apply_gate_inplace(generator, *angle));
                pending += added;

                if self.verbose_log.is_some() && gate_idx % self.log_every == 0 {
                    let outbox_terms: usize = self.outboxes.iter()
                        .flat_map(|r| r.iter()).map(|v| v.len()).sum();
                    let map_terms = self.total_terms;
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"map_terms":{map_terms},"outbox_terms":{outbox_terms}}}"#
                        );
                    }
                }

                // Only flush at compound-gate boundaries. In the reversed iteration,
                // a compound gate [R_final, R_inter, ..., R_inter] ends when the next
                // rotation is not intermediate (or there is no next rotation).
                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni)| *ni);
                if !next_is_intermediate && max_terms.map_or(false, |max| self.total_terms + pending >= max) {
                    py.allow_threads(|| self.flush_and_maybe_truncate(tp.as_ref(), gate_idx, layer_idx, "threshold"));
                    pending = 0;
                }

                if collect_n_terms {
                    n_terms.push(self.total_terms);
                }
                Self::tick_progress_bar(py, &pbar, &postfix, self.total_terms)?;
                gate_idx += 1;
            }
        }

        Self::close_progress_bar(py, &pbar)?;

        // Final flush + truncation.
        py.allow_threads(|| self.flush_and_maybe_truncate(tp.as_ref(), gate_idx, circuit_data.len(), "final"));

        self.finalize_to(evolved);

        // Generic Python policy: apply truncation to the finalized map.
        if tp.is_none() {
            if let Some(ref t) = self.truncation {
                evolved.truncate(t.bind(py))?;
            }
        }

        if let Some(ref mut log) = self.verbose_log {
            let _ = log.flush();
        }

        Ok(n_terms)
    }

    pub fn run_propagate(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
    ) -> PyResult<()>
    where
        M: for<'py> FromPyObject<'py>,
    {
        self.run_propagation_inner(py, evolved, circuit, false)?;
        Ok(())
    }

    pub fn run_expectation_value(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
    ) -> PyResult<PropagationResult>
    where
        M: for<'py> FromPyObject<'py>,
    {
        let n_terms = self.run_propagation_inner(py, evolved, circuit, true)?;

        let total: Complex64 = evolved
            .terms
            .iter()
            .map(|(term, coeff)| *coeff * term.trace_with_fock_state(fock_state))
            .sum();

        Ok(PropagationResult { n_terms, expectation_value: total.re })
    }

    fn open_log(&mut self) -> PyResult<()> {
        if let Some(ref filename) = self.log_filename {
            let f = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(filename)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
            self.verbose_log = Some(BufWriter::new(f));
        }
        Ok(())
    }

    fn uniform_damping(&self, py: Python<'_>) -> Option<f64> {
        if let Some(ref noise_obj) = self.noise {
            let noise = noise_obj.bind(py);
            if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
                return Some(unm.damping);
            }
        }
        None
    }

    fn make_progress_bar(
        &self,
        py: Python<'_>,
        total: usize,
    ) -> PyResult<(Option<Py<PyAny>>, Option<Py<PyAny>>)> {
        if !self.progress_bar {
            return Ok((None, None));
        }
        py.import("warnings")?.call_method1(
            "warn",
            ("propaq: the progress bar term count stays stale between truncation \
              flushes. Reduce `truncation_threshold` for more frequent updates.",),
        )?;
        let tqdm = py.import("tqdm.auto")?;
        let postfix = pyo3::types::PyDict::new(py);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("total", total)?;
        kwargs.set_item("desc", "Propagating through gates")?;
        let pbar = tqdm.call_method("tqdm", (), Some(&kwargs))?;
        Ok((Some(pbar.into()), Some(postfix.into())))
    }

    fn tick_progress_bar(
        py: Python<'_>,
        pbar: &Option<Py<PyAny>>,
        postfix: &Option<Py<PyAny>>,
        n_terms: usize,
    ) -> PyResult<()> {
        if let (Some(pbar), Some(postfix)) = (pbar, postfix) {
            let pbar = pbar.bind(py);
            let postfix = postfix.bind(py);
            postfix.set_item("terms", n_terms)?;
            pbar.call_method("set_postfix", (), Some(postfix.downcast()?))?;
            pbar.call_method0("update")?;
        }
        Ok(())
    }

    fn close_progress_bar(py: Python<'_>, pbar: &Option<Py<PyAny>>) -> PyResult<()> {
        if let Some(pbar) = pbar {
            pbar.bind(py).call_method0("close")?;
        }
        Ok(())
    }
}
