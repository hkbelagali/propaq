use pyo3::prelude::*;
use rayon::prelude::*;
use num_complex::Complex64;
use std::marker::PhantomData;
use std::sync::Arc;
use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::OpenOptions;
use rustc_hash::FxHashMap;
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;

use crate::termsum::AbstractTermSum;
use crate::noise::UniformNoiseModel;
use crate::truncation::TruncationPolicy;
use crate::traits::AbstractTerm;
use crate::logger::Logger;

/// Fibonacci hashing multiplier for multiply-shift uniform partition distribution.
const PARTITION_HASH_MUL: u64 = 0x517cc1b727220a95;

// Limit for uniform noise LUT 
const EXP_LUT_SIZE: usize = 4096;

struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    unsafe fn offset(&self, idx: usize) -> *mut T { self.0.add(idx) }
}

/// Result returned by `expectation_value`: per-gate term counts and the final expectation value.
#[pyclass(module = "propaq._rust_core")]
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

/// Serialize `terms` to a gzip-compressed binary file at `path`.
///
/// Format (all integers little-endian):
///   u64  n_terms
///   u64  key_stride    (bytes per key; 0 when n_terms == 0)
///   u64  system_size   (n_qubits for Pauli, n_modes for Majorana)
///   For each term:
///     [u8; key_stride]  key bytes from AbstractTerm::to_bytes_vec()
///     f64               coefficient real part
///     f64               coefficient imaginary part
pub fn save_terms_to_file<M: AbstractTerm>(
    terms: &FxHashMap<M, Complex64>,
    path: &str,
) -> PyResult<()> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut enc = GzEncoder::new(BufWriter::new(file), Compression::default());

    let n_terms = terms.len() as u64;
    enc.write_all(&n_terms.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let first = terms.keys().next();
    let key_stride: u64 = first.map_or(0, |t| t.to_bytes_vec().len() as u64);
    let system_size: u64 = first.map_or(0, |t| t.system_size());
    enc.write_all(&key_stride.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    enc.write_all(&system_size.to_le_bytes())
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    for (term, coeff) in terms.iter() {
        enc.write_all(&term.to_bytes_vec())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        enc.write_all(&coeff.re.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        enc.write_all(&coeff.im.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    enc.finish()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

/// Deserialize a term map from a file produced by `save_terms_to_file`.
pub fn load_terms_from_file<M: AbstractTerm>(path: &str) -> PyResult<FxHashMap<M, Complex64>> {
    let file = std::fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let mut dec = BufReader::new(GzDecoder::new(file));

    let mut u64_buf = [0u8; 8];
    let mut f64_buf = [0u8; 8];

    let io_err = |e: std::io::Error| pyo3::exceptions::PyIOError::new_err(e.to_string());

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let n_terms = u64::from_le_bytes(u64_buf) as usize;

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let key_stride = u64::from_le_bytes(u64_buf) as usize;

    dec.read_exact(&mut u64_buf).map_err(io_err)?;
    let system_size = u64::from_le_bytes(u64_buf);

    let mut terms = FxHashMap::default();
    terms.reserve(n_terms);
    let mut key_buf = vec![0u8; key_stride];

    for _ in 0..n_terms {
        dec.read_exact(&mut key_buf).map_err(io_err)?;
        dec.read_exact(&mut f64_buf).map_err(io_err)?;
        let re = f64::from_le_bytes(f64_buf);
        dec.read_exact(&mut f64_buf).map_err(io_err)?;
        let im = f64::from_le_bytes(f64_buf);
        let term = M::from_bytes_vec(&key_buf, system_size);
        terms.insert(term, Complex64::new(re, im));
    }
    Ok(terms)
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
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
<<<<<<< HEAD
=======
    current_qiskit_gate_idx: Option<usize>,
>>>>>>> origin/main
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
            last_log_instant: None,
            last_log_gate_idx: 0,
<<<<<<< HEAD
=======
            current_qiskit_gate_idx: None,
>>>>>>> origin/main
            _marker: PhantomData,
        })
    }

    /// Partition `evolved.terms` into per-partition maps at the start of a run.

    fn initialize_from(&mut self, evolved: &AbstractTermSum<M>) {
        let n = self.n_partitions;
        let log2_n = self.log2_n;

        let mut buckets: Vec<Vec<(&M, Complex64)>> = (0..n).map(|_| Vec::new()).collect();
        for (term, coeff) in &evolved.terms {
            buckets[owner_of(term, log2_n)].push((term, *coeff));
        }

        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(buckets.par_iter())
                .for_each(|(map, bucket)| {
                    map.clear();
                    map.reserve(bucket.len());
                    for (term, coeff) in bucket {
                        map.insert((*term).clone(), *coeff);
                    }
                });
        });

        self.total_terms = evolved.terms.len();
    }

    /// Reassemble `evolved.terms` from per-partition maps at the end of a run.
    fn finalize_to(&self, evolved: &mut AbstractTermSum<M>) {
        evolved.terms.clear();
        evolved.terms.reserve(self.total_terms);
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        let all_items: Vec<(M, Complex64)> = pool.install(|| {
            thread_maps
                .par_iter()
                .flat_map_iter(|map| map.iter().map(|(k, v)| (k.clone(), *v)))
                .collect()
        });
        evolved.terms.extend(all_items);
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

    /// Apply the truncation policy only to threads where `mask[i]` is true.
    fn retain_by_policy_masked(&mut self, tp: &TruncationPolicy, mask: &[bool]) {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(mask.par_iter())
                .for_each(|(map, &apply)| {
                    if apply {
                        map.retain(|t, c| wc.map_or(true, |w| t.weight() <= w) && c.norm() >= cc);
                    }
                });
        });
    }

    /// Single parallel pass returning per-thread surviving counts and aggregate discard
    /// statistics (discarded_coeff_l1, discarded_coeff_max). Replaces the two separate
    /// `count_surviving_per_thread` and `collect_discard_stats` passes.
    fn collect_stats_and_count_surviving(&self, tp: &TruncationPolicy) -> ((f64, f64), Vec<usize>) {
        let wc = tp.weight_cutoff;
        let cc = tp.coeff_cutoff;
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        let per_thread: Vec<((f64, f64), usize)> = pool.install(|| {
            thread_maps
                .par_iter()
                .map(|map| {
                    let (mut dl1, mut dmax, mut surv) = (0.0f64, 0.0f64, 0usize);
                    for (t, c) in map.iter() {
                        if wc.map_or(true, |w| t.weight() <= w) && c.norm() >= cc {
                            surv += 1;
                        } else {
                            let norm = c.norm();
                            dl1 += norm;
                            dmax = dmax.max(norm);
                        }
                    }
                    ((dl1, dmax), surv)
                })
                .collect()
        });
        let (dl1, dmax) = per_thread.iter().fold((0.0f64, 0.0f64), |acc, ((dl1, dmax), _)| {
            (acc.0 + dl1, acc.1.max(*dmax))
        });
        let surviving: Vec<usize> = per_thread.into_iter().map(|(_, s)| s).collect();
        ((dl1, dmax), surviving)
    }

    fn flush_and_maybe_truncate(
        &mut self,
        tp: Option<&TruncationPolicy>,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();
        let n = self.n_partitions;

        // Phase 1: parallel transpose — reuse scratch_inboxes to avoid allocation
        for inbox in &mut self.scratch_inboxes {
            inbox.clear();
        }
        let pool = Arc::clone(&self.pool);
        let outboxes = &mut self.outboxes;
        let scratch_inboxes = &mut self.scratch_inboxes;
        let outboxes_ptr = SendPtr(outboxes.as_mut_ptr());
        pool.install(|| {
            scratch_inboxes
                .par_iter_mut()
                .enumerate()
                .for_each(|(dst, inbox)| {
                    for src in 0..n {
                        // SAFETY: Each parallel task owns a unique `dst`.
                        // Thread `dst` drains outboxes[src][dst] for all src.
                        // No two threads share a (src, dst) pair; the cells are
                        // distinct heap allocations. `outboxes` is not resized
                        // during this block (n is fixed, no push occurs).
                        let cell = unsafe { &mut (&mut *outboxes_ptr.offset(src))[dst] };
                        inbox.extend(cell.drain(..));
                    }
                });
        });

        self.insert_from_inboxes();
        let total_before: usize = self.thread_maps.iter().map(|m| m.len()).sum();
        self.total_terms = total_before;

        if let Some(tp) = tp {
            let min_terms = tp.truncation_range.0.unwrap_or(0);
            if total_before >= min_terms {
                let need_surviving = min_terms > 0;
                let need_stats = self.verbose_log.is_some();

                // Single parallel pass instead of the former two separate scans.
                let (disc_l1, disc_max, surviving) = if need_surviving || need_stats {
                    let ((dl1, dmax), surv) = self.collect_stats_and_count_surviving(tp);
                    let (dl1, dmax) = if need_stats { (dl1, dmax) } else { (0.0, 0.0) };
                    (dl1, dmax, surv)
                } else {
                    (0.0, 0.0, Vec::new())
                };

                if need_surviving {
                    // Check whether applying the policy globally would drop the total
                    // below min_terms. If so, redistribute: only truncate threads where
                    // surviving[i] meets its proportional share of min_terms so that the
                    // global total stays >= min_terms after truncation.
                    let total_surviving: usize = surviving.iter().sum();
                    if total_surviving < min_terms {
                        let mask: Vec<bool> = surviving.iter()
                            .zip(self.thread_maps.iter())
                            .map(|(&surv, map)| surv >= map.len() * min_terms / total_before)
                            .collect();
                        self.retain_by_policy_masked(tp, &mask);
                    } else {
                        self.retain_by_policy(tp);
                    }
                } else {
                    self.retain_by_policy(tp);
                }

                let total_after: usize = self.thread_maps.iter().map(|m| m.len()).sum();
                self.total_terms = total_after;

                if let Some(ref mut log) = self.verbose_log {
                    let actual_discarded = total_before - total_after;
                    let wc_str = tp.weight_cutoff
                        .map_or_else(|| "null".to_string(), |w| w.to_string());
                    let cc = tp.coeff_cutoff;
                    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
<<<<<<< HEAD
                    let _ = writeln!(
                        log,
                        r#"{{"event":"truncation","gate_idx":{gate_idx},"layer_idx":{layer_idx},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{actual_discarded},"discarded_coeff_l1":{disc_l1:.6e},"discarded_coeff_max":{disc_max:.6e},"weight_cutoff":{wc_str},"coeff_cutoff":{cc:.6e},"elapsed_ms":{elapsed_ms:.3e}}}"#
=======
                    let qki = match self.current_qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    let _ = writeln!(
                        log,
                        r#"{{"event":"truncation","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"trigger":"{trigger}","terms_before":{total_before},"terms_after":{total_after},"terms_discarded":{actual_discarded},"discarded_coeff_l1":{disc_l1:.6e},"discarded_coeff_max":{disc_max:.6e},"weight_cutoff":{wc_str},"coeff_cutoff":{cc:.6e},"elapsed_ms":{elapsed_ms:.3e}}}"#
>>>>>>> origin/main
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
            let exp_lut: Vec<f64> = (0..=EXP_LUT_SIZE).map(|w| (-d * w as f64).exp()).collect();
            py.allow_threads(|| {
                pool.install(|| {
                    self.thread_maps
                        .par_iter_mut()
                        .zip(self.outboxes.par_iter_mut())
                        .for_each(|(map, outbox_row)| {
                            map.iter_mut().for_each(|(term, coeff)| {
                                *coeff *= exp_lut[term.weight() as usize];
                            });
                            for outbox in outbox_row.iter_mut() {
                                for (term, coeff) in outbox.iter_mut() {
                                    *coeff *= exp_lut[term.weight() as usize];
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

<<<<<<< HEAD
        let circuit_data: Vec<Vec<(M, f64, bool)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool)> {
=======
        let circuit_data: Vec<Vec<(M, f64, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool, Option<usize>)> {
>>>>>>> origin/main
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let angle: f64 = rot.getattr("angle")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
<<<<<<< HEAD
                    Ok((generator, angle, is_intermediate))
=======
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    Ok((generator, angle, is_intermediate, qiskit_gate_idx))
>>>>>>> origin/main
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
<<<<<<< HEAD
            for (idx, (generator, angle, _is_intermediate)) in reversed_layer.iter().enumerate() {
                let added = py.allow_threads(|| self.apply_gate_inplace(generator, *angle));
                pending += added;

=======
            for (idx, (generator, angle, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let added = py.allow_threads(|| self.apply_gate_inplace(generator, *angle));
                pending += added;

                self.current_qiskit_gate_idx = *qiskit_gate_idx;

>>>>>>> origin/main
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
                    let outbox_terms: usize = self.outboxes.iter()
                        .flat_map(|r| r.iter()).map(|v| v.len()).sum();
                    let map_terms = self.total_terms;
<<<<<<< HEAD
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"map_terms":{map_terms},"outbox_terms":{outbox_terms},"avg_ms_per_gate":{avg_ms_per_gate_str}}}"#
=======
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{map_terms},"outbox_terms":{outbox_terms},"avg_ms_per_gate":{avg_ms_per_gate_str}}}"#
>>>>>>> origin/main
                        );
                    }
                }

                // Only flush at compound-gate boundaries. In the reversed iteration,
                // a compound gate [R_final, R_inter, ..., R_inter] ends when the next
                // rotation is not intermediate (or there is no next rotation).
<<<<<<< HEAD
                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni)| *ni);
=======
                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni, _)| *ni);
>>>>>>> origin/main
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
        filename: Option<&str>,
    ) -> PyResult<()>
    where
        M: for<'py> FromPyObject<'py>,
    {
        self.run_propagation_inner(py, evolved, circuit, false)?;
        if let Some(path) = filename {
            save_terms_to_file(&evolved.terms, path)?;
        }
        Ok(())
    }

    pub fn run_expectation_value(
        &mut self,
        py: Python<'_>,
        evolved: &mut AbstractTermSum<M>,
        circuit: &Bound<'_, PyAny>,
        fock_state: u64,
        filename: Option<&str>,
    ) -> PyResult<PropagationResult>
    where
        M: for<'py> FromPyObject<'py>,
    {
        let n_terms = self.run_propagation_inner(py, evolved, circuit, true)?;

        let pool = Arc::clone(&self.pool);
        let total: Complex64 = pool.install(|| {
            evolved.terms
                .par_iter()
                .map(|(term, coeff)| *coeff * term.trace_with_fock_state(fock_state))
                .sum()
        });

        if let Some(path) = filename {
            save_terms_to_file(&evolved.terms, path)?;
        }

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
        self.last_log_instant = None;
        self.last_log_gate_idx = 0;
<<<<<<< HEAD
=======
        self.current_qiskit_gate_idx = None;
>>>>>>> origin/main
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
