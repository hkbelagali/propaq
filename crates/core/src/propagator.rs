///
/// The main propagator implementation. 
///
/// The propagator employs a merging-BFS strategy to exploit 
/// multithreading. Each thread is responsible for a disjoint 
/// partition of the term space, and each thread maintains a 
/// hashmap of terms in its partition, as well as a set of 
/// outboxes for terms belonging to other partitions.
///
/// When a gate is applied, each thread iterates over its terms 
/// (both in its partition and in the outboxes) to apply the gate.
/// The resulting terms are placed in the appropriate outboxes.
///
/// Truncation demands uniqueness of terms in the containers, so 
/// prior to a truncation step, the outboxes are merged into the 
/// partition maps. This is done using a parallel transpose 
/// operation, where each destination thread drains its 
/// column of outboxes into its own partition map.
///
/// This deferred merge strategy aims to prevent lock contention 
/// while preventing unnecessary memory allocation for duplicate 
/// threads. The outbox flush becomes a bottleneck, 
/// but the cost is amortized over the propagation run.
///
use pyo3::prelude::*;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use std::io::{BufReader, BufWriter, Read, Write};
use std::fs::OpenOptions;
use rustc_hash::{FxBuildHasher, FxHashMap};
use flate2::write::GzEncoder;
use flate2::read::GzDecoder;
use flate2::Compression;

use crate::termsum::AbstractTermSum;
use crate::noise::UniformNoiseModel;
use crate::truncators::{resolve_config, FlushSchedule, ResolvedConfig, Truncator};
use crate::traits::AbstractTerm;
use crate::coeff::CoeffRepr;
use crate::logger::Logger;

/// Fibonacci hashing multiplier for multiply-shift uniform partition distribution.
/// This is used to distribute the terms (roughly evenly) over the thread-local partitions.
const PARTITION_HASH_MUL: u64 = 0x517cc1b727220a95;

/// Per-partition term map. 
type PartitionMap<M, C> = hashbrown::HashMap<M, C, FxBuildHasher>;

const EXP_LUT_SIZE: usize = 4096;

/// Minimum chunk size before `apply_gate_inplace` splits sub-partition work
/// further. Deliberately low relative to `evaluate`'s
/// analogous threshold: per-term cost here can be arbitrarily large and
/// arbitrarily skewed (one term's whole symbolic coefficient), unlike
/// `evaluate`'s near-uniform per-monomial cost, so it's worth being more
/// willing to split even at modest partition sizes.
///
/// Overridable once at process start via the `PROPAQ_GATE_PAR_MIN_LEN`
/// environment variable so it can be swept.
const GATE_PAR_MIN_LEN_DEFAULT: usize = 256;

#[inline]
fn gate_par_min_len() -> usize {
    static V: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("PROPAQ_GATE_PAR_MIN_LEN")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(GATE_PAR_MIN_LEN_DEFAULT)
    });
    *V
}

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
///     f64               coefficient (real)
pub fn save_terms_to_file<M: AbstractTerm>(
    terms: &FxHashMap<M, f64>,
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
        enc.write_all(&coeff.to_le_bytes())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    }

    enc.finish()
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    Ok(())
}

/// Deserialize a term map from a file produced by `save_terms_to_file`.
pub fn load_terms_from_file<M: AbstractTerm>(path: &str) -> PyResult<FxHashMap<M, f64>> {
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
        let coeff = f64::from_le_bytes(f64_buf);
        let term = M::from_bytes_vec(&key_buf, system_size);
        terms.insert(term, coeff);
    }
    Ok(terms)
}

/// Multiply-shift partition hash: XOR-folds the term's bits, then maps
/// uniformly into [0, n_partitions) where n_partitions is a power of two.
#[inline]
pub fn owner_of<M: AbstractTerm>(term: &M, log2_n: u32) -> usize {
    if log2_n == 0 {
        return 0;
    }
    (term.partition_key().wrapping_mul(PARTITION_HASH_MUL) >> (64 - log2_n)) as usize
}

pub struct AbstractPropagator<M: AbstractTerm, C: CoeffRepr> {
    pub noise: Option<PyObject>,
    pub schedule: FlushSchedule,
    pub truncators: Vec<Truncator>,
    pub pool: Arc<rayon::ThreadPool>,
    pub progress_bar: bool,
    n_partitions: usize,
    log2_n: u32,
    thread_maps: Vec<PartitionMap<M, C>>,
    outboxes: Vec<Vec<Vec<(M, C)>>>,
    total_terms: usize,
    scratch_new_terms: Vec<Vec<(usize, M, C)>>,
    scratch_snap: Vec<Vec<usize>>,
    verbose_log: Option<BufWriter<std::fs::File>>,
    log_filename: Option<String>,
    log_every: usize,
    last_log_instant: Option<std::time::Instant>,
    last_log_gate_idx: usize,
    current_qiskit_gate_idx: Option<usize>,
    _marker: PhantomData<(M, C)>,
}

impl<M: AbstractTerm, C: CoeffRepr> AbstractPropagator<M, C> {
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
        let n_threads_actual = pool.current_num_threads();
        let n_partitions = n_threads_actual.next_power_of_two().max(1);
        let log2_n = n_partitions.trailing_zeros();
        let thread_maps = (0..n_partitions).map(|_| PartitionMap::default()).collect();
        let outboxes = (0..n_partitions)
            .map(|_| (0..n_partitions).map(|_| Vec::new()).collect())
            .collect();
        let scratch_new_terms = (0..n_partitions).map(|_| Vec::new()).collect();
        let scratch_snap = (0..n_partitions).map(|_| Vec::new()).collect();
        let (log_filename, log_every) = match logger {
            Some(ref obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<pyo3::PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(AbstractPropagator {
            noise,
            schedule,
            truncators,
            pool,
            progress_bar,
            n_partitions,
            log2_n,
            thread_maps,
            outboxes,
            total_terms: 0,
            scratch_new_terms,
            scratch_snap,
            verbose_log: None,
            log_filename,
            log_every,
            last_log_instant: None,
            last_log_gate_idx: 0,
            current_qiskit_gate_idx: None,
            _marker: PhantomData,
        })
    }

    /// Partition `evolved.terms` into per-partition maps at the start of a run.
    /// Parallelized over partitions, but not terms within a partition.
    pub fn initialize_from(&mut self, evolved: &AbstractTermSum<M>) {
        let n = self.n_partitions;
        let log2_n = self.log2_n;

        let mut buckets: Vec<Vec<(&M, C)>> = (0..n).map(|_| Vec::new()).collect();
        for (term, coeff) in &evolved.terms {
            buckets[owner_of(term, log2_n)].push((term, C::from_real(*coeff)));
        }

        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(buckets.par_iter_mut())
                .for_each(|(map, bucket)| {
                    map.clear();
                    map.reserve(bucket.len());
                    for (term, coeff) in bucket.drain(..) {
                        map.insert((*term).clone(), coeff);
                    }
                });
        });

        self.total_terms = evolved.terms.len();
    }

    /// Apply a gate in place. Returns `(count, size)`: the number of new
    /// outbox entries created, and the sum of their `CoeffRepr::size_hint()`
    /// (equal to `count` for scalar coefficients; the total monomial count
    /// added for `SymbolicCoeff`).
    ///
    /// This is the probably the most frequently used function in the entire 
    /// propagator, so it's parallelized over both partitions and terms. 
    /// The latter is largely unnecessary for numerical coefficients 
    /// since the per-term cost is O(1), but relevant for symbolic coefficients 
    /// where the information processed grows with the number of monomials 
    /// in the coefficient. This setup allows work-stealing to balance the load 
    /// across threads. 
    /// 
    /// Below a certain threshold, the per-partition work is done serially to avoid 
    /// the unnecessary overhead of splitting and joining tasks. This is set by the 
    /// `PROPAQ_GATE_PAR_MIN_LEN` environment variable.
    pub fn apply_gate_inplace(&mut self, generator: &M, param: C::GateParam) -> (usize, usize) {
        let log2_n = self.log2_n;
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        let outboxes = &mut self.outboxes;
        let scratch_new_terms = &mut self.scratch_new_terms;
        let scratch_snap = &mut self.scratch_snap;

        /// Collect all of the terms, both in the partition map and in the 
        /// outboxes, that anticommute with the generator. 
        /// Apply the gate to each of these terms, and place the results 
        /// in the appropriate outboxes.
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(outboxes.par_iter_mut())
                .zip(scratch_new_terms.par_iter_mut())
                .zip(scratch_snap.par_iter_mut())
                .map(|(((local_map, outbox_row), new_terms), snap)| {
                    new_terms.clear();
                    
                    /// If the workload is high, then sub-threads are spawned to process the partition in parallel.
                    if local_map.len() >= gate_par_min_len() {
                        new_terms.par_extend(
                            local_map
                                .par_iter_mut()
                                .filter_map(|(term, coeff)| {
                                    if term.commutes_with(generator) {
                                        return None;
                                    }
                                    let (phase, new_term) = generator.matmul_internal(term);
                                    let new_coeff = coeff.apply_rotation(&param, phase);
                                    let dst = owner_of(&new_term, log2_n);
                                    Some((dst, new_term, new_coeff))
                                }),
                        );
                    } else { /// Defer to the serial loop  (per-thread) for smaller partitions to avoid unnecessary parallel overhead
                        for (term, coeff) in local_map.iter_mut() {
                            if !term.commutes_with(generator) {
                                let (phase, new_term) = generator.matmul_internal(term);
                                let new_coeff = coeff.apply_rotation(&param, phase);
                                let dst = owner_of(&new_term, log2_n);
                                new_terms.push((dst, new_term, new_coeff));
                            }
                        }
                    }

                    // Apply gate to existing outbox items. Snapshot lengths first so
                    // items appended in this same pass (the drain below, which only
                    // happens after both loops complete) are not re-processed.
                    snap.clear();
                    snap.extend(outbox_row.iter().map(|v| v.len()));
                    new_terms.par_extend(
                        outbox_row
                            .par_iter_mut()
                            .zip(snap.par_iter())
                            .flat_map(|(bucket, &take)| {
                                bucket[..take]
                                    .par_iter_mut()
                                    .with_min_len(gate_par_min_len())
                                    .filter_map(|(term, coeff)| {
                                        if term.commutes_with(generator) {
                                            return None;
                                        }
                                        let (phase, new_term) = generator.matmul_internal(term);
                                        let new_coeff = coeff.apply_rotation(&param, phase);
                                        let new_dst = owner_of(&new_term, log2_n);
                                        Some((new_dst, new_term, new_coeff))
                                    })
                            }),
                    );

                    let count = new_terms.len();
                    let mut size = 0usize;
                    for (dst, term, coeff) in new_terms.drain(..) {
                        size += coeff.size_hint();
                        outbox_row[dst].push((term, coeff));
                    }
                    (count, size)
                })
                .reduce(|| (0usize, 0usize), |(ac, asz), (c, sz)| (ac + c, asz + sz))
        })
    }

    /// Transpose all outboxes directly into thread_maps and update
    /// `total_terms`.
    pub fn flush_outboxes_to_maps(&mut self) {
        let n = self.n_partitions;
        let pool = Arc::clone(&self.pool);
        let outboxes = &mut self.outboxes;
        let thread_maps = &mut self.thread_maps;
        let outboxes_ptr = SendPtr(outboxes.as_mut_ptr());
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .enumerate()
                .for_each(|(dst, map)| {
                    for src in 0..n {
                        // SAFETY: Each parallel task owns a unique `dst`.
                        // Thread `dst` drains outboxes[src][dst] for all src.
                        // No two threads share a (src, dst) pair. 
                        // `outboxes` is not resized during this block 
                        // (n is fixed, no push occurs).
                        let cell = unsafe { &mut (&mut *outboxes_ptr.offset(src))[dst] };
                        const PREFETCH_AHEAD: usize = 4;
                        let n_cell = cell.len();
                        // SAFETY: length is zeroed *before* the manual moves
                        // below, so a panic mid-loop leaks the unmoved
                        // tail instead of double-dropping the moved head. The
                        // buffer itself stays allocated and valid.
                        unsafe { cell.set_len(0) };
                        let base = cell.as_ptr();
                        for i in 0..n_cell {
                            if i + PREFETCH_AHEAD < n_cell {
                                // SAFETY: in-bounds read of an element we
                                // exclusively own.
                                unsafe { (*base.add(i + PREFETCH_AHEAD)).1.prefetch_read() };
                            }
                            // SAFETY: each index in 0..n_cell is read exactly
                            // once, transferring ownership out of the buffer.
                            let (term, coeff) = unsafe { std::ptr::read(base.add(i)) };
                            let entry = map.entry(term).or_default();
                            entry.add_assign(coeff);
                            entry.post_merge();
                        }
                    }
                });
        });

        self.total_terms = self.thread_maps.iter().map(|m| m.len()).sum();
    }

    /// Apply uniform per-weight noise damping to all live terms and outbox items.
    pub fn apply_uniform_noise_inplace(&mut self, damping: f64) {
        /// We store the exponential of the damping factor 
        // for each weight in an LUT to avoid repeatedly
        // computing the same thing over and over.
        let exp_lut: Vec<f64> = (0..=EXP_LUT_SIZE).map(|w| (-damping * w as f64).exp()).collect();
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        let outboxes = &mut self.outboxes;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(outboxes.par_iter_mut())
                .for_each(|(map, outbox_row)| {
                    map.iter_mut().for_each(|(term, coeff)| {
                        coeff.scale_real(exp_lut[term.weight() as usize]);
                    });
                    for outbox in outbox_row.iter_mut() {
                        for (term, coeff) in outbox.iter_mut() {
                            coeff.scale_real(exp_lut[term.weight() as usize]);
                        }
                    }
                });
        });
    }

    pub fn open_log(&mut self) -> PyResult<()> {
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
        self.current_qiskit_gate_idx = None;
        Ok(())
    }

    pub fn uniform_damping(&self, py: Python<'_>) -> Option<f64> {
        if let Some(ref noise_obj) = self.noise {
            let noise = noise_obj.bind(py);
            if let Ok(unm) = noise.extract::<PyRef<UniformNoiseModel>>() {
                return Some(unm.damping);
            }
        }
        None
    }

    /// Helper functions for using tqdm progress bars in Python.
    /// These claim and release the GIL since they are called 
    /// from the main thread. 
    pub fn make_progress_bar(
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

    pub fn tick_progress_bar(
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

    pub fn close_progress_bar(py: Python<'_>, pbar: &Option<Py<PyAny>>) -> PyResult<()> {
        if let Some(pbar) = pbar {
            pbar.bind(py).call_method0("close")?;
        }
        Ok(())
    }

    /// Current number of live terms across all partitions.
    #[inline]
    pub fn total_terms(&self) -> usize {
        self.total_terms
    }

    /// Number of items pending in all outboxes (not yet flushed).
    pub fn n_outbox_terms(&self) -> usize {
        self.outboxes.iter().flat_map(|r| r.iter()).map(|v| v.len()).sum()
    }

    /// Log-every value configured at construction.
    #[inline]
    pub fn log_every(&self) -> usize {
        self.log_every
    }

    /// Current qiskit gate index being processed, for log entries.
    #[inline]
    pub fn current_qiskit_gate_idx(&self) -> Option<usize> {
        self.current_qiskit_gate_idx
    }

    /// Set current qiskit gate index.
    #[inline]
    pub fn set_current_qiskit_gate_idx(&mut self, idx: Option<usize>) {
        self.current_qiskit_gate_idx = idx;
    }

    /// Mutate every coefficient across all partition maps with `f`, then drop
    /// entries for which `keep` returns `false`. 
    ///
    // Returns the total `size_hint()` across all surviving coefficients.
    pub fn map_and_retain_coeffs_inplace<F, K>(&mut self, f: F, keep: K) -> usize
    where
        F: Fn(&M, &mut C) + Sync,
        K: Fn(&M, &C) -> bool + Sync,
    {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        let total_size = pool.install(|| {
            thread_maps
                .par_iter_mut()
                .map(|map| {
                    let (size, n_discard) = map
                        .par_iter_mut()
                        .fold(
                            || (0usize, 0usize),
                            |(mut size, mut n_discard), (t, c)| {
                                f(t, c);
                                if keep(t, c) {
                                    size += c.size_hint();
                                } else {
                                    n_discard += 1;
                                }
                                (size, n_discard)
                            },
                        )
                        .reduce(|| (0usize, 0usize), |(s1, d1), (s2, d2)| (s1 + s2, d1 + d2));
                    if n_discard > 0 {
                        map.retain(|t, c| keep(t, c));
                    }
                    size
                })
                .sum()
        });
        self.total_terms = self.thread_maps.iter().map(|m| m.len()).sum();
        total_size
    }

    /// Apply `f` to every live coefficient across all partition maps, in
    /// parallel over both partitions and the coefficients within each. `f` must
    /// mutate only the coefficient it is given (no shared state), so callers use
    /// this for independent per-coefficient work like `deduplicate`.
    pub fn par_for_each_coeff_mut<F>(&mut self, f: F)
    where
        F: Fn(&mut C) + Sync,
    {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps.par_iter_mut().for_each(|map| {
                map.par_iter_mut().for_each(|(_t, c)| f(c));
            });
        });
    }

    pub fn drain_collect_terms<R, F>(&mut self, f: F) -> Vec<R>
    where
        R: Send,
        F: Fn(M, C) -> Option<R> + Sync,
    {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .flat_map_iter(|map| map.drain().filter_map(|(t, c)| f(t, c)))
                .collect()
        })
    }

    /// Parallel sum of a per-coefficient quantity over all live terms, without
    /// allocating a result the size of the term count (unlike `collect_terms`).
    pub fn sum_coeffs<F>(&self, f: F) -> usize
    where
        F: Fn(&C) -> usize + Sync,
    {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        pool.install(|| {
            thread_maps.par_iter().map(|map| map.values().map(&f).sum::<usize>()).sum()
        })
    }

    /// Parallel per-coefficient fold across all partition maps, merged via
    /// `combine`. 
    pub fn fold_coeffs<T, F, R>(&self, identity: impl Fn() -> T + Sync, fold: F, combine: R) -> T
    where
        T: Send,
        F: Fn(T, &C) -> T + Sync,
        R: Fn(T, T) -> T + Sync,
    {
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter()
                .map(|map| map.values().fold(identity(), &fold))
                .reduce(&identity, &combine)
        })
    }

    /// Sequentially apply `f` to every live coefficient across all partition maps.
    pub fn for_each_coeff_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut C),
    {
        for map in self.thread_maps.iter_mut() {
            for coeff in map.values_mut() {
                f(coeff);
            }
        }
    }

    /// Access verbose log writer for writing custom log entries from surrogate propagator.
    pub fn verbose_log_mut(&mut self) -> Option<&mut BufWriter<std::fs::File>> {
        self.verbose_log.as_mut()
    }

    /// Whether verbose logging is active.
    pub fn has_verbose_log(&self) -> bool {
        self.verbose_log.is_some()
    }

    /// Timing fields for logging, exposed for external run loops.
    pub fn last_log_instant(&self) -> Option<std::time::Instant> {
        self.last_log_instant
    }

    pub fn set_last_log_instant(&mut self, t: Option<std::time::Instant>) {
        self.last_log_instant = t;
    }

    pub fn last_log_gate_idx(&self) -> usize {
        self.last_log_gate_idx
    }

    pub fn set_last_log_gate_idx(&mut self, idx: usize) {
        self.last_log_gate_idx = idx;
    }
}

impl<M: AbstractTerm> AbstractPropagator<M, f64> {
    /// Remove terms from thread_maps that fail the resolved weight/coefficient
    /// cutoffs
    fn retain_by_policy(&mut self, cfg: &ResolvedConfig) {
        let wc = cfg.weight;
        let cc = cfg.coefficient.unwrap_or(0.0);
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps.par_iter_mut().for_each(|map| {
                map.retain(|t, c| {
                    wc.map_or(true, |w| t.weight() <= w) && c.abs() >= cc
                });
            });
        });
    }

    /// Apply the resolved cutoffs only to threads where `mask[i]` is true.
    fn retain_by_policy_masked(&mut self, cfg: &ResolvedConfig, mask: &[bool]) {
        let wc = cfg.weight;
        let cc = cfg.coefficient.unwrap_or(0.0);
        let pool = Arc::clone(&self.pool);
        let thread_maps = &mut self.thread_maps;
        pool.install(|| {
            thread_maps
                .par_iter_mut()
                .zip(mask.par_iter())
                .for_each(|(map, &apply)| {
                    if apply {
                        map.retain(|t, c| wc.map_or(true, |w| t.weight() <= w) && c.abs() >= cc);
                    }
                });
        });
    }

    /// Single parallel pass returning per-thread surviving counts and aggregate discard
    /// statistics (discarded_coeff_l1, discarded_coeff_max).
    fn collect_stats_and_count_surviving(&self, cfg: &ResolvedConfig) -> ((f64, f64), Vec<usize>) {
        let wc = cfg.weight;
        let cc = cfg.coefficient.unwrap_or(0.0);
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        let per_thread: Vec<((f64, f64), usize)> = pool.install(|| {
            thread_maps
                .par_iter()
                .map(|map| {
                    let (mut dl1, mut dmax, mut surv) = (0.0f64, 0.0f64, 0usize);
                    for (t, c) in map.iter() {
                        if wc.map_or(true, |w| t.weight() <= w) && c.abs() >= cc {
                            surv += 1;
                        } else {
                            let norm = c.abs();
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
        cfg: Option<&ResolvedConfig>,
        gate_idx: usize,
        layer_idx: usize,
        trigger: &str,
    ) {
        let t0 = std::time::Instant::now();

        self.flush_outboxes_to_maps();
        let total_before = self.total_terms;

        // A weight or coefficient cutoff is the only lossy term-level filter for
        // the numerical propagator; `None` cfg (a noise pre-flush) or a cfg with
        // neither filter means "flush only".
        if let Some(cfg) = cfg.filter(|c| c.weight.is_some() || c.coefficient.is_some()) {
            let min_terms = cfg.min_terms.unwrap_or(0);
            if total_before >= min_terms {
                let need_surviving = min_terms > 0;
                let need_stats = self.verbose_log.is_some();

                let (disc_l1, disc_max, surviving) = if need_surviving || need_stats {
                    let ((dl1, dmax), surv) = self.collect_stats_and_count_surviving(cfg);
                    let (dl1, dmax) = if need_stats { (dl1, dmax) } else { (0.0, 0.0) };
                    (dl1, dmax, surv)
                } else {
                    (0.0, 0.0, Vec::new())
                };

                if need_surviving {
                    let total_surviving: usize = surviving.iter().sum();
                    if total_surviving < min_terms {
                        let mask: Vec<bool> = surviving.iter()
                            .zip(self.thread_maps.iter())
                            .map(|(&surv, map)| surv >= map.len() * min_terms / total_before)
                            .collect();
                        self.retain_by_policy_masked(cfg, &mask);
                    } else {
                        self.retain_by_policy(cfg);
                    }
                } else {
                    self.retain_by_policy(cfg);
                }

                let total_after: usize = self.thread_maps.iter().map(|m| m.len()).sum();
                self.total_terms = total_after;

                if let Some(ref mut log) = self.verbose_log {
                    let actual_discarded = total_before - total_after;
                    let wc_str = cfg.weight
                        .map_or_else(|| "null".to_string(), |w| w.to_string());
                    let cc = cfg.coefficient.unwrap_or(0.0);
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
        }
    }

    /// Apply per-term damping to all live terms: thread_maps and outboxes alike.
    fn apply_layer_noise(
        &mut self,
        py: Python<'_>,
        damping: Option<f64>,
        gate_idx: usize,
        layer_idx: usize,
    ) -> PyResult<()> {
        if self.noise.is_none() {
            return Ok(());
        }

        if let Some(d) = damping {
            py.allow_threads(|| self.apply_uniform_noise_inplace(d));
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

    /// Reassemble `evolved.terms` from per-partition maps at the end of a run.
    fn finalize_to(&self, evolved: &mut AbstractTermSum<M>) {
        evolved.terms.clear();
        evolved.terms.reserve(self.total_terms);
        let pool = Arc::clone(&self.pool);
        let thread_maps = &self.thread_maps;
        let all_items: Vec<(M, f64)> = pool.install(|| {
            thread_maps
                .par_iter()
                .flat_map_iter(|map| map.iter().map(|(k, v)| (k.clone(), *v)))
                .collect()
        });
        evolved.terms.extend(all_items);
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

        let circuit_data: Vec<Vec<(M, f64, bool, Option<usize>)>> = layers
            .iter()
            .map(|layer| {
                layer.iter().map(|rot_obj| -> PyResult<(M, f64, bool, Option<usize>)> {
                    let rot = rot_obj.bind(py);
                    let generator: M = rot.getattr("generator")?.extract()?;
                    let angle: f64 = rot.getattr("angle")?.extract()?;
                    let is_intermediate: bool = rot.getattr("is_intermediate")?.extract()?;
                    let qiskit_gate_idx: Option<usize> = rot
                        .getattr("qiskit_gate_idx")
                        .ok()
                        .and_then(|v| v.extract::<Option<usize>>().ok())
                        .flatten();
                    Ok((generator, angle, is_intermediate, qiskit_gate_idx))
                }).collect::<PyResult<_>>()
            })
            .collect::<PyResult<_>>()?;

        // Resolve the truncation pipeline once (Copy config: the flush trigger
        // and the weight/coefficient/min-terms cutoffs the numerical propagator
        // honors). Symbolic-only operators are rejected at construction, so any
        // present here are numerical-applicable.
        let cfg = resolve_config(&self.truncators);
        let max_terms: Option<usize> = cfg.max_terms;
        let merge_max_terms: Option<usize> = self.schedule.merge_max_terms;

        let mut n_terms: Vec<usize> = Vec::new();
        let damping = self.uniform_damping(py);
        let total_rotations: usize = circuit_data.iter().map(|l| l.len()).sum();
        let (pbar, postfix) = self.make_progress_bar(py, total_rotations)?;

        self.initialize_from(evolved);

        let mut gate_idx: usize = 0;
        let mut pending: usize = 0;
        for (layer_idx, layer_data) in circuit_data.iter().rev().enumerate() {
            self.apply_layer_noise(py, damping, gate_idx, layer_idx)?;

            let reversed_layer: Vec<_> = layer_data.iter().rev().collect();
            for (idx, (generator, angle, _is_intermediate, qiskit_gate_idx)) in reversed_layer.iter().enumerate() {
                let (added, _) = py.allow_threads(|| self.apply_gate_inplace(generator, *angle));
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
                    let outbox_terms: usize = self.outboxes.iter()
                        .flat_map(|r| r.iter()).map(|v| v.len()).sum();
                    let map_terms = self.total_terms;
                    let qki = match qiskit_gate_idx {
                        Some(v) => v.to_string(),
                        None => "null".to_string(),
                    };
                    if let Some(ref mut log) = self.verbose_log {
                        let _ = writeln!(
                            log,
                            r#"{{"event":"gate","gate_idx":{gate_idx},"layer_idx":{layer_idx},"qiskit_gate_idx":{qki},"map_terms":{map_terms},"outbox_terms":{outbox_terms},"avg_ms_per_gate":{avg_ms_per_gate_str}}}"#
                        );
                    }
                }

                let next_is_intermediate = reversed_layer.get(idx + 1).map_or(false, |(_, _, ni, _)| *ni);
                if !next_is_intermediate && max_terms.map_or(false, |max| self.total_terms + pending >= max) {
                    py.allow_threads(|| self.flush_and_maybe_truncate(Some(&cfg), gate_idx, layer_idx, "threshold"));
                    pending = 0;
                } else if !next_is_intermediate && merge_max_terms.map_or(false, |m| pending >= m) {
                    // Finer lossless merge cadence (shared with the surrogate):
                    // collapse duplicate strings out of the outboxes without
                    // truncating, curbing within-window path-count growth.
                    py.allow_threads(|| self.flush_outboxes_to_maps());
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

        py.allow_threads(|| self.flush_and_maybe_truncate(Some(&cfg), gate_idx, circuit_data.len(), "final"));

        self.finalize_to(evolved);

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
        let total: f64 = pool.install(|| {
            evolved.terms
                .par_iter()
                .map(|(term, coeff)| *coeff * term.trace_with_fock_state(fock_state))
                .sum()
        });

        if let Some(path) = filename {
            save_terms_to_file(&evolved.terms, path)?;
        }

        Ok(PropagationResult { n_terms, expectation_value: total })
    }
}
