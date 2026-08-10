///
/// What a propagator holds between calls: the worker pool and the settings a run
/// reads from it.
///
/// This is all that survives of the old `SoaPropagator`. That type was both a
/// configuration holder and an engine; the engine is now
/// [`crate::partitioned::PartitionedOperator`], driven by each basis's own
/// dispatch, so what is left is the part the Python class owns across calls.
///
use std::sync::Arc;

use pyo3::prelude::*;

use crate::logger::Logger;
use crate::truncators::{FlushSchedule, Truncator};

/// Pool and settings for a propagator.
pub struct RunConfig {
    /// One worker per partition, pinned unless the caller opted out.
    pub pool: Arc<rayon::ThreadPool>,
    /// The noise model, as the Python object; resolving it needs the GIL.
    pub noise: Option<PyObject>,
    /// Retained for API compatibility. The partitioned engine folds duplicates
    /// on insert, so there is no merge cadence to schedule.
    pub schedule: FlushSchedule,
    /// The truncation pipeline, in list order.
    pub truncators: Vec<Truncator>,
    /// Where verbose events go, if anywhere.
    pub log_filename: Option<String>,
    /// Emit one record every this many gates.
    pub log_every: usize,
    /// Accepted for API compatibility and currently inert: the engine runs its
    /// gate loop with the GIL released and cannot drive a tqdm bar from there.
    pub progress_bar: bool,
}

impl RunConfig {
    /// Builds the pool and resolves the logger.
    ///
    /// Pinning binds worker `i` to the `i`th CPU in the process's own affinity
    /// mask, which is what keeps a partition's store in one core's cache across
    /// gates. See the propagator docstrings for when to turn it off.
    pub fn new(
        noise: Option<PyObject>,
        schedule: FlushSchedule,
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        progress_bar: bool,
        logger: Option<PyObject>,
        pin_threads: bool,
    ) -> PyResult<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        if let Some(n) = n_threads {
            builder = builder.num_threads(n);
        }
        if pin_threads {
            let cpus = crate::affinity::available_cpus();
            if !cpus.is_empty() {
                builder = builder.start_handler(move |index| {
                    if let Some(cpu) = crate::affinity::cpu_for_worker(index, &cpus) {
                        crate::affinity::pin_current_thread(cpu);
                    }
                });
            }
        }
        let pool = Arc::new(
            builder.build().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        let (log_filename, log_every) = match logger {
            Some(ref obj) => Python::with_gil(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(RunConfig {
            pool,
            noise,
            schedule,
            truncators,
            log_filename,
            log_every,
            progress_bar,
        })
    }

    /// Partitions to use, which is one per worker.
    pub fn partitions(&self) -> usize {
        self.pool.current_num_threads().max(1)
    }
}
