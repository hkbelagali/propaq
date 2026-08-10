//! 
//! Parse frontend objects into a runtime configuration for the propagator. 
//! 

use std::sync::Arc;

use pyo3::prelude::*;

use crate::logger::Logger;
use crate::truncators::Truncator;

/// Pool and settings for a propagator.
pub struct RunConfig {
    /// One worker per partition, pinned unless the caller opted out.
    pub pool: Arc<rayon::ThreadPool>,
    /// The noise model, as the Python object
    pub noise: Option<Py<PyAny>>,
    /// The truncation pipeline
    pub truncators: Vec<Truncator>,
    /// Where verbose events go, if anywhere.
    pub log_filename: Option<String>,
    /// Emit one record every this many gates.
    pub log_every: usize,
}

impl RunConfig {
    /// Builds the pool and resolves the logger.
    pub fn new(
        noise: Option<Py<PyAny>>,
        truncators: Vec<Truncator>,
        n_threads: Option<usize>,
        logger: Option<Py<PyAny>>,
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
            builder
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );
        let (log_filename, log_every) = match logger {
            Some(ref obj) => Python::attach(|py| -> PyResult<_> {
                let lg = obj.bind(py).extract::<PyRef<Logger>>()?;
                Ok((Some(lg.filename.clone()), lg.log_every))
            })?,
            None => (None, 1),
        };
        Ok(RunConfig {
            pool,
            noise,
            truncators,
            log_filename,
            log_every,
        })
    }

    /// Partitions to use, which is one per worker.
    pub fn partitions(&self) -> usize {
        self.pool.current_num_threads().max(1)
    }
}
