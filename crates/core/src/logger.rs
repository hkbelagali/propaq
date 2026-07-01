use pyo3::prelude::*;

/// Structured event logger for propagator runs, writing JSON Lines to a file.
///
/// Arguments:
///     filename: Path to write the JSON Lines event log.
///     log_every: Emit a gate-event record every N gate applications (default 1).
#[pyclass(module = "propaq._rust_core")]
pub struct Logger {
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub log_every: usize,
}

#[pymethods]
impl Logger {
    /// Configure verbose logging for a propagator run.
    ///
    /// Events are written as JSON Lines to *filename*. Each gate application
    /// and truncation step produces one record.
    ///
    /// Arguments:
    ///     filename: Path to write the JSON Lines event log.
    ///     log_every: Emit a gate-event record every N gate applications (default 1).
    #[new]
    #[pyo3(signature = (filename, log_every=1))]
    pub fn new(filename: String, log_every: usize) -> Self {
        Logger { filename, log_every: log_every.max(1) }
    }
}
