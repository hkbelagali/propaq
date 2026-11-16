///
/// Logger for structured event logging of propagator runs.
/// The logger outputs key information, such as term, monomial (for surrogate propagation)
/// gate/truncation events, timing, and other relevant data to a JSONL file.
///
/// On the Python side, there's a LogParser class that can read the JSONL file and
/// provide the information directly as lists. See the example notebook `examples/propaq.ipynb`
/// for usage.
///
use pyo3::prelude::*;

/// Structured event logger for propagator runs, writing JSON Lines to a file.
///
/// Arguments:
///     filename: Path to write the JSON Lines event log.
///     log_every: Emit a gate-event record every N gate applications (default 1).
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(module = "propaq._rust_core")]
pub struct Logger {
    #[pyo3(get)]
    pub filename: String,
    #[pyo3(get)]
    pub log_every: usize,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl Logger {
    /// Configure verbose logging for a propagator run.
    ///
    /// Events are written as JSON Lines to *filename*. Each gate application
    /// and truncation step produces one record.
    ///
    /// *filename* is overwritten (truncated), not appended to.
    ///
    /// Arguments:
    ///     filename: Path to write the JSON Lines event log. Overwritten if it already exists.
    ///     log_every: Emit a gate-event record every N gate applications (default 1).
    #[new]
    #[pyo3(signature = (filename, log_every=1))]
    pub fn new(filename: String, log_every: usize) -> Self {
        Logger {
            filename,
            log_every: log_every.max(1),
        }
    }
}
