use pyo3::prelude::*;

// NOTE: mimalloc was tried here as the global allocator to reduce contention
// from heavy concurrent small-allocation traffic, but was reverted: on the
// real long-running surrogate workload it retains freed memory in per-thread
// arenas for reuse rather than returning it to the OS promptly (a deliberate
// throughput/latency trade-off for that allocator), which inflated RSS far
// beyond live data size and lowered the effective memory ceiling badly
// (observed failing around 250GB where the default allocator ran to ~1TB
// fine). Do not re-add without confirming retention/purge behavior is tuned
// for this alloc/free-heavy, memory-ceiling-constrained workload.

use propaq_core::{TruncationPolicy, UniformNoiseModel, GateNoiseModel, PropagationResult, Logger};
use propaq_majorana::{MajoranaMonomial, MajoranaTermSum, MajoranaPropagator, MajoranaTermStreamer};
use propaq_pauli::{PauliString, PauliTermSum, PauliPropagator, PauliTermStreamer};
use propaq_surrogate::{
    FrequencyTruncationPolicy,
    PauliSurrogateModel, MajoranaSurrogateModel,
    PauliSurrogatePropagator, MajoranaSurrogatePropagator,
};

#[pyfunction]
fn rust_available() -> bool {
    true
}

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_available, m)?)?;
    m.add_class::<MajoranaMonomial>()?;
    m.add_class::<PauliString>()?;
    m.add_class::<MajoranaTermSum>()?;
    m.add_class::<MajoranaTermStreamer>()?;
    m.add_class::<PauliTermSum>()?;
    m.add_class::<PauliTermStreamer>()?;
    m.add_class::<TruncationPolicy>()?;
    m.add_class::<UniformNoiseModel>()?;
    m.add_class::<GateNoiseModel>()?;
    m.add_class::<MajoranaPropagator>()?;
    m.add_class::<PauliPropagator>()?;
    m.add_class::<PropagationResult>()?;
    m.add_class::<Logger>()?;
    m.add_class::<FrequencyTruncationPolicy>()?;
    m.add_class::<PauliSurrogateModel>()?;
    m.add_class::<MajoranaSurrogateModel>()?;
    m.add_class::<PauliSurrogatePropagator>()?;
    m.add_class::<MajoranaSurrogatePropagator>()?;
    Ok(())
}
