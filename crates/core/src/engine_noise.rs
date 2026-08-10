///
/// Resolving a noise channel into a damping factor per term weight.
///
/// Shared by both bases' engines. All three of [`NoiseDispatch`]'s
/// arms are functions of the term's weight alone, so each collapses to one table
/// indexed by weight, built once with the GIL held and then read inside the
/// worker pool. That matters structurally: a `Bound<PyAny>` is not `Sync` and
/// cannot cross into rayon at all, so a per-term callback would force the whole
/// propagation loop back onto the calling thread.
///
/// `active_modes` is passed as zero, exactly as every call site in the previous
/// engine's kernels passed it; nothing has ever supplied a meaning for it.
///
/// **One behaviour change.** The previous engine called a Python noise model once per
/// *term*; this calls it once per distinct *weight*. For a model that honours its
/// documented contract, "return the multiplicative factor for a term of this
/// weight", the two are identical and this is merely faster. A model that
/// returns something different on each call for the same weight, which the
/// interface does not permit, would see the difference.
///
use pyo3::prelude::*;

use crate::native_noise::NativeNoiseModel;
use crate::noise::UniformNoiseModel;


/// How a noise channel is resolved before it is applied.
///
/// Lived on the old propagator; kept here because the three arms are the
/// three kinds of model, not an engine detail.
#[derive(Clone, Copy)]
enum NoiseDispatch {
    /// Built-in uniform damping.
    Uniform(f64),
    /// A dynamically loaded native plugin.
    Native(crate::native_noise::NativeNoiseHandle),
    /// A user-supplied Python model, called back into.
    Python,
}

/// Damping factors indexed by term weight, `0..=max_weight`.
pub type NoiseTable = Vec<f64>;

/// Builds the damping table for `model`, or `None` when there is no model.
///
/// `max_weight` is the largest weight a term can have, which is the unit count:
/// a Pauli's weight is its support and a Majorana's is the weight of its
/// Jordan-Wigner image, and neither can exceed the register.
pub fn resolve_noise_table(
    model: Option<&Bound<'_, PyAny>>,
    max_weight: usize,
) -> PyResult<Option<NoiseTable>> {
    let Some(model) = model else {
        return Ok(None);
    };
    let dispatch = if let Ok(uniform) = model.extract::<PyRef<UniformNoiseModel>>() {
        NoiseDispatch::Uniform(uniform.damping)
    } else if let Ok(native) = model.extract::<PyRef<NativeNoiseModel>>() {
        NoiseDispatch::Native(*native.handle())
    } else {
        NoiseDispatch::Python
    };

    let table = match dispatch {
        NoiseDispatch::Uniform(damping) => {
            (0..=max_weight).map(|w| (-damping * w as f64).exp()).collect()
        }
        NoiseDispatch::Native(handle) => {
            (0..=max_weight).map(|w| handle.damping_factor(w as u32, 0)).collect()
        }
        NoiseDispatch::Python => {
            let mut t = Vec::with_capacity(max_weight + 1);
            for w in 0..=max_weight {
                t.push(model.call_method1("damping_factor", (w as u32, 0u32))?.extract()?);
            }
            t
        }
    };
    Ok(Some(table))
}
