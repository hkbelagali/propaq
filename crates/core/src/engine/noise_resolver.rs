//! 
//! Resolve a noise channel into the engine's backend.
//! 
//! Noise models that are a function of weight are parsed 
//! into a lookup table indexed by weight to avoid GIL contention.
//! This does not apply to a noise model that accepts the term 
//! as an argument, which is instead resolved into a 
//! [`crate::term_kernel::NoiseKernel`] with a Sync trait, 
//! or a serial prototype for Python objects.
//! 

use std::sync::Arc;

use pyo3::prelude::*;

use crate::native_noise::NativeNoiseModel;
use crate::noise::{GateNoiseModel, UniformNoiseModel};
use crate::term_kernel::NoiseKernel;

/// The method a Python model defines to opt into per-term, key-aware damping.
pub const PYTHON_TERM_HOOK: &str = "damping_factor_term";

/// Damping factors indexed by term weight.
pub type NoiseTable = Vec<f64>;

pub enum ResolvedNoise {
    /// A function of weight alone, pre-tabulated.
    WeightTable(NoiseTable),
    /// A key-aware native kernel.
    TermKernel(Arc<dyn NoiseKernel>),
    /// A key-aware Python object, applied per term with the GIL held.
    PythonTerm(Py<PyAny>),
}

impl ResolvedNoise {
    /// True if the noise model requires the term as a parameter. 
    /// In this case, the engine disables Clifford deferral for 
    /// simplicity. 
    pub fn is_term_aware(&self) -> bool {
        !matches!(self, ResolvedNoise::WeightTable(_))
    }
}

/// Resolves `model` into the form the engine applies, or `None` when there is
/// no model.
pub fn resolve_noise(
    model: Option<&Bound<'_, PyAny>>,
    max_weight: usize,
) -> PyResult<Option<ResolvedNoise>> {
    let Some(model) = model else {
        return Ok(None);
    };

    if let Ok(uniform) = model.extract::<PyRef<UniformNoiseModel>>() {
        let damping = uniform.damping;
        return Ok(Some(ResolvedNoise::WeightTable(
            (0..=max_weight)
                .map(|w| (-damping * w as f64).exp())
                .collect(),
        )));
    }

    if let Ok(native) = model.extract::<PyRef<NativeNoiseModel>>() {
        let handle = *native.handle();
        if handle.is_term_aware() {
            return Ok(Some(ResolvedNoise::TermKernel(Arc::new(handle))));
        }
        return Ok(Some(ResolvedNoise::WeightTable(
            (0..=max_weight)
                .map(|w| handle.damping_factor(w as u32, 0))
                .collect(),
        )));
    }

    /// Unwrap the wrapper into a parsed noise model.
    let target = match model.extract::<PyRef<GateNoiseModel>>() {
        Ok(gate) => gate.inner(model.py()).into_bound(model.py()),
        Err(_) => model.clone(),
    };

    if target.hasattr(PYTHON_TERM_HOOK)? {
        return Ok(Some(ResolvedNoise::PythonTerm(target.unbind())));
    }

    let mut table = Vec::with_capacity(max_weight + 1);
    for w in 0..=max_weight {
        table.push(
            target
                .call_method1("damping_factor", (w as u32, 0u32))?
                .extract()?,
        );
    }
    Ok(Some(ResolvedNoise::WeightTable(table)))
}
