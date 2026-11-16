//!
//! Resolve a noise channel into the engine's backend.
//!
//! The strategy follows what the model *declares it depends on*, not a version
//! number. A model that reads weight alone is collapsed into a lookup table
//! indexed by weight, so the hot loop never crosses the FFI (or GIL) boundary
//! again. A model that also reads the circuit position keeps that fast path,
//! but the table is rebuilt at each layer boundary. Only a model that reads the
//! term's key has to be called per term, which is also what costs the run its
//! Clifford deferral.
//!

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::basis::BasisKind;
use crate::native_noise::NativeNoiseModel;
use crate::noise::{KeyedNoiseModel, UniformNoiseModel};
use crate::term_kernel::{LayerContext, NoiseKernel, TermView};

/// The method a Python model defines to opt into per-term, key-aware damping.
pub const PYTHON_TERM_HOOK: &str = "damping_factor_term";

/// Damping factors indexed by term weight.
pub type NoiseTable = Vec<f64>;

pub enum ResolvedNoise {
    /// A function of weight alone, pre-tabulated once.
    WeightTable(NoiseTable),
    /// A function of weight and circuit position
    LayeredWeightTable(Arc<dyn NoiseKernel>),
    /// A key-aware native kernel.
    TermKernel(Arc<dyn NoiseKernel>),
    /// A key-aware Python object, applied per term with the GIL held.
    PythonTerm(Py<PyAny>),
    /// Factors by the basis `noise_key` program, one table per layer.
    KeyTable(KeyedNoiseModel),
}

impl ResolvedNoise {
    /// True if the noise model reads the term's key, which is what forces the
    /// engine to give up Clifford deferral: a deferred tableau leaves stored
    /// keys pre-conjugation, and a key-reading model has to see physical ones.
    pub fn depends_on_key(&self) -> bool {
        matches!(
            self,
            ResolvedNoise::TermKernel(_) | ResolvedNoise::PythonTerm(_) | ResolvedNoise::KeyTable(_)
        )
    }
}

/// Fills `out` with this kernel's damping factor for every weight `0..=max_weight`.
///
/// The weight-only paths run through here, so the table a plugin produces is
/// the same one a per-term evaluation would, entry for entry. `words` is empty
/// because a model on this path declared it does not read the key.
pub fn retabulate(
    kernel: &dyn NoiseKernel,
    basis_kind: BasisKind,
    max_weight: usize,
    layer: LayerContext,
    out: &mut NoiseTable,
) {
    out.clear();
    out.reserve(max_weight + 1);
    for w in 0..=max_weight {
        out.push(kernel.factor(TermView {
            basis_kind,
            words: &[],
            n_units: max_weight,
            weight: w as u32,
            layer,
        }));
    }
}

fn uniform_table(damping: f64, max_weight: usize) -> ResolvedNoise {
    ResolvedNoise::WeightTable((0..=max_weight).map(|w| (-damping * w as f64).exp()).collect())
}

/// Reads the dict form of a noise setting.
fn resolve_noise_dict(
    dict: &Bound<'_, PyDict>,
    max_weight: usize,
) -> PyResult<Option<ResolvedNoise>> {
    let bad = |m: String| pyo3::exceptions::PyValueError::new_err(m);
    for (k, _) in dict.iter() {
        let k: String = k.extract()?;
        if !matches!(k.as_str(), "damping" | "key_factors" | "default") {
            return Err(bad(format!(
                "unknown noise setting {k:?}; use \"damping\", or \"key_factors\" with \"default\""
            )));
        }
    }
    match (dict.get_item("damping")?, dict.get_item("key_factors")?) {
        (Some(_), Some(_)) => Err(bad("give either \"damping\" or \"key_factors\", not both".into())),
        (Some(d), None) => Ok(Some(uniform_table(d.extract()?, max_weight))),
        (None, Some(f)) => {
            let f = f.cast_into::<PyDict>().map_err(|_| bad("\"key_factors\" must be a dict".into()))?;
            let default = dict.get_item("default")?.map_or(Ok(1.0), |d| d.extract())?;
            let model = KeyedNoiseModel::from_dict(&f, default)?;
            Ok(Some(ResolvedNoise::KeyTable(model)))
        }
        (None, None) => Ok(None),
    }
}

/// Resolves `model` into the form the engine applies, or `None` when there is
/// no model.
pub fn resolve_noise(
    model: Option<&Bound<'_, PyAny>>,
    max_weight: usize,
    basis_kind: BasisKind,
) -> PyResult<Option<ResolvedNoise>> {
    let Some(model) = model else {
        return Ok(None);
    };

    // Plain data: a number is a uniform damping rate. A dict is
    // {"damping": rate} or {"key_factors": {key: factor}, "default": factor}.
    if let Ok(rate) = model.extract::<f64>() {
        return Ok(Some(uniform_table(rate, max_weight)));
    }
    if let Ok(dict) = model.cast::<PyDict>() {
        return resolve_noise_dict(dict, max_weight);
    }

    if let Ok(uniform) = model.extract::<PyRef<UniformNoiseModel>>() {
        return Ok(Some(uniform_table(uniform.damping, max_weight)));
    }
    if let Ok(keyed) = model.extract::<PyRef<KeyedNoiseModel>>() {
        return Ok(Some(ResolvedNoise::KeyTable(keyed.clone())));
    }

    if let Ok(native) = model.extract::<PyRef<NativeNoiseModel>>() {
        let handle = *native.handle();
        let depends = handle.depends();
        if depends.key() {
            return Ok(Some(ResolvedNoise::TermKernel(Arc::new(handle))));
        }
        if depends.layer() {
            return Ok(Some(ResolvedNoise::LayeredWeightTable(Arc::new(handle))));
        }
        let mut table = NoiseTable::new();
        retabulate(
            &handle,
            basis_kind,
            max_weight,
            LayerContext::new(0, 0),
            &mut table,
        );
        return Ok(Some(ResolvedNoise::WeightTable(table)));
    }

    // A GateNoiseModel subclass (or any other custom Python noise object)
    // defines its damping hooks directly on itself.
    if model.hasattr(PYTHON_TERM_HOOK)? {
        return Ok(Some(ResolvedNoise::PythonTerm(model.clone().unbind())));
    }

    let mut table = Vec::with_capacity(max_weight + 1);
    for w in 0..=max_weight {
        table.push(
            model
                .call_method1("damping_factor", (w as u32, 0u32))?
                .extract()?,
        );
    }
    Ok(Some(ResolvedNoise::WeightTable(table)))
}
