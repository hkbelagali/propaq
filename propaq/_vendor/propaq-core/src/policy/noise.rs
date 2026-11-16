//!
//! Noise models for propagator runs.
//!
//! Currently, only uniform depolarising noise is implemented
//! in the Rust core. However, the `GateNoiseModel` class
//! allows one to wrap a custom Python noise model object
//! that implements the same interface as `UniformNoiseModel`.
//! propaq also supports dynamically loaded noise models from
//! shared libraries written in compatible languages. These are
//! significantly faster than Python noise models due to GIL overhead,
//! and are competitive with built-in noise model performance.
//! We strongly recommend custom noise models be implemented
//! via the plugin ABI rather than Python.
//!

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

/// Exponential damping noise: each term of weight w is scaled by \(\exp(-\gamma w)\), where \(w\) is the term's Pauli weight.
///
/// Arguments:
///     damping: Damping rate \(\gamma\).
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct UniformNoiseModel {
    #[pyo3(get, set)]
    pub damping: f64,
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl UniformNoiseModel {
    /// Initialize the uniform noise model.
    ///
    /// Arguments:
    ///     damping: Per-weight damping rate \(\gamma\). Each term is multiplied by \(\exp(-\gamma w)\).
    #[new]
    fn new(damping: f64) -> Self {
        UniformNoiseModel { damping }
    }

    /// Return \(\exp(-\gamma w)\): the multiplicative factor applied to a term's coefficient.
    ///
    /// Arguments:
    ///     term_weight: Pauli weight of the term.
    ///     active_modes: Unused for uniform noise; present for API compatibility.
    // `active_modes` is unused by uniform damping but is part of the documented
    // noise-model interface, and pyo3 exposes the parameter name to Python, so
    // renaming it would break `damping_factor(w, active_modes=0)`.
    #[allow(unused_variables)]
    fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        (-self.damping * term_weight as f64).exp()
    }

    /// Apply uniform damping to all terms in *term_sum* in-place.
    ///
    /// Arguments:
    ///     term_sum: A MajoranaTermSum or PauliTermSum to damp in-place.
    fn apply_noise(&self, py: Python<'_>, term_sum: &Bound<'_, PyAny>) -> PyResult<()> {
        term_sum.call_method1("apply_damping", (self.clone().into_pyobject(py)?, 0u32))?;
        Ok(())
    }
}

/// Base class for a custom Python noise model. Subclass it and define
/// `damping_factor` or `damping_factor_term` directly on the subclass.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct GateNoiseModel;

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl GateNoiseModel {
    /// A subclass's own `__init__` constructor arguments are accepted and
    /// ignored here (`*args`/`**kwargs`)
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        GateNoiseModel
    }

    // Override this in subclasses
    fn damping_factor(&self, _term_weight: u32, _active_modes: u32) -> PyResult<f64> {
        Err(not_overridden("damping_factor"))
    }

    // We don't need to scaffold `damping_factor_term` because its existence
    // determines if the model is key-aware or not.
}

fn not_overridden(method: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "GateNoiseModel.{method} must be overridden by a subclass"
    ))
}

/// Largest key span a dense factor table may cover.
const DENSE_KEY_SPAN: i64 = 1 << 16;

/// Damping factors by noise key, for one layer. Built by [`KeyedNoiseModel::table_for`].
#[derive(Clone, Debug)]
pub struct KeyTable {
    lo: i64,
    dense: Vec<f64>,
    sparse: HashMap<i64, f64>,
    default: f64,
}

impl KeyTable {
    /// The factor of one key. A key without an entry gets the default.
    #[inline]
    pub fn get(&self, key: i64) -> f64 {
        let i = key.wrapping_sub(self.lo);
        if (0..self.dense.len() as i64).contains(&i) {
            return self.dense[i as usize];
        }
        self.sparse.get(&key).copied().unwrap_or(self.default)
    }
}

/// Noise that scales each term by a factor chosen from the term's noise key.
///
/// A traced basis computes the key with its `noise_key` program. This model
/// maps each key to a damping factor. A key without an entry gets `default`.
/// A factor is a number, or a list with one number per layer. Layer 0 is the
/// first layer applied to the observable. A list shorter than the circuit
/// repeats its last entry.
///
/// Arguments:
///     factors: Mapping from noise key (int) to a factor or a list of factors.
///     default: Factor for a key that is not in `factors`. Default 1.0.
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct KeyedNoiseModel {
    entries: Vec<(i64, Vec<f64>)>,
    #[pyo3(get)]
    pub default: f64,
}

fn factor_list(value: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let list: Vec<f64> = match value.extract::<f64>() {
        Ok(x) => vec![x],
        Err(_) => value.extract()?,
    };
    if list.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "a factor list must not be empty",
        ));
    }
    Ok(list)
}

#[pymethods]
impl KeyedNoiseModel {
    #[new]
    #[pyo3(signature = (factors, default=1.0))]
    fn new(factors: &Bound<'_, PyDict>, default: f64) -> PyResult<Self> {
        Self::from_dict(factors, default)
    }

    /// The factors as a dict: key to a list with one factor per layer entry.
    #[getter]
    fn factors(&self) -> HashMap<i64, Vec<f64>> {
        self.entries.iter().cloned().collect()
    }

    /// The factor of `key` in layer `layer`.
    #[pyo3(signature = (key, layer=0))]
    fn factor(&self, key: i64, layer: usize) -> f64 {
        self.table_for(layer).get(key)
    }

    fn __repr__(&self) -> String {
        format!(
            "KeyedNoiseModel({} keys, default={})",
            self.entries.len(),
            self.default
        )
    }
}

impl KeyedNoiseModel {
    /// Reads `{key: factor or [factor per layer]}`.
    pub fn from_dict(factors: &Bound<'_, PyDict>, default: f64) -> PyResult<Self> {
        let mut entries = Vec::with_capacity(factors.len());
        for (k, v) in factors.iter() {
            entries.push((k.extract::<i64>()?, factor_list(&v)?));
        }
        entries.sort_by_key(|e| e.0);
        Ok(KeyedNoiseModel { entries, default })
    }

    /// The factor table of one layer.
    pub fn table_for(&self, layer: usize) -> KeyTable {
        let at = |v: &Vec<f64>| v[layer.min(v.len() - 1)];
        let lo = self.entries.first().map_or(0, |e| e.0);
        let hi = self.entries.last().map_or(0, |e| e.0);
        let span = hi.saturating_sub(lo);
        let mut table = KeyTable {
            lo,
            dense: Vec::new(),
            sparse: HashMap::new(),
            default: self.default,
        };
        if span < DENSE_KEY_SPAN {
            table.dense = vec![self.default; (span + 1) as usize];
            for (k, v) in &self.entries {
                table.dense[(k - lo) as usize] = at(v);
            }
        } else {
            for (k, v) in &self.entries {
                table.sparse.insert(*k, at(v));
            }
        }
        table
    }
}
