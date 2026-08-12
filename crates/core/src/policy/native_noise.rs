//!
//! Dynamically loaded noise models!
//! Lets a user supply a noise model compiled from C, Rust, or (AOT-compiled)
//! Julia as a shared library.
//!
//! ## ABI contract
//!
//! There is one entry point. A plugin declares *what it reads* through
//! `propaq_noise_depends`, and that declaration — not a version number — is
//! what picks the engine's evaluation strategy.
//!
//! ```c
//! uint32_t propaq_noise_abi_version(void);                 // required, must equal 1
//! uint32_t propaq_noise_depends(void);                     // optional; absent => 0
//! void*    propaq_noise_create(const char* config_json);   // optional
//! void     propaq_noise_destroy(void* ctx);                // required iff create is present
//!
//! double   propaq_noise_factor(void* ctx, uint32_t basis_kind,
//!                              const uint64_t* words, size_t n_words,
//!                              uint32_t n_units, uint32_t weight,
//!                              uint32_t layer_index, uint32_t n_layers);   // required
//!
//! int32_t  propaq_noise_factor_batch(void* ctx, uint32_t basis_kind,
//!                                    const uint64_t* words, size_t n_words_per_term,
//!                                    uint32_t n_units, const uint32_t* weights,
//!                                    uint32_t layer_index, uint32_t n_layers,
//!                                    double* out, size_t n_terms);          // optional
//! ```
//!
//! `depends` is a bitmask: `PROPAQ_DEPENDS_KEY` (1) if the model reads `words`,
//! `PROPAQ_DEPENDS_LAYER` (2) if it reads the circuit position. Declaring
//! nothing means "a function of weight alone", which lets the engine collapse
//! the model into a table indexed by weight and never call it again. See
//! [`crate::term_kernel::Depends`].
//!
//! `basis_kind` is `0` for Pauli and `1` for Majorana. `words` holds two bits per
//! unit, interleaved; see [`crate::term_kernel::TermView`] for the layout. In the
//! batch form the terms are contiguous, `n_words_per_term` words each.
//!
//! **`words` is `NULL` for a plugin that did not declare `PROPAQ_DEPENDS_KEY`.**
//!
//! The batch entry point is an optional performance path. If present, the
//! damping pass calls it once per chunk instead of once per term, which lets
//! performance-sensitive plugin authors amortize the FFI boundary cost across
//! many terms.
//!
//! ## Safety contract
//! Loading and calling a native plugin is unsandboxed arbitrary code
//! execution. A key-reading plugin's functions are called concurrently from
//! arbitrary rayon worker threads with a shared `ctx` pointer. The plugin
//! author is responsible for making `ctx` safe under concurrent read access,
//! and for never panicking/unwinding/longjmp-ing across the FFI boundary.
//!
//!
use std::ffi::{c_void, CString};
use std::os::raw::c_char;

use libloading::{Library, Symbol};
use pyo3::prelude::*;

use crate::basis::BasisKind;
use crate::native_truncator::{basis_kind_from_u32, depends_from_bits};
use crate::term_kernel::{Depends, LayerContext, NoiseKernel, TermView};

/// The only ABI version.
pub const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type DependsFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type FactorFn =
    unsafe extern "C" fn(*mut c_void, u32, *const u64, usize, u32, u32, u32, u32) -> f64;
type FactorBatchFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const u64,
    usize,
    u32,
    *const u32,
    u32,
    u32,
    *mut f64,
    usize,
) -> i32;

/// Wraps the resolved plugin entry points and its `ctx` pointer so they
/// can cross into a rayon parallel closure.
#[derive(Clone, Copy)]
pub struct NativeNoiseHandle {
    ctx: *mut c_void,
    /// What the plugin declared it reads.
    depends: Depends,
    factor_fn: FactorFn,
    factor_batch_fn: Option<FactorBatchFn>,
}
unsafe impl Send for NativeNoiseHandle {}
unsafe impl Sync for NativeNoiseHandle {}

impl NativeNoiseHandle {
    /// What the plugin declared it reads.
    #[inline]
    pub fn depends(&self) -> Depends {
        self.depends
    }

    /// The key pointer to hand the plugin: real only when it declared it reads
    /// keys, `NULL` otherwise.
    #[inline]
    fn key_ptr(&self, words: &[u64]) -> (*const u64, usize) {
        if self.depends.key() {
            (words.as_ptr(), words.len())
        } else {
            (std::ptr::null(), 0)
        }
    }
}

impl NoiseKernel for NativeNoiseHandle {
    #[inline]
    fn depends(&self) -> Depends {
        self.depends
    }

    #[inline]
    fn factor(&self, term: TermView<'_>) -> f64 {
        let (words, n_words) = self.key_ptr(term.words);
        unsafe {
            (self.factor_fn)(
                self.ctx,
                term.basis_kind.as_u32(),
                words,
                n_words,
                term.n_units as u32,
                term.weight,
                term.layer.index,
                term.layer.total,
            )
        }
    }

    fn factor_batch(
        &self,
        basis_kind: BasisKind,
        words: &[u64],
        stride: usize,
        weights: &[u32],
        n_units: usize,
        layer: LayerContext,
        out: &mut [f64],
    ) {
        let n = weights.len();
        debug_assert_eq!(words.len(), n * stride);
        debug_assert_eq!(out.len(), n);
        if let Some(batch_fn) = self.factor_batch_fn {
            let (wptr, wstride) = if self.depends.key() {
                (words.as_ptr(), stride)
            } else {
                (std::ptr::null(), 0)
            };
            let rc = unsafe {
                batch_fn(
                    self.ctx,
                    basis_kind.as_u32(),
                    wptr,
                    wstride,
                    n_units as u32,
                    weights.as_ptr(),
                    layer.index,
                    layer.total,
                    out.as_mut_ptr(),
                    n,
                )
            };
            if rc == 0 {
                return;
            }
        }
        for (i, &weight) in weights.iter().enumerate() {
            out[i] = self.factor(TermView {
                basis_kind,
                words: &words[i * stride..(i + 1) * stride],
                n_units,
                weight,
                layer,
            });
        }
    }
}

/// Noise model backed by a dynamically loaded native (C/Rust/AOT-Julia)
/// plugin.
///
/// Arguments:
///     path: Filesystem path to the plugin shared library (.so/.dylib/.dll).
///     config: Optional JSON string passed once to the plugin's
///             `propaq_noise_create`, if it exports one.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
pub struct NativeNoiseModel {
    handle: NativeNoiseHandle,
    destroy_fn: Option<DestroyFn>,
    _lib: Library,
}

impl NativeNoiseModel {
    pub fn handle(&self) -> &NativeNoiseHandle {
        &self.handle
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl NativeNoiseModel {
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(path: String, config: Option<String>) -> PyResult<Self> {
        let lib = unsafe { Library::new(&path) }.map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!(
                "failed to load noise plugin '{path}': {e}"
            ))
        })?;

        let abi_version: Symbol<AbiVersionFn> = unsafe { lib.get(b"propaq_noise_abi_version\0") }
            .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' does not export propaq_noise_abi_version: {e}"
            ))
        })?;
        let version = unsafe { abi_version() };
        if version != PROPAQ_NOISE_ABI_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' targets ABI version {version}, expected {PROPAQ_NOISE_ABI_VERSION}"
            )));
        }

        let depends = match unsafe { lib.get::<DependsFn>(b"propaq_noise_depends\0") } {
            Ok(f) => depends_from_bits(unsafe { f() }, "noise", &path)?,
            Err(_) => Depends::NONE,
        };

        let factor: Symbol<FactorFn> =
            unsafe { lib.get(b"propaq_noise_factor\0") }.map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "noise plugin '{path}' does not export propaq_noise_factor: {e}"
                ))
            })?;
        let factor_fn = *factor;
        let factor_batch_fn = unsafe { lib.get::<FactorBatchFn>(b"propaq_noise_factor_batch\0") }
            .ok()
            .map(|s| *s);

        let create_fn: Option<Symbol<CreateFn>> = unsafe { lib.get(b"propaq_noise_create\0") }.ok();
        let destroy_fn: Option<Symbol<DestroyFn>> =
            unsafe { lib.get(b"propaq_noise_destroy\0") }.ok();
        if create_fn.is_some() != destroy_fn.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' must export both propaq_noise_create and propaq_noise_destroy, or neither"
            )));
        }

        let ctx = match &create_fn {
            Some(create) => {
                let config_c = match &config {
                    Some(s) => Some(
                        CString::new(s.as_str())
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
                    ),
                    None => None,
                };
                let config_ptr = config_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
                unsafe { create(config_ptr) }
            }
            None => std::ptr::null_mut(),
        };
        let destroy_fn = destroy_fn.map(|s| *s);

        Ok(NativeNoiseModel {
            handle: NativeNoiseHandle {
                ctx,
                depends,
                factor_fn,
                factor_batch_fn,
            },
            destroy_fn,
            _lib: lib,
        })
    }

    /// The ABI version the loaded plugin declared.
    #[getter(abi_version)]
    fn get_abi_version(&self) -> u32 {
        PROPAQ_NOISE_ABI_VERSION
    }

    /// The dependency bitmask the plugin declared: 1 = reads the term's key,
    /// 2 = reads the layer index. 0 means a function of weight alone.
    #[getter(depends)]
    fn get_depends(&self) -> u32 {
        self.handle.depends.bits()
    }

    /// Delegate to the plugin's `propaq_noise_factor`.
    ///
    /// Exposed so a plugin can be exercised from Python without running a
    /// circuit; propagation calls the same entry point directly from the pool.
    ///
    /// Arguments:
    ///     basis_kind: 0 for Pauli, 1 for Majorana.
    ///     words: The term's raw basis-string words, two bits per unit. Ignored
    ///         (and passed as NULL) unless the plugin declared it reads keys.
    ///     n_units: Qubits (Pauli) or modes (Majorana) of the register.
    ///     weight: The term's weight.
    ///     layer_index: Zero-based circuit layer.
    ///     n_layers: Layers in the circuit.
    #[pyo3(signature = (basis_kind, words, n_units, weight, layer_index=0, n_layers=0))]
    fn factor_term(
        &self,
        basis_kind: u32,
        words: Vec<u64>,
        n_units: usize,
        weight: u32,
        layer_index: u32,
        n_layers: u32,
    ) -> PyResult<f64> {
        Ok(self.handle.factor(TermView {
            basis_kind: basis_kind_from_u32(basis_kind)?,
            words: &words,
            n_units,
            weight,
            layer: LayerContext::new(layer_index, n_layers),
        }))
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeNoiseModel(<native plugin, depends={}>)",
            self.handle.depends.bits()
        )
    }
}

impl Drop for NativeNoiseModel {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy_fn {
            unsafe { destroy(self.handle.ctx) };
        }
    }
}
