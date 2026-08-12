//!
//! Dynamically loaded truncation policies!
//! Lets a user supply a per-term keep/discard predicate
//! compiled from C, Rust, or (AOT-compiled) Julia as a
//! shared library.
//!
//! ## ABI contract
//!
//! One entry point, with the plugin declaring what it reads through
//! `propaq_truncator_depends` — see [`crate::term_kernel::Depends`] and the
//! noise counterpart in [`crate::native_noise`].
//!
//! ```c
//! uint32_t propaq_truncator_abi_version(void);                 // required, must equal 1
//! uint32_t propaq_truncator_depends(void);                     // optional; absent => 0
//! void*    propaq_truncator_create(const char* config_json);   // optional
//! void     propaq_truncator_destroy(void* ctx);                // required iff create is present
//!
//! int32_t  propaq_truncator_keep(void* ctx, uint32_t basis_kind,
//!                                const uint64_t* words, size_t n_words,
//!                                uint32_t n_units, uint32_t weight,
//!                                double coeff_magnitude,
//!                                uint32_t layer_index, uint32_t n_layers);
//!                                                              // required; nonzero = keep
//!
//! int32_t  propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind,
//!                                      const uint64_t* words, size_t n_words_per_term,
//!                                      uint32_t n_units, const uint32_t* weights,
//!                                      const double* coeff_magnitudes,
//!                                      uint32_t layer_index, uint32_t n_layers,
//!                                      uint8_t* out_keep, size_t n_terms);
//!                                                              // optional; returns 0 on success
//! ```
//!
//! **`words` is `NULL` for a plugin that did not declare `PROPAQ_DEPENDS_KEY`.**
//!
//! The batch form is only reachable from the reclaim sweep that follows noise.
//! An emit-time decision is taken inside the branching loop, one child at a
//! time, so there is nothing to batch there and the scalar entry point carries
//! it.
//!
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::{Library, Symbol};
use pyo3::prelude::*;

use crate::basis::BasisKind;
use crate::term_kernel::{Depends, LayerContext, TermView, TruncationKernel};

/// The only ABI version.
pub const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type DependsFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type KeepFn =
    unsafe extern "C" fn(*mut c_void, u32, *const u64, usize, u32, u32, f64, u32, u32) -> i32;
type KeepBatchFn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const u64,
    usize,
    u32,
    *const u32,
    *const f64,
    u32,
    u32,
    *mut u8,
    usize,
) -> i32;

struct NativeTruncatorInner {
    ctx: *mut c_void,
    /// What the plugin declared it reads.
    depends: Depends,
    keep_fn: KeepFn,
    keep_batch_fn: Option<KeepBatchFn>,
    destroy_fn: Option<DestroyFn>,
    _lib: Library,
}
unsafe impl Send for NativeTruncatorInner {}
unsafe impl Sync for NativeTruncatorInner {}

impl Drop for NativeTruncatorInner {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy_fn {
            unsafe { destroy(self.ctx) };
        }
    }
}

/// Truncation policy backed by a dynamically loaded native (C/Rust/AOT-Julia)
/// plugin.
///
/// Arguments:
///     path: Filesystem path to the plugin shared library (.so/.dylib/.dll).
///     config: Optional JSON string passed once to the plugin's
///             `propaq_truncator_create`, if it exports one.
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct NativeTruncator {
    inner: Arc<NativeTruncatorInner>,
}

impl NativeTruncator {
    /// What the plugin declared it reads.
    #[inline]
    pub fn depends(&self) -> Depends {
        self.inner.depends
    }

    /// This policy as the engine's term kernel. Every plugin is one now; the
    /// declaration decides what it is handed, not which trait it implements.
    pub fn as_term_kernel(&self) -> Arc<dyn TruncationKernel> {
        Arc::new(self.clone()) as Arc<dyn TruncationKernel>
    }

    /// The key pointer to hand the plugin: real only when it declared it reads
    /// keys, `NULL` otherwise.
    #[inline]
    fn key_ptr(&self, words: &[u64]) -> (*const u64, usize) {
        if self.inner.depends.key() {
            (words.as_ptr(), words.len())
        } else {
            (std::ptr::null(), 0)
        }
    }
}

impl TruncationKernel for NativeTruncator {
    #[inline]
    fn depends(&self) -> Depends {
        self.inner.depends
    }

    #[inline]
    fn keep(&self, term: TermView<'_>, coeff_magnitude: f64) -> bool {
        let (words, n_words) = self.key_ptr(term.words);
        unsafe {
            (self.inner.keep_fn)(
                self.inner.ctx,
                term.basis_kind.as_u32(),
                words,
                n_words,
                term.n_units as u32,
                term.weight,
                coeff_magnitude,
                term.layer.index,
                term.layer.total,
            ) != 0
        }
    }

    fn keep_batch(
        &self,
        basis_kind: BasisKind,
        words: &[u64],
        stride: usize,
        weights: &[u32],
        n_units: usize,
        coeff_magnitudes: &[f64],
        layer: LayerContext,
        out: &mut [u8],
    ) -> bool {
        let Some(batch_fn) = self.inner.keep_batch_fn else {
            return false;
        };
        let n = weights.len();
        debug_assert_eq!(words.len(), n * stride);
        debug_assert_eq!(coeff_magnitudes.len(), n);
        debug_assert_eq!(out.len(), n);
        let (wptr, wstride) = if self.inner.depends.key() {
            (words.as_ptr(), stride)
        } else {
            (std::ptr::null(), 0)
        };
        let rc = unsafe {
            batch_fn(
                self.inner.ctx,
                basis_kind.as_u32(),
                wptr,
                wstride,
                n_units as u32,
                weights.as_ptr(),
                coeff_magnitudes.as_ptr(),
                layer.index,
                layer.total,
                out.as_mut_ptr(),
                n,
            )
        };
        rc == 0
    }
}

#[pyo3_stub_gen::derive::gen_stub_pymethods]
#[pymethods]
impl NativeTruncator {
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(path: String, config: Option<String>) -> PyResult<Self> {
        let lib = unsafe { Library::new(&path) }.map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!(
                "failed to load truncator plugin '{path}': {e}"
            ))
        })?;

        let abi_version: Symbol<AbiVersionFn> =
            unsafe { lib.get(b"propaq_truncator_abi_version\0") }.map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "truncator plugin '{path}' does not export propaq_truncator_abi_version: {e}"
                ))
            })?;
        let version = unsafe { abi_version() };
        if version != PROPAQ_TRUNCATOR_ABI_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' targets ABI version {version}, expected {PROPAQ_TRUNCATOR_ABI_VERSION}"
            )));
        }

        let depends = match unsafe { lib.get::<DependsFn>(b"propaq_truncator_depends\0") } {
            Ok(f) => depends_from_bits(unsafe { f() }, "truncator", &path)?,
            Err(_) => Depends::NONE,
        };

        let keep: Symbol<KeepFn> = unsafe { lib.get(b"propaq_truncator_keep\0") }.map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' does not export propaq_truncator_keep: {e}"
            ))
        })?;
        let keep_fn = *keep;
        let keep_batch_fn = unsafe { lib.get::<KeepBatchFn>(b"propaq_truncator_keep_batch\0") }
            .ok()
            .map(|s| *s);

        let create_fn: Option<Symbol<CreateFn>> =
            unsafe { lib.get(b"propaq_truncator_create\0") }.ok();
        let destroy_fn: Option<Symbol<DestroyFn>> =
            unsafe { lib.get(b"propaq_truncator_destroy\0") }.ok();
        if create_fn.is_some() != destroy_fn.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' must export both propaq_truncator_create and propaq_truncator_destroy, or neither"
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

        Ok(NativeTruncator {
            inner: Arc::new(NativeTruncatorInner {
                ctx,
                depends,
                keep_fn,
                keep_batch_fn,
                destroy_fn,
                _lib: lib,
            }),
        })
    }

    /// The ABI version the loaded plugin declared.
    #[getter(abi_version)]
    fn get_abi_version(&self) -> u32 {
        PROPAQ_TRUNCATOR_ABI_VERSION
    }

    /// The dependency bitmask the plugin declared: 1 = reads the term's key,
    /// 2 = reads the layer index. 0 means weight and magnitude alone.
    #[getter(depends)]
    fn get_depends(&self) -> u32 {
        self.inner.depends.bits()
    }

    /// Delegate to the plugin's `propaq_truncator_keep`.
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
    ///     coeff_magnitude: The term's coefficient magnitude.
    ///     layer_index: Zero-based circuit layer.
    ///     n_layers: Layers in the circuit.
    #[pyo3(signature = (basis_kind, words, n_units, weight, coeff_magnitude, layer_index=0, n_layers=0))]
    #[allow(clippy::too_many_arguments)]
    fn keep_term(
        &self,
        basis_kind: u32,
        words: Vec<u64>,
        n_units: usize,
        weight: u32,
        coeff_magnitude: f64,
        layer_index: u32,
        n_layers: u32,
    ) -> PyResult<bool> {
        Ok(TruncationKernel::keep(
            self,
            TermView {
                basis_kind: basis_kind_from_u32(basis_kind)?,
                words: &words,
                n_units,
                weight,
                layer: LayerContext::new(layer_index, n_layers),
            },
            coeff_magnitude,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeTruncator(<native plugin, depends={}>)",
            self.inner.depends.bits()
        )
    }
}

/// The ABI's basis encoding, checked.
pub(crate) fn basis_kind_from_u32(kind: u32) -> PyResult<BasisKind> {
    match kind {
        0 => Ok(BasisKind::Pauli),
        1 => Ok(BasisKind::Majorana),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "basis_kind must be 0 (Pauli) or 1 (Majorana), got {other}"
        ))),
    }
}

/// The ABI's dependency bitmask, checked.
pub(crate) fn depends_from_bits(bits: u32, kind: &str, path: &str) -> PyResult<Depends> {
    Depends::from_bits(bits).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "{kind} plugin '{path}' declares unknown dependency bits {bits:#x} \
             (this build understands {:#x}); it was likely built against a newer propaq",
            Depends::KNOWN
        ))
    })
}
