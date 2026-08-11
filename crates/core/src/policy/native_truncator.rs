//!
//! Dynamically loaded truncation policies!
//! Lets a user supply a per-term keep/discard predicate
//! compiled from C, Rust, or (AOT-compiled) Julia as a
//! shared library.
//!
//! ## ABI contract, version 1
//!
//! ```c
//! uint32_t propaq_truncator_abi_version(void);                 // returns 1
//! void*    propaq_truncator_create(const char* config_json);   // optional
//! void     propaq_truncator_destroy(void* ctx);                // required iff create is present
//! int32_t  propaq_truncator_keep(void* ctx, uint32_t term_weight, double coeff_magnitude,
//!                                 uint32_t active_modes);       // required; nonzero = keep
//! int32_t  propaq_truncator_keep_batch(void* ctx, const uint32_t* term_weights,
//!                                       const double* coeff_magnitudes, const uint32_t* active_modes,
//!                                       uint8_t* out_keep, size_t n);  // optional; returns 0 on success
//! ```
//!
//! ## ABI contract, version 2
//!
//! Similar to the noise models, this provides access to an entire term's bitmask, 
//! allowing for structure-aware truncation decisions. This still maintains backwards 
//! compatibility with the v1 ABI, so a plugin can implement either one.
//!
//! ```c
//! uint32_t propaq_truncator_abi_version(void);                 // returns 2
//! void*    propaq_truncator_create(const char* config_json);   // optional
//! void     propaq_truncator_destroy(void* ctx);                // required iff create is present
//! int32_t  propaq_truncator_keep_v2(void* ctx, uint32_t basis_kind, const uint64_t* words,
//!                                    size_t n_words, uint32_t n_units, uint32_t weight,
//!                                    double coeff_magnitude);   // required; nonzero = keep
//! int32_t  propaq_truncator_keep_batch_v2(void* ctx, uint32_t basis_kind, const uint64_t* words,
//!                                          size_t n_words_per_term, uint32_t n_units,
//!                                          const uint32_t* weights, const double* coeff_magnitudes,
//!                                          uint8_t* out_keep, size_t n_terms);
//!                                                               // optional; returns 0 on success
//! ```
//!
//! `basis_kind` is `0` for Pauli and `1` for Majorana. `words` holds two bits per
//! unit, interleaved; see [`crate::term_kernel::TermView`] for the layout. In the
//! batch form the terms are contiguous, `n_words_per_term` words each.
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
use crate::term_kernel::{TermView, TruncationKernel};

/// The original weight/magnitude ABI.
pub const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

/// The basis-string-aware ABI.
pub const PROPAQ_TRUNCATOR_ABI_VERSION_V2: u32 = 2;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type KeepFn = unsafe extern "C" fn(*mut c_void, u32, f64, u32) -> i32;
type KeepBatchFn =
    unsafe extern "C" fn(*mut c_void, *const u32, *const f64, *const u32, *mut u8, usize) -> i32;
type KeepV2Fn = unsafe extern "C" fn(*mut c_void, u32, *const u64, usize, u32, u32, f64) -> i32;
type KeepBatchV2Fn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const u64,
    usize,
    u32,
    *const u32,
    *const f64,
    *mut u8,
    usize,
) -> i32;

struct NativeTruncatorInner {
    ctx: *mut c_void,
    /// Which ABI the plugin declared
    abi_version: u32,
    keep_fn: Option<KeepFn>,
    keep_batch_fn: Option<KeepBatchFn>,
    keep_v2_fn: Option<KeepV2Fn>,
    keep_batch_v2_fn: Option<KeepBatchV2Fn>,
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
    /// The ABI version the loaded plugin declared.
    #[inline]
    pub fn abi_version(&self) -> u32 {
        self.inner.abi_version
    }

    pub fn as_term_kernel(&self) -> Option<Arc<dyn TruncationKernel>> {
        self.inner
            .keep_v2_fn
            .map(|_| Arc::new(self.clone()) as Arc<dyn TruncationKernel>)
    }

    #[inline]
    pub fn keep(&self, term_weight: u32, coeff_magnitude: f64, active_modes: u32) -> bool {
        let keep_fn = self
            .inner
            .keep_fn
            .expect("a v1 truncator plugin exports propaq_truncator_keep");
        unsafe { keep_fn(self.inner.ctx, term_weight, coeff_magnitude, active_modes) != 0 }
    }

    #[inline]
    pub fn try_keep_batch(
        &self,
        weights: &[u32],
        coeff_magnitudes: &[f64],
        active_modes: &[u32],
        out: &mut [u8],
    ) -> bool {
        let Some(batch_fn) = self.inner.keep_batch_fn else {
            return false;
        };
        let n = weights.len();
        debug_assert_eq!(coeff_magnitudes.len(), n);
        debug_assert_eq!(active_modes.len(), n);
        debug_assert_eq!(out.len(), n);
        let rc = unsafe {
            batch_fn(
                self.inner.ctx,
                weights.as_ptr(),
                coeff_magnitudes.as_ptr(),
                active_modes.as_ptr(),
                out.as_mut_ptr(),
                n,
            )
        };
        rc == 0
    }
}

impl TruncationKernel for NativeTruncator {
    #[inline]
    fn keep(&self, term: TermView<'_>, coeff_magnitude: f64) -> bool {
        let keep_fn = self
            .inner
            .keep_v2_fn
            .expect("a v2 truncator plugin exports propaq_truncator_keep_v2");
        unsafe {
            keep_fn(
                self.inner.ctx,
                term.basis_kind.as_u32(),
                term.words.as_ptr(),
                term.words.len(),
                term.n_units as u32,
                term.weight,
                coeff_magnitude,
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
        out: &mut [u8],
    ) -> bool {
        let Some(batch_fn) = self.inner.keep_batch_v2_fn else {
            return false;
        };
        let n = weights.len();
        debug_assert_eq!(words.len(), n * stride);
        debug_assert_eq!(coeff_magnitudes.len(), n);
        debug_assert_eq!(out.len(), n);
        let rc = unsafe {
            batch_fn(
                self.inner.ctx,
                basis_kind.as_u32(),
                words.as_ptr(),
                stride,
                n_units as u32,
                weights.as_ptr(),
                coeff_magnitudes.as_ptr(),
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

        if version != PROPAQ_TRUNCATOR_ABI_VERSION && version != PROPAQ_TRUNCATOR_ABI_VERSION_V2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' targets ABI version {version}, expected {PROPAQ_TRUNCATOR_ABI_VERSION} or {PROPAQ_TRUNCATOR_ABI_VERSION_V2}"
            )));
        }
        let is_v2 = version == PROPAQ_TRUNCATOR_ABI_VERSION_V2;

        let mut keep_fn: Option<KeepFn> = None;
        let mut keep_batch_fn: Option<KeepBatchFn> = None;
        let mut keep_v2_fn: Option<KeepV2Fn> = None;
        let mut keep_batch_v2_fn: Option<KeepBatchV2Fn> = None;
        if is_v2 {
            let f: Symbol<KeepV2Fn> = unsafe { lib.get(b"propaq_truncator_keep_v2\0") }
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "truncator plugin '{path}' declares ABI version 2 but does not export propaq_truncator_keep_v2: {e}"
                    ))
                })?;
            keep_v2_fn = Some(*f);
            keep_batch_v2_fn =
                unsafe { lib.get::<KeepBatchV2Fn>(b"propaq_truncator_keep_batch_v2\0") }
                    .ok()
                    .map(|s| *s);
        } else {
            let f: Symbol<KeepFn> =
                unsafe { lib.get(b"propaq_truncator_keep\0") }.map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "truncator plugin '{path}' does not export propaq_truncator_keep: {e}"
                    ))
                })?;
            keep_fn = Some(*f);
            keep_batch_fn = unsafe { lib.get::<KeepBatchFn>(b"propaq_truncator_keep_batch\0") }
                .ok()
                .map(|s| *s);
        }

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
                abi_version: version,
                keep_fn,
                keep_batch_fn,
                keep_v2_fn,
                keep_batch_v2_fn,
                destroy_fn,
                _lib: lib,
            }),
        })
    }

    /// The ABI version the loaded plugin declared, 1 or 2.
    #[getter(abi_version)]
    fn get_abi_version(&self) -> u32 {
        self.inner.abi_version
    }

    fn keep_term(
        &self,
        term_weight: u32,
        coeff_magnitude: f64,
        active_modes: u32,
    ) -> PyResult<bool> {
        if self.inner.keep_fn.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this plugin targets ABI version 2; call keep_term_v2 instead",
            ));
        }
        Ok(self.keep(term_weight, coeff_magnitude, active_modes))
    }

    /// Delegate to the plugin's `propaq_truncator_keep_v2`. ABI v2 plugins only.
    ///
    /// Exposed so a plugin can be exercised from Python without running a
    /// circuit; propagation calls the same entry point directly from the pool.
    ///
    /// Arguments:
    ///     basis_kind: 0 for Pauli, 1 for Majorana.
    ///     words: The term's raw basis-string words, two bits per unit.
    ///     n_units: Qubits (Pauli) or modes (Majorana) of the register.
    ///     weight: The term's weight.
    ///     coeff_magnitude: The term's coefficient magnitude.
    fn keep_term_v2(
        &self,
        basis_kind: u32,
        words: Vec<u64>,
        n_units: usize,
        weight: u32,
        coeff_magnitude: f64,
    ) -> PyResult<bool> {
        if self.inner.keep_v2_fn.is_none() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this plugin targets ABI version 1; call keep_term instead",
            ));
        }
        Ok(TruncationKernel::keep(
            self,
            TermView {
                basis_kind: basis_kind_from_u32(basis_kind)?,
                words: &words,
                n_units,
                weight,
            },
            coeff_magnitude,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeTruncator(<native plugin, abi v{}>)",
            self.inner.abi_version
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
