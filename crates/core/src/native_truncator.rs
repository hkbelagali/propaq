///
/// Native (dynamically loaded) truncation policies!
/// Lets a user supply a per-term keep/discard predicate 
/// compiled from C, Rust, or (AOT-compiled) Julia as a 
/// shared library.
///
/// ## ABI contract
///
/// ```c
/// uint32_t propaq_truncator_abi_version(void);
/// void*    propaq_truncator_create(const char* config_json);   // optional
/// void     propaq_truncator_destroy(void* ctx);                // required iff create is present
/// int32_t  propaq_truncator_keep(void* ctx, uint32_t term_weight, double coeff_magnitude,
///                                 uint32_t active_modes);       // required; nonzero = keep
/// int32_t  propaq_truncator_keep_batch(void* ctx, const uint32_t* term_weights,
///                                       const double* coeff_magnitudes, const uint32_t* active_modes,
///                                       uint8_t* out_keep, size_t n);  // optional; returns 0 on success
/// ```
///
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::sync::Arc;

use libloading::{Library, Symbol};
use pyo3::prelude::*;

/// Bump on any ABI signature change; checked against the plugin's
/// exported `propaq_truncator_abi_version` at load time.
pub const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type KeepFn = unsafe extern "C" fn(*mut c_void, u32, f64, u32) -> i32;
type KeepBatchFn =
    unsafe extern "C" fn(*mut c_void, *const u32, *const f64, *const u32, *mut u8, usize) -> i32;

struct NativeTruncatorInner {
    ctx: *mut c_void,
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
#[pyclass(subclass, module = "propaq._rust_core")]
#[derive(Clone)]
pub struct NativeTruncator {
    inner: Arc<NativeTruncatorInner>,
}

impl NativeTruncator {
    #[inline]
    pub fn keep(&self, term_weight: u32, coeff_magnitude: f64, active_modes: u32) -> bool {
        unsafe { (self.inner.keep_fn)(self.inner.ctx, term_weight, coeff_magnitude, active_modes) != 0 }
    }

    #[inline]
    pub fn try_keep_batch(
        &self,
        weights: &[u32],
        coeff_magnitudes: &[f64],
        active_modes: &[u32],
        out: &mut [u8],
    ) -> bool {
        let Some(batch_fn) = self.inner.keep_batch_fn else { return false };
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

#[pymethods]
impl NativeTruncator {
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(path: String, config: Option<String>) -> PyResult<Self> {
        let lib = unsafe { Library::new(&path) }.map_err(|e| {
            pyo3::exceptions::PyOSError::new_err(format!("failed to load truncator plugin '{path}': {e}"))
        })?;

        let abi_version: Symbol<AbiVersionFn> = unsafe { lib.get(b"propaq_truncator_abi_version\0") }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' does not export propaq_truncator_abi_version: {e}"
            )))?;
        let version = unsafe { abi_version() };
        if version != PROPAQ_TRUNCATOR_ABI_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' targets ABI version {version}, expected {PROPAQ_TRUNCATOR_ABI_VERSION}"
            )));
        }

        let keep_fn: Symbol<KeepFn> = unsafe { lib.get(b"propaq_truncator_keep\0") }.map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "truncator plugin '{path}' does not export propaq_truncator_keep: {e}"
            ))
        })?;
        let keep_fn = *keep_fn;

        let keep_batch_fn: Option<KeepBatchFn> =
            unsafe { lib.get::<KeepBatchFn>(b"propaq_truncator_keep_batch\0") }.ok().map(|s| *s);

        let create_fn: Option<Symbol<CreateFn>> = unsafe { lib.get(b"propaq_truncator_create\0") }.ok();
        let destroy_fn: Option<Symbol<DestroyFn>> = unsafe { lib.get(b"propaq_truncator_destroy\0") }.ok();
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
            inner: Arc::new(NativeTruncatorInner { ctx, keep_fn, keep_batch_fn, destroy_fn, _lib: lib }),
        })
    }

    /// Delegate to the plugin's `propaq_truncator_keep`.
    fn keep_term(&self, term_weight: u32, coeff_magnitude: f64, active_modes: u32) -> bool {
        self.keep(term_weight, coeff_magnitude, active_modes)
    }

    fn __repr__(&self) -> String {
        "NativeTruncator(<native plugin>)".to_string()
    }
}
