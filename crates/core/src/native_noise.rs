///
/// Native (dynamically loaded) noise models!
/// Lets a user supply a noise model compiled from C, Rust, or (AOT-compiled) 
/// Julia as a shared library.
///
/// ## ABI contract
/// A plugin is a dylib exporting these fixed symbol names
///
/// ```c
/// uint32_t propaq_noise_abi_version(void);
/// void*    propaq_noise_create(const char* config_json);   // optional
/// void     propaq_noise_destroy(void* ctx);                // required iff create is present
/// double   propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes);
/// int32_t  propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
///                                     const uint32_t* active_modes, double* out, size_t n); // optional
/// ```
///
/// `propaq_noise_damping_batch` is an optional performance path: if
/// present, the hot loop calls it once per rayon chunk instead of once
/// per term, which lets performance-sensitive plugin authors (C, Rust,
/// or AOT-compiled Julia via `PackageCompiler.compile_shlib`) amortize
/// the FFI boundary cost across many terms.
///
/// ## Safety contract
/// Loading and calling a native plugin is unsandboxed arbitrary code
/// execution. The plugin's `damping_factor`/`damping_batch` functions
/// are called concurrently from arbitrary rayon worker threads with a
/// shared `ctx` pointer. The plugin author is responsible for making
/// `ctx` safe under concurrent read access, and for never
/// panicking/unwinding/longjmp-ing across the FFI boundary.
///
use std::ffi::{c_void, CString};
use std::os::raw::c_char;

use libloading::{Library, Symbol};
use pyo3::prelude::*;

/// Bump on any ABI signature change; checked against the plugin's
/// exported `propaq_noise_abi_version` at load time.
pub const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type DampingFactorFn = unsafe extern "C" fn(*mut c_void, u32, u32) -> f64;
type DampingBatchFn =
    unsafe extern "C" fn(*mut c_void, *const u32, *const u32, *mut f64, usize) -> i32;

/// Wraps the resolved plugin entry points and its `ctx` pointer so they
/// can cross into a rayon parallel closure. Sound only under the
/// safety contract documented on the module: the plugin must tolerate
/// concurrent calls sharing `ctx`. See `crate::soa::kernels::SendPtr`
/// for the same pattern used elsewhere in this codebase.
#[derive(Clone, Copy)]
pub struct NativeNoiseHandle {
    ctx: *mut c_void,
    damping_factor_fn: DampingFactorFn,
    damping_batch_fn: Option<DampingBatchFn>,
}
unsafe impl Send for NativeNoiseHandle {}
unsafe impl Sync for NativeNoiseHandle {}

impl NativeNoiseHandle {
    #[inline]
    pub fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        unsafe { (self.damping_factor_fn)(self.ctx, term_weight, active_modes) }
    }

    #[inline]
    pub fn try_damping_batch(&self, weights: &[u32], active_modes: &[u32], out: &mut [f64]) -> bool {
        let Some(batch_fn) = self.damping_batch_fn else { return false };
        let n = weights.len();
        debug_assert_eq!(active_modes.len(), n);
        debug_assert_eq!(out.len(), n);
        let rc = unsafe {
            batch_fn(self.ctx, weights.as_ptr(), active_modes.as_ptr(), out.as_mut_ptr(), n)
        };
        rc == 0
    }
}

/// Noise model backed by a dynamically loaded native (C/Rust/AOT-Julia)
/// plugin.
///
/// Arguments:
///     path: Filesystem path to the plugin shared library (.so/.dylib/.dll).
///     config: Optional JSON string passed once to the plugin's
///             `propaq_noise_create`, if it exports one.
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

#[pymethods]
impl NativeNoiseModel {
    #[new]
    #[pyo3(signature = (path, config=None))]
    fn new(path: String, config: Option<String>) -> PyResult<Self> {
        let lib = unsafe { Library::new(&path) }
            .map_err(|e| pyo3::exceptions::PyOSError::new_err(format!("failed to load noise plugin '{path}': {e}")))?;

        let abi_version: Symbol<AbiVersionFn> = unsafe { lib.get(b"propaq_noise_abi_version\0") }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' does not export propaq_noise_abi_version: {e}"
            )))?;
        let version = unsafe { abi_version() };
        if version != PROPAQ_NOISE_ABI_VERSION {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' targets ABI version {version}, expected {PROPAQ_NOISE_ABI_VERSION}"
            )));
        }

        let damping_factor_fn: Symbol<DampingFactorFn> =
            unsafe { lib.get(b"propaq_noise_damping_factor\0") }.map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "noise plugin '{path}' does not export propaq_noise_damping_factor: {e}"
                ))
            })?;
        let damping_factor_fn = *damping_factor_fn;

        let damping_batch_fn: Option<DampingBatchFn> =
            unsafe { lib.get::<DampingBatchFn>(b"propaq_noise_damping_batch\0") }
                .ok()
                .map(|s| *s);

        let create_fn: Option<Symbol<CreateFn>> =
            unsafe { lib.get(b"propaq_noise_create\0") }.ok();
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
            handle: NativeNoiseHandle { ctx, damping_factor_fn, damping_batch_fn },
            destroy_fn,
            _lib: lib,
        })
    }

    /// Delegate to the plugin's `propaq_noise_damping_factor`.
    fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        self.handle.damping_factor(term_weight, active_modes)
    }
}

impl Drop for NativeNoiseModel {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy_fn {
            unsafe { destroy(self.handle.ctx) };
        }
    }
}
