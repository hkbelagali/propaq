//!
//! Dynamically loaded noise models!
//! Lets a user supply a noise model compiled from C, Rust, or (AOT-compiled)
//! Julia as a shared library.
//!
//! ## ABI contract, version 1
//! A plugin is a dylib exporting these fixed symbol names
//!
//! ```c
//! uint32_t propaq_noise_abi_version(void);                 // returns 1
//! void*    propaq_noise_create(const char* config_json);   // optional
//! void     propaq_noise_destroy(void* ctx);                // required iff create is present
//! double   propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes);
//! int32_t  propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
//!                                     const uint32_t* active_modes, double* out, size_t n); // optional
//! ```
//!
//! A v1 model is a function of weight alone, so the engine collapses it to one
//! table indexed by weight before propagation starts and never calls it again.
//!
//! ## ABI contract, version 2
//!
//! Version 2 handles noise models that need to see a term's entire symplectic representation, 
//! so they're called per-term rather than being compressed into an LUT.
//!
//! ```c
//! uint32_t propaq_noise_abi_version(void);                 // returns 2
//! void*    propaq_noise_create(const char* config_json);   // optional
//! void     propaq_noise_destroy(void* ctx);                // required iff create is present
//! double   propaq_noise_factor_v2(void* ctx, uint32_t basis_kind, const uint64_t* words,
//!                                 size_t n_words, uint32_t n_units, uint32_t weight);
//! int32_t  propaq_noise_factor_batch_v2(void* ctx, uint32_t basis_kind, const uint64_t* words,
//!                                       size_t n_words_per_term, uint32_t n_units,
//!                                       const uint32_t* weights, double* out,
//!                                       size_t n_terms);   // optional
//! ```
//!
//! `basis_kind` is `0` for Pauli and `1` for Majorana. `words` holds two bits per
//! unit, interleaved; see [`crate::term_kernel::TermView`] for the layout. In the
//! batch form the terms are contiguous, `n_words_per_term` words each.
//!
//! The batch entry points in both versions are an optional performance path. If
//! present, the damping pass calls one once per chunk instead of once per term,
//! which lets performance-sensitive plugin authors (C, Rust, or AOT-compiled
//! Julia via `PackageCompiler.create_library`) amortize the FFI boundary cost
//! across many terms. NOTE: Designing a batch path could incur race conditions 
//! for particular types of stateful noise models. This is something we plan
//! to work on more in the future. 
//!
//! ## Safety contract
//! Loading and calling a native plugin is unsandboxed arbitrary code
//! execution. The plugin's `damping_factor`/`damping_batch` functions
//! are called concurrently from arbitrary rayon worker threads with a
//! shared `ctx` pointer. The plugin author is responsible for making
//! `ctx` safe under concurrent read access, and for never
//! panicking/unwinding/longjmp-ing across the FFI boundary.
//!
use std::ffi::{c_void, CString};
use std::os::raw::c_char;

use libloading::{Library, Symbol};
use pyo3::prelude::*;

use crate::basis::BasisKind;
use crate::native_truncator::basis_kind_from_u32;
use crate::term_kernel::{NoiseKernel, TermView};

/// The original weight-only ABI.
pub const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

/// The basis-string-aware ABI.
pub const PROPAQ_NOISE_ABI_VERSION_V2: u32 = 2;

type AbiVersionFn = unsafe extern "C" fn() -> u32;
type CreateFn = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type DestroyFn = unsafe extern "C" fn(*mut c_void);
type DampingFactorFn = unsafe extern "C" fn(*mut c_void, u32, u32) -> f64;
type DampingBatchFn =
    unsafe extern "C" fn(*mut c_void, *const u32, *const u32, *mut f64, usize) -> i32;
type FactorV2Fn = unsafe extern "C" fn(*mut c_void, u32, *const u64, usize, u32, u32) -> f64;
type FactorBatchV2Fn = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const u64,
    usize,
    u32,
    *const u32,
    *mut f64,
    usize,
) -> i32;

/// Wraps the resolved plugin entry points and its `ctx` pointer so they
/// can cross into a rayon parallel closure. 
#[derive(Clone, Copy)]
pub struct NativeNoiseHandle {
    ctx: *mut c_void,
    /// Which ABI the plugin declared,
    abi_version: u32,
    damping_factor_fn: Option<DampingFactorFn>,
    damping_batch_fn: Option<DampingBatchFn>,
    factor_v2_fn: Option<FactorV2Fn>,
    factor_batch_v2_fn: Option<FactorBatchV2Fn>,
}
unsafe impl Send for NativeNoiseHandle {}
unsafe impl Sync for NativeNoiseHandle {}

impl NativeNoiseHandle {
    /// The ABI version the loaded plugin declared.
    #[inline]
    pub fn abi_version(&self) -> u32 {
        self.abi_version
    }

    /// True when this plugin reads term keys rather than weights.
    #[inline]
    pub fn is_term_aware(&self) -> bool {
        self.factor_v2_fn.is_some()
    }

    #[inline]
    pub fn damping_factor(&self, term_weight: u32, active_modes: u32) -> f64 {
        let f = self
            .damping_factor_fn
            .expect("a v1 noise plugin exports propaq_noise_damping_factor");
        unsafe { f(self.ctx, term_weight, active_modes) }
    }

    #[inline]
    pub fn try_damping_batch(
        &self,
        weights: &[u32],
        active_modes: &[u32],
        out: &mut [f64],
    ) -> bool {
        let Some(batch_fn) = self.damping_batch_fn else {
            return false;
        };
        let n = weights.len();
        debug_assert_eq!(active_modes.len(), n);
        debug_assert_eq!(out.len(), n);
        let rc = unsafe {
            batch_fn(
                self.ctx,
                weights.as_ptr(),
                active_modes.as_ptr(),
                out.as_mut_ptr(),
                n,
            )
        };
        rc == 0
    }
}

/// The v2 factor, as the engine's term-aware noise kernel.
impl NoiseKernel for NativeNoiseHandle {
    #[inline]
    fn factor(&self, term: TermView<'_>) -> f64 {
        let f = self
            .factor_v2_fn
            .expect("a v2 noise plugin exports propaq_noise_factor_v2");
        unsafe {
            f(
                self.ctx,
                term.basis_kind.as_u32(),
                term.words.as_ptr(),
                term.words.len(),
                term.n_units as u32,
                term.weight,
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
        out: &mut [f64],
    ) {
        let n = weights.len();
        debug_assert_eq!(words.len(), n * stride);
        debug_assert_eq!(out.len(), n);
        if let Some(batch_fn) = self.factor_batch_v2_fn {
            let rc = unsafe {
                batch_fn(
                    self.ctx,
                    basis_kind.as_u32(),
                    words.as_ptr(),
                    stride,
                    n_units as u32,
                    weights.as_ptr(),
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

        if version != PROPAQ_NOISE_ABI_VERSION && version != PROPAQ_NOISE_ABI_VERSION_V2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "noise plugin '{path}' targets ABI version {version}, expected {PROPAQ_NOISE_ABI_VERSION} or {PROPAQ_NOISE_ABI_VERSION_V2}"
            )));
        }
        let is_v2 = version == PROPAQ_NOISE_ABI_VERSION_V2;

        let mut damping_factor_fn: Option<DampingFactorFn> = None;
        let mut damping_batch_fn: Option<DampingBatchFn> = None;
        let mut factor_v2_fn: Option<FactorV2Fn> = None;
        let mut factor_batch_v2_fn: Option<FactorBatchV2Fn> = None;
        if is_v2 {
            let f: Symbol<FactorV2Fn> = unsafe { lib.get(b"propaq_noise_factor_v2\0") }
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "noise plugin '{path}' declares ABI version 2 but does not export propaq_noise_factor_v2: {e}"
                    ))
                })?;
            factor_v2_fn = Some(*f);
            factor_batch_v2_fn =
                unsafe { lib.get::<FactorBatchV2Fn>(b"propaq_noise_factor_batch_v2\0") }
                    .ok()
                    .map(|s| *s);
        } else {
            let f: Symbol<DampingFactorFn> = unsafe { lib.get(b"propaq_noise_damping_factor\0") }
                .map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "noise plugin '{path}' does not export propaq_noise_damping_factor: {e}"
                ))
            })?;
            damping_factor_fn = Some(*f);
            damping_batch_fn =
                unsafe { lib.get::<DampingBatchFn>(b"propaq_noise_damping_batch\0") }
                    .ok()
                    .map(|s| *s);
        }

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
                abi_version: version,
                damping_factor_fn,
                damping_batch_fn,
                factor_v2_fn,
                factor_batch_v2_fn,
            },
            destroy_fn,
            _lib: lib,
        })
    }

    /// The ABI version the loaded plugin declared, 1 or 2.
    #[getter(abi_version)]
    fn get_abi_version(&self) -> u32 {
        self.handle.abi_version()
    }

    /// Delegate to the plugin's `propaq_noise_damping_factor`. ABI v1 plugins only.
    fn damping_factor(&self, term_weight: u32, active_modes: u32) -> PyResult<f64> {
        if self.handle.is_term_aware() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this plugin targets ABI version 2; call factor_term instead",
            ));
        }
        Ok(self.handle.damping_factor(term_weight, active_modes))
    }

    /// Delegate to the plugin's `propaq_noise_factor_v2`. ABI v2 plugins only.
    ///
    /// Exposed so a plugin can be exercised from Python without running a
    /// circuit; propagation calls the same entry point directly from the pool.
    ///
    /// Arguments:
    ///     basis_kind: 0 for Pauli, 1 for Majorana.
    ///     words: The term's raw basis-string words, two bits per unit.
    ///     n_units: Qubits (Pauli) or modes (Majorana) of the register.
    ///     weight: The term's weight.
    fn factor_term(
        &self,
        basis_kind: u32,
        words: Vec<u64>,
        n_units: usize,
        weight: u32,
    ) -> PyResult<f64> {
        if !self.handle.is_term_aware() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "this plugin targets ABI version 1; call damping_factor instead",
            ));
        }
        Ok(self.handle.factor(TermView {
            basis_kind: basis_kind_from_u32(basis_kind)?,
            words: &words,
            n_units,
            weight,
        }))
    }
}

impl Drop for NativeNoiseModel {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy_fn {
            unsafe { destroy(self.handle.ctx) };
        }
    }
}
