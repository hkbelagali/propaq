/// Example propaq native noise plugin, implemented in Rust.
///
/// A demonstration of *stateful ctx*: damping grows with the number of calls
/// already made through this ctx, rather than staying fixed like
/// UniformNoiseModel.
///
///   damping_factor = exp(-damping * weight * (1 + drift_rate * call_index))
///
/// This is not depth-dependent noise, notice it doesn't use the layer index 
/// dependency.
///
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};

const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

struct Ctx {
    damping: f64,
    drift_rate: f64,
    call_index: AtomicU64,
}

fn parse_field(config: Option<&str>, key: &str, fallback: f64) -> f64 {
    let Some(s) = config else { return fallback };
    let Some(idx) = s.find(key) else { return fallback };
    let Some(colon) = s[idx..].find(':') else { return fallback };
    let rest = s[idx + colon + 1..].trim_start();
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(fallback)
}

fn damping_factor(ctx: &Ctx, weight: u32, call_index: u64) -> f64 {
    let factor = 1.0 + ctx.drift_rate * call_index as f64;
    (-ctx.damping * weight as f64 * factor).exp()
}

#[no_mangle]
pub extern "C" fn propaq_noise_abi_version() -> u32 {
    PROPAQ_NOISE_ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_create(config_json: *const c_char) -> *mut c_void {
    let config = if config_json.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(config_json) }.to_str().ok()
    };
    let ctx = Box::new(Ctx {
        damping: parse_field(config, "\"damping\"", 0.001),
        drift_rate: parse_field(config, "\"drift_rate\"", 0.0),
        call_index: AtomicU64::new(0),
    });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_noise_factor(
    ctx: *mut c_void,
    _basis_kind: u32,
    _words: *const u64,
    _n_words: usize,
    _n_units: u32,
    weight: u32,
    _layer_index: u32,
    _n_layers: u32,
) -> f64 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let call_index = c.call_index.fetch_add(1, Ordering::Relaxed);
    damping_factor(c, weight, call_index)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_noise_factor_batch(
    ctx: *mut c_void,
    _basis_kind: u32,
    _words: *const u64,
    _n_words_per_term: usize,
    _n_units: u32,
    weights: *const u32,
    _layer_index: u32,
    _n_layers: u32,
    out: *mut f64,
    n_terms: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let base = c.call_index.fetch_add(n_terms as u64, Ordering::Relaxed);
    let weights = unsafe { std::slice::from_raw_parts(weights, n_terms) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n_terms) };
    for (i, (o, &w)) in out.iter_mut().zip(weights).enumerate() {
        *o = damping_factor(c, w, base + i as u64);
    }
    0
}
