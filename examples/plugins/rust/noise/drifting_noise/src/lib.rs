/// Example propaq native noise plugin, implemented in Rust.
///
/// Non-stationary noise: damping grows with the number of calls already made
/// through this ctx, modeling drift/heating that worsens over circuit depth
/// rather than staying fixed like UniformNoiseModel.
///
///   damping_factor = exp(-damping * weight * (1 + drift_rate * call_index))
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
        drift_rate: parse_field(config, "\"drift_rate\"", 1e-5),
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

fn damping_factor(ctx: &Ctx, term_weight: u32, call_index: u64) -> f64 {
    let factor = 1.0 + ctx.drift_rate * call_index as f64;
    (-ctx.damping * term_weight as f64 * factor).exp()
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_damping_factor(ctx: *mut c_void, term_weight: u32, _active_modes: u32) -> f64 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let call_index = c.call_index.fetch_add(1, Ordering::Relaxed);
    damping_factor(c, term_weight, call_index)
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_damping_batch(
    ctx: *mut c_void,
    term_weights: *const u32,
    _active_modes: *const u32,
    out: *mut f64,
    n: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    // Reserve the whole chunk's worth of call indices with one atomic op
    // instead of one fetch_add per term.
    let base = c.call_index.fetch_add(n as u64, Ordering::Relaxed);
    let weights = unsafe { std::slice::from_raw_parts(term_weights, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (i, (o, &w)) in out.iter_mut().zip(weights).enumerate() {
        *o = damping_factor(c, w, base + i as u64);
    }
    0
}
