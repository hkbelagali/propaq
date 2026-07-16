/// Example propaq native truncator plugin, implemented in Rust.
///
/// Joint weight/coefficient score instead of two independent hard cutoffs:
/// composing WeightTruncator and CoefficientTruncator ANDs two separate
/// thresholds, which can't express a smooth tradeoff between them. This keeps
/// a term if its coefficient magnitude, discounted by an exponential in its
/// weight, clears a single threshold:
///
///   keep <=> coeff_magnitude * exp(-alpha * weight) > threshold
///
/// alpha = 0 reduces to a plain coefficient cutoff.
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

struct Ctx {
    threshold: f64,
    alpha: f64,
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
pub extern "C" fn propaq_truncator_abi_version() -> u32 {
    PROPAQ_TRUNCATOR_ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_create(config_json: *const c_char) -> *mut c_void {
    let config = if config_json.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(config_json) }.to_str().ok()
    };
    let ctx = Box::new(Ctx {
        threshold: parse_field(config, "\"threshold\"", 1e-6),
        alpha: parse_field(config, "\"alpha\"", 0.1),
    });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

fn keep(ctx: &Ctx, term_weight: u32, coeff_magnitude: f64) -> bool {
    let score = coeff_magnitude * (-ctx.alpha * term_weight as f64).exp();
    score > ctx.threshold
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_keep(
    ctx: *mut c_void,
    term_weight: u32,
    coeff_magnitude: f64,
    _active_modes: u32,
) -> i32 {
    keep(unsafe { &*(ctx as *const Ctx) }, term_weight, coeff_magnitude) as i32
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_keep_batch(
    ctx: *mut c_void,
    term_weights: *const u32,
    coeff_magnitudes: *const f64,
    _active_modes: *const u32,
    out_keep: *mut u8,
    n: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let weights = unsafe { std::slice::from_raw_parts(term_weights, n) };
    let coeffs = unsafe { std::slice::from_raw_parts(coeff_magnitudes, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_keep, n) };
    for ((o, &w), &m) in out.iter_mut().zip(weights).zip(coeffs) {
        *o = keep(c, w, m) as u8;
    }
    0
}
