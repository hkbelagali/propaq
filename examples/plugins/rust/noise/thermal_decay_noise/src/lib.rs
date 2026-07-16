/// Example propaq native noise plugin, implemented in Rust.
///
/// Stretched-exponential relaxation: damping_factor = exp(-(gamma * weight)^beta).
/// beta = 1 reduces exactly to UniformNoiseModel's plain exponential; beta != 1
/// gives a different decay shape (sub-/super-exponential in weight), which is
/// not expressible by UniformNoiseModel's single-parameter formula.
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

struct Ctx {
    gamma: f64,
    beta: f64,
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
        gamma: parse_field(config, "\"gamma\"", 0.001),
        beta: parse_field(config, "\"beta\"", 1.0),
    });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

fn damping_factor(ctx: &Ctx, term_weight: u32) -> f64 {
    (-(ctx.gamma * term_weight as f64).powf(ctx.beta)).exp()
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_damping_factor(ctx: *mut c_void, term_weight: u32, _active_modes: u32) -> f64 {
    damping_factor(unsafe { &*(ctx as *const Ctx) }, term_weight)
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
    let weights = unsafe { std::slice::from_raw_parts(term_weights, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (o, &w) in out.iter_mut().zip(weights) {
        *o = damping_factor(c, w);
    }
    0
}
