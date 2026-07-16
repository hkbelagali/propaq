/// Example propaq native noise plugin, implemented in Rust.

use std::ffi::{c_char, c_void, CStr};

const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

struct Ctx {
    damping: f64,
}

fn parse_damping(config_json: Option<&str>) -> f64 {
    let Some(s) = config_json else { return 0.0 };
    let Some(idx) = s.find("\"damping\"") else { return 0.0 };
    let Some(colon) = s[idx..].find(':') else { return 0.0 };
    let rest = s[idx + colon + 1..].trim_start();
    let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+' || c == 'e' || c == 'E'))
        .unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(0.0)
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
    let ctx = Box::new(Ctx { damping: parse_damping(config) });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_damping_factor(ctx: *mut c_void, term_weight: u32, _active_modes: u32) -> f64 {
    let damping = unsafe { &*(ctx as *const Ctx) }.damping;
    (-damping * term_weight as f64).exp()
}

#[no_mangle]
pub unsafe extern "C" fn propaq_noise_damping_batch(
    ctx: *mut c_void,
    term_weights: *const u32,
    _active_modes: *const u32,
    out: *mut f64,
    n: usize,
) -> i32 {
    let damping = unsafe { &*(ctx as *const Ctx) }.damping;
    let weights = unsafe { std::slice::from_raw_parts(term_weights, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (o, &w) in out.iter_mut().zip(weights) {
        *o = (-damping * w as f64).exp();
    }
    0
}
