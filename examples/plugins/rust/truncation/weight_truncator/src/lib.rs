/// Example propaq native truncator plugin, implemented in Rust.
///
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

struct Ctx {
    max_weight: u32,
}

fn parse_max_weight(config_json: Option<&str>) -> u32 {
    let Some(s) = config_json else { return u32::MAX };
    let Some(idx) = s.find("\"max_weight\"") else { return u32::MAX };
    let Some(colon) = s[idx..].find(':') else { return u32::MAX };
    let rest = s[idx + colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(u32::MAX)
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
        max_weight: parse_max_weight(config),
    });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_truncator_keep(
    ctx: *mut c_void,
    _basis_kind: u32,
    _words: *const u64,
    _n_words: usize,
    _n_units: u32,
    weight: u32,
    _coeff_magnitude: f64,
    _layer_index: u32,
    _n_layers: u32,
) -> i32 {
    let max_weight = unsafe { &*(ctx as *const Ctx) }.max_weight;
    (weight <= max_weight) as i32
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_truncator_keep_batch(
    ctx: *mut c_void,
    _basis_kind: u32,
    _words: *const u64,
    _n_words_per_term: usize,
    _n_units: u32,
    weights: *const u32,
    _coeff_magnitudes: *const f64,
    _layer_index: u32,
    _n_layers: u32,
    out_keep: *mut u8,
    n_terms: usize,
) -> i32 {
    let max_weight = unsafe { &*(ctx as *const Ctx) }.max_weight;
    let weights = unsafe { std::slice::from_raw_parts(weights, n_terms) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_keep, n_terms) };
    for (o, &w) in out.iter_mut().zip(weights) {
        *o = (w <= max_weight) as u8;
    }
    0
}
