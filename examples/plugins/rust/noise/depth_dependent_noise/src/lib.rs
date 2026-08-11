/// Example propaq noise plugin, implemented in Rust.
///
/// Decoherence that worsens with circuit depth
///
///   damping_factor = exp(-damping * weight * (1 + rate * layer_index / n_layers))
///
/// This reads the layer index, so it declares PROPAQ_DEPENDS_LAYER. 
///
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

const PROPAQ_DEPENDS_LAYER: u32 = 1 << 1;

struct Ctx {
    damping: f64,
    rate: f64,
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

fn damping_factor(ctx: &Ctx, weight: u32, layer_index: u32, n_layers: u32) -> f64 {
    // Depth as a fraction of the circuit, so `rate` means the same thing
    // regardless of how deep the circuit happens to be.
    let depth = if n_layers > 0 {
        layer_index as f64 / n_layers as f64
    } else {
        0.0
    };
    (-ctx.damping * weight as f64 * (1.0 + ctx.rate * depth)).exp()
}

#[no_mangle]
pub extern "C" fn propaq_noise_abi_version() -> u32 {
    PROPAQ_NOISE_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn propaq_noise_depends() -> u32 {
    PROPAQ_DEPENDS_LAYER
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
        rate: parse_field(config, "\"rate\"", 1.0),
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
    layer_index: u32,
    n_layers: u32,
) -> f64 {
    damping_factor(unsafe { &*(ctx as *const Ctx) }, weight, layer_index, n_layers)
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
    layer_index: u32,
    n_layers: u32,
    out: *mut f64,
    n_terms: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let weights = unsafe { std::slice::from_raw_parts(weights, n_terms) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n_terms) };
    for (o, &w) in out.iter_mut().zip(weights) {
        *o = damping_factor(c, w, layer_index, n_layers);
    }
    0
}
