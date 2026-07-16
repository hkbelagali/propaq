/// Example propaq native truncator plugin, implemented in Rust.
///
/// Importance-sampling truncation: instead of a deterministic cutoff, keep
/// each term with probability min(1, coeff_magnitude / threshold). Small
/// coefficients are discarded most (but not all) of the time rather than
/// always, which trades a small amount of variance for keeping the estimator
/// unbiased in expectation.
///
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};

const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

struct Ctx {
    threshold: f64,
    seed: u64,
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

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Uniform double in [0, 1) from the top 53 bits, the standard construction.
fn unit_interval(bits: u64) -> f64 {
    (bits >> 11) as f64 * (1.0 / 9007199254740992.0) // 1 / 2^53
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
        seed: parse_field(config, "\"seed\"", 42.0) as u64,
        call_index: AtomicU64::new(0),
    });
    Box::into_raw(ctx) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_destroy(ctx: *mut c_void) {
    if !ctx.is_null() {
        drop(unsafe { Box::from_raw(ctx as *mut Ctx) });
    }
}

fn keep(ctx: &Ctx, coeff_magnitude: f64, call_index: u64) -> bool {
    let probability = coeff_magnitude / ctx.threshold;
    if probability >= 1.0 {
        return true;
    }
    let draw = unit_interval(splitmix64(ctx.seed ^ call_index));
    draw < probability
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_keep(
    ctx: *mut c_void,
    _term_weight: u32,
    coeff_magnitude: f64,
    _active_modes: u32,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let call_index = c.call_index.fetch_add(1, Ordering::Relaxed);
    keep(c, coeff_magnitude, call_index) as i32
}

#[no_mangle]
pub unsafe extern "C" fn propaq_truncator_keep_batch(
    ctx: *mut c_void,
    _term_weights: *const u32,
    coeff_magnitudes: *const f64,
    _active_modes: *const u32,
    out_keep: *mut u8,
    n: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let base = c.call_index.fetch_add(n as u64, Ordering::Relaxed);
    let coeffs = unsafe { std::slice::from_raw_parts(coeff_magnitudes, n) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_keep, n) };
    for (i, (o, &m)) in out.iter_mut().zip(coeffs).enumerate() {
        *o = keep(c, m, base + i as u64) as u8;
    }
    0
}
