/// Example propaq truncator plugin, implemented in Rust.
///
/// Importance-sampling truncation, instead of a deterministic cutoff, keep
/// each term with probability min(1, coeff_magnitude / threshold). 
///
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

const PROPAQ_DEPENDS_KEY: u32 = 1 << 0;

struct Ctx {
    threshold: f64,
    seed: u64,
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

unsafe fn words_of<'a>(words: *const u64, n: usize) -> &'a [u64] {
    if words.is_null() || n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(words, n) }
    }
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
    (bits >> 11) as f64 * (1.0 / 9007199254740992.0)
}

/// A stream position derived from the term itself rather than from call order.
fn key_stream(ctx: &Ctx, words: &[u64]) -> u64 {
    let mut h = ctx.seed;
    for &w in words {
        h = splitmix64(h ^ w);
    }
    h
}

fn keep(ctx: &Ctx, words: &[u64], coeff_magnitude: f64) -> bool {
    let probability = coeff_magnitude / ctx.threshold;
    if probability >= 1.0 {
        return true;
    }
    unit_interval(splitmix64(key_stream(ctx, words))) < probability
}

#[no_mangle]
pub extern "C" fn propaq_truncator_abi_version() -> u32 {
    PROPAQ_TRUNCATOR_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn propaq_truncator_depends() -> u32 {
    PROPAQ_DEPENDS_KEY
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
    words: *const u64,
    n_words: usize,
    _n_units: u32,
    _weight: u32,
    coeff_magnitude: f64,
    _layer_index: u32,
    _n_layers: u32,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    keep(c, unsafe { words_of(words, n_words) }, coeff_magnitude) as i32
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_truncator_keep_batch(
    ctx: *mut c_void,
    _basis_kind: u32,
    words: *const u64,
    n_words_per_term: usize,
    _n_units: u32,
    _weights: *const u32,
    coeff_magnitudes: *const f64,
    _layer_index: u32,
    _n_layers: u32,
    out_keep: *mut u8,
    n_terms: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let all = unsafe { words_of(words, n_words_per_term * n_terms) };
    let mags = unsafe { std::slice::from_raw_parts(coeff_magnitudes, n_terms) };
    let out = unsafe { std::slice::from_raw_parts_mut(out_keep, n_terms) };
    for (i, (o, &m)) in out.iter_mut().zip(mags).enumerate() {
        *o = keep(c, &all[i * n_words_per_term..(i + 1) * n_words_per_term], m) as u8;
    }
    0
}
