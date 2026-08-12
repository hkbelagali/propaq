/// Example propaq native truncator plugin, implemented in Rust.
///
/// Locality-aware truncation: terms are cheap while they stay inside a region of
/// interest and are penalized for every unit they touch outside it.
///
///   keep <=> coeff_magnitude * exp(-alpha * |support(term) \ mask|) > threshold
///
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_TRUNCATOR_ABI_VERSION: u32 = 1;

const PROPAQ_DEPENDS_KEY: u32 = 1 << 0;

struct Ctx {
    threshold: f64,
    alpha: f64,
    mask: u64,
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

fn parse_mask(config: Option<&str>, key: &str, fallback: u64) -> u64 {
    let Some(s) = config else { return fallback };
    let Some(idx) = s.find(key) else { return fallback };
    let Some(colon) = s[idx..].find(':') else { return fallback };
    let rest = s[idx + colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().unwrap_or(fallback)
}

/// The key slice, or empty when propaq passed NULL because this plugin did not
/// declare PROPAQ_DEPENDS_KEY.
unsafe fn words_of<'a>(words: *const u64, n: usize) -> &'a [u64] {
    if words.is_null() || n == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(words, n) }
    }
}

fn outside_support(ctx: &Ctx, words: &[u64], n_units: u32) -> u32 {
    let mut count = 0;
    for q in 0..n_units {
        let bit = 2 * q as usize;
        let word = bit / 64;
        if word >= words.len() {
            break;
        }
        if (words[word] >> (bit % 64)) & 3 == 0 {
            continue;
        }
        // Units past 64 have no mask bit and count as outside the region.
        if q < 64 && (ctx.mask >> q) & 1 != 0 {
            continue;
        }
        count += 1;
    }
    count
}

fn keep(ctx: &Ctx, words: &[u64], n_units: u32, coeff_magnitude: f64) -> bool {
    let score = coeff_magnitude * (-ctx.alpha * outside_support(ctx, words, n_units) as f64).exp();
    score > ctx.threshold
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
        alpha: parse_field(config, "\"alpha\"", 1.0),
        mask: parse_mask(config, "\"mask\"", u64::MAX),
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
    n_units: u32,
    _weight: u32,
    coeff_magnitude: f64,
    _layer_index: u32,
    _n_layers: u32,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    keep(c, unsafe { words_of(words, n_words) }, n_units, coeff_magnitude) as i32
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_truncator_keep_batch(
    ctx: *mut c_void,
    _basis_kind: u32,
    words: *const u64,
    n_words_per_term: usize,
    n_units: u32,
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
        *o = keep(c, &all[i * n_words_per_term..(i + 1) * n_words_per_term], n_units, m) as u8;
    }
    0
}
