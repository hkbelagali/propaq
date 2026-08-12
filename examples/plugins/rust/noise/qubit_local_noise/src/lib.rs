/// Example propaq noise plugin, implemented in Rust.
///
/// Per-qubit damping, only the units named by a bitmask are noisy, and a term is
/// damped once for each noisy unit it acts on non-trivially.
///
///   damping_factor = exp(-damping * |support(term) & mask|)
///
/// It declares PROPAQ_DEPENDS_KEY since it needs the entire term's representation
///
use std::ffi::{c_char, c_void, CStr};

const PROPAQ_NOISE_ABI_VERSION: u32 = 1;

const PROPAQ_DEPENDS_KEY: u32 = 1 << 0;

struct Ctx {
    damping: f64,
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

fn masked_support(ctx: &Ctx, words: &[u64], n_units: u32) -> u32 {
    let limit = n_units.min(64);
    let mut count = 0;
    for q in 0..limit {
        if (ctx.mask >> q) & 1 == 0 {
            continue;
        }
        let bit = 2 * q as usize;
        let word = bit / 64;
        if word >= words.len() {
            break;
        }
        if (words[word] >> (bit % 64)) & 3 != 0 {
            count += 1;
        }
    }
    count
}

fn factor(ctx: &Ctx, words: &[u64], n_units: u32) -> f64 {
    (-ctx.damping * masked_support(ctx, words, n_units) as f64).exp()
}

#[no_mangle]
pub extern "C" fn propaq_noise_abi_version() -> u32 {
    PROPAQ_NOISE_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn propaq_noise_depends() -> u32 {
    PROPAQ_DEPENDS_KEY
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
        mask: parse_mask(config, "\"mask\"", u64::MAX),
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
    words: *const u64,
    n_words: usize,
    n_units: u32,
    _weight: u32,
    _layer_index: u32,
    _n_layers: u32,
) -> f64 {
    let c = unsafe { &*(ctx as *const Ctx) };
    factor(c, unsafe { words_of(words, n_words) }, n_units)
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn propaq_noise_factor_batch(
    ctx: *mut c_void,
    _basis_kind: u32,
    words: *const u64,
    n_words_per_term: usize,
    n_units: u32,
    _weights: *const u32,
    _layer_index: u32,
    _n_layers: u32,
    out: *mut f64,
    n_terms: usize,
) -> i32 {
    let c = unsafe { &*(ctx as *const Ctx) };
    let all = unsafe { words_of(words, n_words_per_term * n_terms) };
    let out = unsafe { std::slice::from_raw_parts_mut(out, n_terms) };
    for (i, o) in out.iter_mut().enumerate() {
        *o = factor(c, &all[i * n_words_per_term..(i + 1) * n_words_per_term], n_units);
    }
    0
}
