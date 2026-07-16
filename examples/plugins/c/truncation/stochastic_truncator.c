// Example propaq native truncator plugin, implemented in C.
//
// Importance-sampling truncation: instead of a deterministic cutoff, keep
// each term with probability min(1, coeff_magnitude / threshold). Small
// coefficients are discarded most (but not all) of the time rather than
// always, which trades a small amount of variance for keeping the estimator
// unbiased in expectation.
//
// Plugin callbacks run concurrently from arbitrary worker threads sharing one
// ctx, so this deliberately avoids a shared, mutable RNG state (which would
// need a lock to read-modify-write safely). Instead each call draws its
// randomness from splitmix64 seeded by `seed ^ call_index`, where
// `call_index` is a lock-free atomic counter reserved once per call (once per
// chunk for the batch entry point)
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_TRUNCATOR_ABI_VERSION 1u

typedef struct {
    double threshold;
    uint64_t seed;
    _Atomic uint64_t call_index;
} Ctx;

uint32_t propaq_truncator_abi_version(void) {
    return PROPAQ_TRUNCATOR_ABI_VERSION;
}

static double parse_double(const char* config_json, const char* key, double fallback) {
    if (config_json == NULL) return fallback;
    const char* found = strstr(config_json, key);
    if (found == NULL) return fallback;
    const char* colon = strchr(found, ':');
    if (colon == NULL) return fallback;
    return atof(colon + 1);
}

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Uniform double in [0, 1) from the top 53 bits, the standard construction.
static double unit_interval(uint64_t bits) {
    return (double)(bits >> 11) * (1.0 / 9007199254740992.0); // 1 / 2^53
}

void* propaq_truncator_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->threshold = parse_double(config_json, "\"threshold\"", 1e-6);
    ctx->seed = (uint64_t)parse_double(config_json, "\"seed\"", 42.0);
    atomic_init(&ctx->call_index, 0);
    return ctx;
}

void propaq_truncator_destroy(void* ctx) {
    free(ctx);
}

static int32_t keep(const Ctx* ctx, double coeff_magnitude, uint64_t call_index) {
    double probability = coeff_magnitude / ctx->threshold;
    if (probability >= 1.0) return 1;
    double draw = unit_interval(splitmix64(ctx->seed ^ call_index));
    return draw < probability ? 1 : 0;
}

int32_t propaq_truncator_keep(void* ctx, uint32_t term_weight, double coeff_magnitude, uint32_t active_modes) {
    (void)term_weight;
    (void)active_modes;
    Ctx* c = (Ctx*)ctx;
    uint64_t call_index = atomic_fetch_add_explicit(&c->call_index, 1, memory_order_relaxed);
    return keep(c, coeff_magnitude, call_index);
}

int32_t propaq_truncator_keep_batch(void* ctx, const uint32_t* term_weights,
                                     const double* coeff_magnitudes, const uint32_t* active_modes,
                                     uint8_t* out_keep, size_t n) {
    (void)term_weights;
    (void)active_modes;
    Ctx* c = (Ctx*)ctx;
    uint64_t base = atomic_fetch_add_explicit(&c->call_index, (uint64_t)n, memory_order_relaxed);
    for (size_t i = 0; i < n; i++) {
        out_keep[i] = (uint8_t)keep(c, coeff_magnitudes[i], base + (uint64_t)i);
    }
    return 0;
}
