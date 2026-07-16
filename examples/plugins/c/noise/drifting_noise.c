// Example propaq native noise plugin, implemented in C.
//
// Non-stationary noise: damping grows with the number of calls already made
// through this ctx, modeling drift/heating that worsens over circuit depth
// rather than staying fixed like UniformNoiseModel.
//
//   damping_factor = exp(-damping * weight * (1 + drift_rate * call_index))
//

#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

typedef struct {
    double damping;
    double drift_rate;
    _Atomic uint64_t call_index;
} Ctx;

uint32_t propaq_noise_abi_version(void) {
    return PROPAQ_NOISE_ABI_VERSION;
}

static double parse_double(const char* config_json, const char* key, double fallback) {
    if (config_json == NULL) return fallback;
    const char* found = strstr(config_json, key);
    if (found == NULL) return fallback;
    const char* colon = strchr(found, ':');
    if (colon == NULL) return fallback;
    return atof(colon + 1);
}

void* propaq_noise_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->damping = parse_double(config_json, "\"damping\"", 0.001);
    ctx->drift_rate = parse_double(config_json, "\"drift_rate\"", 1e-5);
    atomic_init(&ctx->call_index, 0);
    return ctx;
}

void propaq_noise_destroy(void* ctx) {
    free(ctx);
}

static double damping_factor(Ctx* ctx, uint32_t term_weight, uint64_t call_index) {
    double factor = 1.0 + ctx->drift_rate * (double)call_index;
    return exp(-ctx->damping * (double)term_weight * factor);
}

double propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes) {
    (void)active_modes;
    Ctx* c = (Ctx*)ctx;
    uint64_t call_index = atomic_fetch_add_explicit(&c->call_index, 1, memory_order_relaxed);
    return damping_factor(c, term_weight, call_index);
}

int32_t propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
                                    const uint32_t* active_modes, double* out, size_t n) {
    (void)active_modes;
    Ctx* c = (Ctx*)ctx;
    // Reserve the whole chunk's worth of call indices with one atomic op
    // instead of one fetch_add per term.
    uint64_t base = atomic_fetch_add_explicit(&c->call_index, (uint64_t)n, memory_order_relaxed);
    for (size_t i = 0; i < n; i++) {
        out[i] = damping_factor(c, term_weights[i], base + (uint64_t)i);
    }
    return 0;
}
