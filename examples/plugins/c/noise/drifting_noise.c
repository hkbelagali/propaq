// Example propaq native noise plugin, implemented in C.
//
// A demonstration of *stateful ctx*: damping grows with the number of calls
// already made through this ctx, rather than staying fixed like
// UniformNoiseModel.
//
//   damping_factor = exp(-damping * weight * (1 + drift_rate * call_index))
//
// NOTE: This model is not layer-dependent depolarizing noise. 
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
    ctx->drift_rate = parse_double(config_json, "\"drift_rate\"", 0.0);
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

double propaq_noise_factor(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    (void)layer_index; (void)n_layers;
    Ctx* c = (Ctx*)ctx;
    uint64_t call_index = atomic_fetch_add_explicit(&c->call_index, 1, memory_order_relaxed);
    return damping_factor(c, weight, call_index);
}

int32_t propaq_noise_factor_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t* weights, uint32_t layer_index, uint32_t n_layers,
                                  double* out, size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    (void)layer_index; (void)n_layers;
    Ctx* c = (Ctx*)ctx;
    uint64_t base = atomic_fetch_add_explicit(&c->call_index, (uint64_t)n_terms, memory_order_relaxed);
    for (size_t i = 0; i < n_terms; i++) {
        out[i] = damping_factor(c, weights[i], base + (uint64_t)i);
    }
    return 0;
}
