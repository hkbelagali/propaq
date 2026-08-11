// Example propaq noise plugin, implemented in C.
//
// Decoherence that worsens with circuit depth
//
//   damping_factor = exp(-damping * weight * (1 + rate * layer_index / n_layers))
//
// This reads the layer index, so it declares
// PROPAQ_DEPENDS_LAYER alone. That keeps it on propaq's tabulated fast path.
//
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

#define PROPAQ_DEPENDS_KEY   (1u << 0)
#define PROPAQ_DEPENDS_LAYER (1u << 1)

typedef struct {
    double damping;
    double rate;
} Ctx;

uint32_t propaq_noise_abi_version(void) {
    return PROPAQ_NOISE_ABI_VERSION;
}

uint32_t propaq_noise_depends(void) {
    return PROPAQ_DEPENDS_LAYER;
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
    ctx->rate = parse_double(config_json, "\"rate\"", 1.0);
    return ctx;
}

void propaq_noise_destroy(void* ctx) {
    free(ctx);
}

static double damping_factor(const Ctx* ctx, uint32_t weight, uint32_t layer_index,
                             uint32_t n_layers) {
    double depth = n_layers > 0u ? (double)layer_index / (double)n_layers : 0.0;
    return exp(-ctx->damping * (double)weight * (1.0 + ctx->rate * depth));
}

double propaq_noise_factor(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    return damping_factor((const Ctx*)ctx, weight, layer_index, n_layers);
}

int32_t propaq_noise_factor_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t* weights, uint32_t layer_index, uint32_t n_layers,
                                  double* out, size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n_terms; i++) {
        out[i] = damping_factor(c, weights[i], layer_index, n_layers);
    }
    return 0;
}
