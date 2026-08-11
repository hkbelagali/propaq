// Example propaq noise plugin, implemented in C.
//
// A function of term weight alone, so it declares no dependencies
//

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

typedef struct {
    double damping;
} Ctx;

uint32_t propaq_noise_abi_version(void) {
    return PROPAQ_NOISE_ABI_VERSION;
}


static double parse_damping(const char* config_json) {
    if (config_json == NULL) return 0.0;
    const char* key = strstr(config_json, "\"damping\"");
    if (key == NULL) return 0.0;
    const char* colon = strchr(key, ':');
    if (colon == NULL) return 0.0;
    return atof(colon + 1);
}

void* propaq_noise_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->damping = parse_damping(config_json);
    return ctx;
}

void propaq_noise_destroy(void* ctx) {
    free(ctx);
}

double propaq_noise_factor(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    (void)layer_index; (void)n_layers;
    double damping = ((Ctx*)ctx)->damping;
    return exp(-damping * (double)weight);
}

int32_t propaq_noise_factor_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t* weights, uint32_t layer_index, uint32_t n_layers,
                                  double* out, size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    (void)layer_index; (void)n_layers;
    double damping = ((Ctx*)ctx)->damping;
    for (size_t i = 0; i < n_terms; i++) {
        out[i] = exp(-damping * (double)weights[i]);
    }
    return 0;
}
