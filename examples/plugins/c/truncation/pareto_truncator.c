// Example propaq truncator plugin, implemented in C.
//
// Joint weight/coefficient score instead of two independent hard cutoffs.
// composing WeightTruncator and CoefficientTruncator ANDs two separate
// thresholds, which can't express a smooth tradeoff between them. This keeps
// a term if its coefficient magnitude, discounted by an exponential in its
// weight, clears a single threshold:
//
//   keep <=> coeff_magnitude * exp(-alpha * weight) > threshold
//
// alpha = 0 reduces to a plain coefficient cutoff.
//
// This only uses weight and magnitude, so doesn't need any dependencies
//

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_TRUNCATOR_ABI_VERSION 1u

typedef struct {
    double threshold;
    double alpha;
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

void* propaq_truncator_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->threshold = parse_double(config_json, "\"threshold\"", 1e-6);
    ctx->alpha = parse_double(config_json, "\"alpha\"", 0.1);
    return ctx;
}

void propaq_truncator_destroy(void* ctx) {
    free(ctx);
}

static int32_t keep(const Ctx* ctx, uint32_t term_weight, double coeff_magnitude) {
    double score = coeff_magnitude * exp(-ctx->alpha * (double)term_weight);
    return score > ctx->threshold ? 1 : 0;
}

int32_t propaq_truncator_keep(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                              uint32_t n_units, uint32_t weight, double coeff_magnitude,
                              uint32_t layer_index, uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    (void)layer_index; (void)n_layers;
    return keep((const Ctx*)ctx, weight, coeff_magnitude);
}

int32_t propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                    size_t n_words_per_term, uint32_t n_units,
                                    const uint32_t* weights, const double* coeff_magnitudes,
                                    uint32_t layer_index, uint32_t n_layers, uint8_t* out_keep,
                                    size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    (void)layer_index; (void)n_layers;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n_terms; i++) {
        out_keep[i] = (uint8_t)keep(c, weights[i], coeff_magnitudes[i]);
    }
    return 0;
}
