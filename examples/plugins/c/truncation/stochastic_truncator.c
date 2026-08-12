// Example propaq native truncator plugin, implemented in C.
//
// Importance-sampling truncation: instead of a deterministic cutoff, keep
// each term with probability min(1, coeff_magnitude / threshold). Small
// coefficients are discarded most (but not all) of the time rather than
// always, which trades a small amount of variance for keeping the estimator
// unbiased in expectation.
//

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_TRUNCATOR_ABI_VERSION 1u

#define PROPAQ_DEPENDS_KEY   (1u << 0)
#define PROPAQ_DEPENDS_LAYER (1u << 1)

typedef struct {
    double threshold;
    uint64_t seed;
} Ctx;

uint32_t propaq_truncator_abi_version(void) {
    return PROPAQ_TRUNCATOR_ABI_VERSION;
}

uint32_t propaq_truncator_depends(void) {
    return PROPAQ_DEPENDS_KEY;
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
    return ctx;
}

void propaq_truncator_destroy(void* ctx) {
    free(ctx);
}

// A stream position derived from the term itself
static uint64_t key_stream(const Ctx* ctx, const uint64_t* words, size_t n_words) {
    uint64_t h = ctx->seed;
    for (size_t i = 0; i < n_words; i++) {
        h = splitmix64(h ^ words[i]);
    }
    return h;
}

static int32_t keep(const Ctx* ctx, const uint64_t* words, size_t n_words, double coeff_magnitude) {
    double probability = coeff_magnitude / ctx->threshold;
    if (probability >= 1.0) return 1;
    double draw = unit_interval(splitmix64(key_stream(ctx, words, n_words)));
    return draw < probability ? 1 : 0;
}

int32_t propaq_truncator_keep(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                              uint32_t n_units, uint32_t weight, double coeff_magnitude,
                              uint32_t layer_index, uint32_t n_layers) {
    (void)basis_kind; (void)n_units; (void)weight; (void)layer_index; (void)n_layers;
    return keep((const Ctx*)ctx, words, n_words, coeff_magnitude);
}

int32_t propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                    size_t n_words_per_term, uint32_t n_units,
                                    const uint32_t* weights, const double* coeff_magnitudes,
                                    uint32_t layer_index, uint32_t n_layers, uint8_t* out_keep,
                                    size_t n_terms) {
    (void)basis_kind; (void)n_units; (void)weights; (void)layer_index; (void)n_layers;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n_terms; i++) {
        out_keep[i] = (uint8_t)keep(c, words + i * n_words_per_term, n_words_per_term,
                                    coeff_magnitudes[i]);
    }
    return 0;
}
