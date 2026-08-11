// Example propaq truncator plugin, implemented in C.
//
// Locality-aware truncation: terms are cheap while they stay inside a region of
// interest and are penalized for every unit they touch outside it.
//
//   keep <=> coeff_magnitude * exp(-alpha * |support(term) \ mask|) > threshold
//
// Reads the term's key, so it declares PROPAQ_DEPENDS_KEY
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
    double alpha;
    uint64_t mask;
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

static uint64_t parse_u64(const char* config_json, const char* key, uint64_t fallback) {
    if (config_json == NULL) return fallback;
    const char* found = strstr(config_json, key);
    if (found == NULL) return fallback;
    const char* colon = strchr(found, ':');
    if (colon == NULL) return fallback;
    return (uint64_t)strtoull(colon + 1, NULL, 10);
}

void* propaq_truncator_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->threshold = parse_double(config_json, "\"threshold\"", 1e-6);
    ctx->alpha = parse_double(config_json, "\"alpha\"", 1.0);
    ctx->mask = parse_u64(config_json, "\"mask\"", ~(uint64_t)0);
    return ctx;
}

void propaq_truncator_destroy(void* ctx) {
    free(ctx);
}

static uint32_t outside_support(const Ctx* ctx, const uint64_t* words, size_t n_words, uint32_t n_units) {
    uint32_t count = 0;
    for (uint32_t q = 0; q < n_units; q++) {
        size_t bit = (size_t)(2u * (size_t)q);
        size_t word = bit / 64u;
        if (word >= n_words) break;
        if (((words[word] >> (bit % 64u)) & 3u) == 0u) continue;
        // Units past 64 have no mask bit and count as outside the region.
        if (q < 64u && ((ctx->mask >> q) & 1u)) continue;
        count++;
    }
    return count;
}

static int32_t keep(const Ctx* ctx, const uint64_t* words, size_t n_words, uint32_t n_units,
                    double coeff_magnitude) {
    double score = coeff_magnitude * exp(-ctx->alpha * (double)outside_support(ctx, words, n_words, n_units));
    return score > ctx->threshold ? 1 : 0;
}

int32_t propaq_truncator_keep(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                              uint32_t n_units, uint32_t weight, double coeff_magnitude,
                              uint32_t layer_index, uint32_t n_layers) {
    (void)basis_kind; (void)weight; (void)layer_index; (void)n_layers;
    return keep((const Ctx*)ctx, words, n_words, n_units, coeff_magnitude);
}

int32_t propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                    size_t n_words_per_term, uint32_t n_units,
                                    const uint32_t* weights, const double* coeff_magnitudes,
                                    uint32_t layer_index, uint32_t n_layers, uint8_t* out_keep,
                                    size_t n_terms) {
    (void)basis_kind; (void)weights; (void)layer_index; (void)n_layers;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n_terms; i++) {
        out_keep[i] = (uint8_t)keep(c, words + i * n_words_per_term, n_words_per_term, n_units,
                                    coeff_magnitudes[i]);
    }
    return 0;
}
