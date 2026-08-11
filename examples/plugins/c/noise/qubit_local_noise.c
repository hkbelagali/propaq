// Example propaq native noise plugin, implemented in C.
//
// Only the masked units are noisy:
//
//   damping_factor = exp(-damping * |support(term) & mask|)
//
// This reads the term's key, so it declares PROPAQ_DEPENDS_KEY.
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
    uint64_t mask;
} Ctx;

uint32_t propaq_noise_abi_version(void) {
    return PROPAQ_NOISE_ABI_VERSION;
}

uint32_t propaq_noise_depends(void) {
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

void* propaq_noise_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->damping = parse_double(config_json, "\"damping\"", 0.001);
    ctx->mask = parse_u64(config_json, "\"mask\"", ~(uint64_t)0);
    return ctx;
}

void propaq_noise_destroy(void* ctx) {
    free(ctx);
}

static uint32_t masked_support(const Ctx* ctx, const uint64_t* words, size_t n_words, uint32_t n_units) {
    uint32_t limit = n_units < 64u ? n_units : 64u;
    uint32_t count = 0;
    for (uint32_t q = 0; q < limit; q++) {
        if (((ctx->mask >> q) & 1u) == 0u) continue;
        size_t bit = (size_t)(2u * q);
        size_t word = bit / 64u;
        if (word >= n_words) break;
        if ((words[word] >> (bit % 64u)) & 3u) count++;
    }
    return count;
}

static double factor(const Ctx* ctx, const uint64_t* words, size_t n_words, uint32_t n_units) {
    return exp(-ctx->damping * (double)masked_support(ctx, words, n_words, n_units));
}

double propaq_noise_factor(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                           uint32_t n_units, uint32_t weight, uint32_t layer_index,
                           uint32_t n_layers) {
    (void)basis_kind; (void)weight; (void)layer_index; (void)n_layers;
    return factor((const Ctx*)ctx, words, n_words, n_units);
}

int32_t propaq_noise_factor_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                  size_t n_words_per_term, uint32_t n_units,
                                  const uint32_t* weights, uint32_t layer_index, uint32_t n_layers,
                                  double* out, size_t n_terms) {
    (void)basis_kind; (void)weights; (void)layer_index; (void)n_layers;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n_terms; i++) {
        out[i] = factor(c, words + i * n_words_per_term, n_words_per_term, n_units);
    }
    return 0;
}
