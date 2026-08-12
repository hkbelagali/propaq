// Example propaq truncator plugin, implemented in C.
//
// Weight alone, so it declares no dependencies
//

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_TRUNCATOR_ABI_VERSION 1u

typedef struct {
    uint32_t max_weight;
} Ctx;

uint32_t propaq_truncator_abi_version(void) {
    return PROPAQ_TRUNCATOR_ABI_VERSION;
}

static uint32_t parse_max_weight(const char* config_json) {
    if (config_json == NULL) return UINT32_MAX;
    const char* key = strstr(config_json, "\"max_weight\"");
    if (key == NULL) return UINT32_MAX;
    const char* colon = strchr(key, ':');
    if (colon == NULL) return UINT32_MAX;
    return (uint32_t)strtoul(colon + 1, NULL, 10);
}

void* propaq_truncator_create(const char* config_json) {
    Ctx* ctx = (Ctx*)malloc(sizeof(Ctx));
    ctx->max_weight = parse_max_weight(config_json);
    return ctx;
}

void propaq_truncator_destroy(void* ctx) {
    free(ctx);
}

int32_t propaq_truncator_keep(void* ctx, uint32_t basis_kind, const uint64_t* words, size_t n_words,
                              uint32_t n_units, uint32_t weight, double coeff_magnitude,
                              uint32_t layer_index, uint32_t n_layers) {
    (void)basis_kind; (void)words; (void)n_words; (void)n_units;
    (void)coeff_magnitude; (void)layer_index; (void)n_layers;
    return weight <= ((Ctx*)ctx)->max_weight ? 1 : 0;
}

int32_t propaq_truncator_keep_batch(void* ctx, uint32_t basis_kind, const uint64_t* words,
                                    size_t n_words_per_term, uint32_t n_units,
                                    const uint32_t* weights, const double* coeff_magnitudes,
                                    uint32_t layer_index, uint32_t n_layers, uint8_t* out_keep,
                                    size_t n_terms) {
    (void)basis_kind; (void)words; (void)n_words_per_term; (void)n_units;
    (void)coeff_magnitudes; (void)layer_index; (void)n_layers;
    uint32_t max_weight = ((Ctx*)ctx)->max_weight;
    for (size_t i = 0; i < n_terms; i++) {
        out_keep[i] = weights[i] <= max_weight ? 1 : 0;
    }
    return 0;
}
