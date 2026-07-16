// Example propaq native noise plugin, implemented in C.

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

double propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes) {
    (void)active_modes;
    double damping = ((Ctx*)ctx)->damping;
    return exp(-damping * (double)term_weight);
}

int32_t propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
                                    const uint32_t* active_modes, double* out, size_t n) {
    (void)active_modes;
    double damping = ((Ctx*)ctx)->damping;
    for (size_t i = 0; i < n; i++) {
        out[i] = exp(-damping * (double)term_weights[i]);
    }
    return 0;
}
