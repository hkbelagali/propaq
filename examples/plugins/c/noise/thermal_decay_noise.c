// Example propaq native noise plugin, implemented in C.
//
// Stretched-exponential relaxation: damping_factor = exp(-(gamma * weight)^beta).
// beta = 1 reduces exactly to UniformNoiseModel's plain exponential; beta != 1
// gives a different decay shape (sub-/super-exponential in weight)

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PROPAQ_NOISE_ABI_VERSION 1u

typedef struct {
    double gamma;
    double beta;
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
    ctx->gamma = parse_double(config_json, "\"gamma\"", 0.001);
    ctx->beta = parse_double(config_json, "\"beta\"", 1.0);
    return ctx;
}

void propaq_noise_destroy(void* ctx) {
    free(ctx);
}

static double damping_factor(const Ctx* ctx, uint32_t term_weight) {
    return exp(-pow(ctx->gamma * (double)term_weight, ctx->beta));
}

double propaq_noise_damping_factor(void* ctx, uint32_t term_weight, uint32_t active_modes) {
    (void)active_modes;
    return damping_factor((const Ctx*)ctx, term_weight);
}

int32_t propaq_noise_damping_batch(void* ctx, const uint32_t* term_weights,
                                    const uint32_t* active_modes, double* out, size_t n) {
    (void)active_modes;
    const Ctx* c = (const Ctx*)ctx;
    for (size_t i = 0; i < n; i++) {
        out[i] = damping_factor(c, term_weights[i]);
    }
    return 0;
}
