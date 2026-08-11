# Example propaq noise plugin, implemented in Julia (AOT-compiled via
# PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Decoherence that worsens with circuit depth:
#
#   damping_factor = exp(-damping * weight * (1 + rate * layer_index / n_layers))
#
# Reads the layer index, so it declares PROPAQ_DEPENDS_LAYER alone. 
#

const PROPAQ_NOISE_ABI_VERSION = UInt32(1)

const PROPAQ_DEPENDS_LAYER = UInt32(1) << 1

mutable struct Ctx
    damping::Float64
    rate::Float64
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

function damping_factor(damping::Float64, rate::Float64, weight::UInt32,
                        layer_index::UInt32, n_layers::UInt32)

    depth = n_layers > 0 ? Float64(layer_index) / Float64(n_layers) : 0.0
    return exp(-damping * Float64(weight) * (1.0 + rate * depth))
end

Base.@ccallable function propaq_noise_abi_version()::UInt32
    PROPAQ_NOISE_ABI_VERSION
end

Base.@ccallable function propaq_noise_depends()::UInt32
    PROPAQ_DEPENDS_LAYER
end

Base.@ccallable function propaq_noise_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "damping", 0.001), parse_field(json, "rate", 1.0))
    key = pointer_from_objref(ctx)
    lock(CTX_LOCK) do
        CTX_STORE[key] = ctx
    end
    return key
end

Base.@ccallable function propaq_noise_destroy(ctx::Ptr{Cvoid})::Cvoid
    lock(CTX_LOCK) do
        delete!(CTX_STORE, ctx)
    end
    return nothing
end

Base.@ccallable function propaq_noise_factor(ctx::Ptr{Cvoid}, basis_kind::UInt32,
                                             words::Ptr{UInt64}, n_words::Csize_t,
                                             n_units::UInt32, weight::UInt32,
                                             layer_index::UInt32, n_layers::UInt32)::Cdouble
    c = unsafe_pointer_to_objref(ctx)::Ctx
    damping_factor(c.damping, c.rate, weight, layer_index, n_layers)
end

Base.@ccallable function propaq_noise_factor_batch(ctx::Ptr{Cvoid}, basis_kind::UInt32,
                                                   words::Ptr{UInt64}, n_words_per_term::Csize_t,
                                                   n_units::UInt32, weights::Ptr{UInt32},
                                                   layer_index::UInt32, n_layers::UInt32,
                                                   out::Ptr{Cdouble}, n_terms::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    w = unsafe_wrap(Array, weights, n_terms)
    result = unsafe_wrap(Array, out, n_terms)
    @inbounds for i in 1:n_terms
        result[i] = damping_factor(c.damping, c.rate, w[i], layer_index, n_layers)
    end
    return Int32(0)
end
