# Example propaq truncator plugin, implemented in Julia (AOT-compiled via
# PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Importance-sampling truncation, instead of a deterministic cutoff, keep
# each term with probability min(1, coeff_magnitude / threshold).
#

const PROPAQ_TRUNCATOR_ABI_VERSION = UInt32(1)

const PROPAQ_DEPENDS_KEY = UInt32(1) << 0

mutable struct Ctx
    threshold::Float64
    seed::UInt64
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

function splitmix64(x::UInt64)::UInt64
    x += 0x9E3779B97F4A7C15
    z = x
    z = xor(z, z >> 30) * 0xBF58476D1CE4E5B9
    z = xor(z, z >> 27) * 0x94D049BB133111EB
    return xor(z, z >> 31)
end

# Uniform double in [0, 1) from the top 53 bits, the standard construction.
unit_interval(bits::UInt64)::Float64 = Float64(bits >> 11) * (1.0 / 9007199254740992.0) # 1 / 2^53

# A stream position derived from the term itself rather than from call order.
function key_stream(seed::UInt64, words, n_words::Integer)::UInt64
    h = seed
    @inbounds for i in 1:n_words
        h = splitmix64(xor(h, words[i]))
    end
    return h
end

function keep(threshold::Float64, seed::UInt64, words, n_words::Integer,
              coeff_magnitude::Cdouble)::Bool
    probability = coeff_magnitude / threshold
    probability >= 1.0 && return true
    draw = unit_interval(splitmix64(key_stream(seed, words, n_words)))
    return draw < probability
end

Base.@ccallable function propaq_truncator_abi_version()::UInt32
    PROPAQ_TRUNCATOR_ABI_VERSION
end

Base.@ccallable function propaq_truncator_depends()::UInt32
    PROPAQ_DEPENDS_KEY
end

Base.@ccallable function propaq_truncator_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "threshold", 1e-6), UInt64(parse_field(json, "seed", 42.0)))
    key = pointer_from_objref(ctx)
    lock(CTX_LOCK) do
        CTX_STORE[key] = ctx
    end
    return key
end

Base.@ccallable function propaq_truncator_destroy(ctx::Ptr{Cvoid})::Cvoid
    lock(CTX_LOCK) do
        delete!(CTX_STORE, ctx)
    end
    return nothing
end

Base.@ccallable function propaq_truncator_keep(ctx::Ptr{Cvoid}, basis_kind::UInt32,
                                               words::Ptr{UInt64}, n_words::Csize_t,
                                               n_units::UInt32, weight::UInt32,
                                               coeff_magnitude::Cdouble,
                                               layer_index::UInt32, n_layers::UInt32)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    w = unsafe_wrap(Array, words, n_words)
    return keep(c.threshold, c.seed, w, n_words, coeff_magnitude) ? Int32(1) : Int32(0)
end

Base.@ccallable function propaq_truncator_keep_batch(ctx::Ptr{Cvoid}, basis_kind::UInt32,
                                                     words::Ptr{UInt64}, n_words_per_term::Csize_t,
                                                     n_units::UInt32, weights::Ptr{UInt32},
                                                     coeff_magnitudes::Ptr{Cdouble},
                                                     layer_index::UInt32, n_layers::UInt32,
                                                     out_keep::Ptr{UInt8}, n_terms::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    all_words = unsafe_wrap(Array, words, n_words_per_term * n_terms)
    coeffs = unsafe_wrap(Array, coeff_magnitudes, n_terms)
    result = unsafe_wrap(Array, out_keep, n_terms)
    @inbounds for i in 1:n_terms
        base = (i - 1) * n_words_per_term
        term = view(all_words, base+1:base+n_words_per_term)
        result[i] = keep(c.threshold, c.seed, term, n_words_per_term, coeffs[i]) ? UInt8(1) : UInt8(0)
    end
    return Int32(0)
end
