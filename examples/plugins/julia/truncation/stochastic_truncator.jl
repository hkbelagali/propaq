# Example propaq native truncator plugin, implemented in Julia (AOT-compiled
# via PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Importance-sampling truncation: instead of a deterministic cutoff, keep
# each term with probability min(1, coeff_magnitude / threshold). Small
# coefficients are discarded most (but not all) of the time rather than
# always, which trades a small amount of variance for keeping the estimator
# unbiased in expectation.

const PROPAQ_TRUNCATOR_ABI_VERSION = UInt32(1)

mutable struct Ctx
    threshold::Float64
    seed::UInt64
    call_index::Threads.Atomic{UInt64}
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
    z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
    return z ⊻ (z >> 31)
end

# Uniform double in [0, 1) from the top 53 bits, the standard construction.
unit_interval(bits::UInt64)::Float64 = Float64(bits >> 11) * (1.0 / 9007199254740992.0) # 1 / 2^53

function keep(threshold::Float64, seed::UInt64, coeff_magnitude::Cdouble, call_index::UInt64)::Bool
    probability = coeff_magnitude / threshold
    probability >= 1.0 && return true
    draw = unit_interval(splitmix64(seed ⊻ call_index))
    return draw < probability
end

Base.@ccallable function propaq_truncator_abi_version()::UInt32
    PROPAQ_TRUNCATOR_ABI_VERSION
end

Base.@ccallable function propaq_truncator_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "threshold", 1e-6), UInt64(parse_field(json, "seed", 42.0)), Threads.Atomic{UInt64}(0))
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

Base.@ccallable function propaq_truncator_keep(ctx::Ptr{Cvoid}, term_weight::UInt32,
                                                coeff_magnitude::Cdouble, active_modes::UInt32)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    call_index = Threads.atomic_add!(c.call_index, UInt64(1))
    return keep(c.threshold, c.seed, coeff_magnitude, call_index) ? Int32(1) : Int32(0)
end

Base.@ccallable function propaq_truncator_keep_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                      coeff_magnitudes::Ptr{Cdouble}, active_modes::Ptr{UInt32},
                                                      out_keep::Ptr{UInt8}, n::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    base = Threads.atomic_add!(c.call_index, UInt64(n))
    coeffs = unsafe_wrap(Array, coeff_magnitudes, n)
    result = unsafe_wrap(Array, out_keep, n)
    @inbounds for i in 1:n
        result[i] = keep(c.threshold, c.seed, coeffs[i], base + UInt64(i - 1)) ? UInt8(1) : UInt8(0)
    end
    return Int32(0)
end
