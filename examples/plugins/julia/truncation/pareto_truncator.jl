# Example propaq native truncator plugin, implemented in Julia (AOT-compiled
# via PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Joint weight/coefficient score instead of two independent hard cutoffs:
# composing WeightTruncator and CoefficientTruncator ANDs two separate
# thresholds, which can't express a smooth tradeoff between them. This keeps
# a term if its coefficient magnitude, discounted by an exponential in its
# weight, clears a single threshold:
#
#   keep <=> coeff_magnitude * exp(-alpha * weight) > threshold
#
# alpha = 0 reduces to a plain coefficient cutoff.
#

const PROPAQ_TRUNCATOR_ABI_VERSION = UInt32(1)

mutable struct Ctx
    threshold::Float64
    alpha::Float64
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

function keep(threshold::Float64, alpha::Float64, term_weight::UInt32, coeff_magnitude::Cdouble)::Bool
    score = coeff_magnitude * exp(-alpha * Float64(term_weight))
    return score > threshold
end

Base.@ccallable function propaq_truncator_abi_version()::UInt32
    PROPAQ_TRUNCATOR_ABI_VERSION
end

Base.@ccallable function propaq_truncator_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "threshold", 1e-6), parse_field(json, "alpha", 0.1))
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
    return keep(c.threshold, c.alpha, term_weight, coeff_magnitude) ? Int32(1) : Int32(0)
end

Base.@ccallable function propaq_truncator_keep_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                      coeff_magnitudes::Ptr{Cdouble}, active_modes::Ptr{UInt32},
                                                      out_keep::Ptr{UInt8}, n::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    weights = unsafe_wrap(Array, term_weights, n)
    coeffs = unsafe_wrap(Array, coeff_magnitudes, n)
    result = unsafe_wrap(Array, out_keep, n)
    @inbounds for i in 1:n
        result[i] = keep(c.threshold, c.alpha, weights[i], coeffs[i]) ? UInt8(1) : UInt8(0)
    end
    return Int32(0)
end
