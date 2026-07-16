# Example propaq native noise plugin, implemented in Julia (AOT-compiled via
# PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Non-stationary noise: damping grows with the number of calls already made
# through this ctx, modeling drift/heating that worsens over circuit depth
# rather than staying fixed like UniformNoiseModel.
#
#   damping_factor = exp(-damping * weight * (1 + drift_rate * call_index))

const PROPAQ_NOISE_ABI_VERSION = UInt32(1)

mutable struct Ctx
    damping::Float64
    drift_rate::Float64
    call_index::Threads.Atomic{UInt64}
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

function damping_factor(damping::Float64, drift_rate::Float64, term_weight::UInt32, call_index::UInt64)
    factor = 1.0 + drift_rate * Float64(call_index)
    return exp(-damping * Float64(term_weight) * factor)
end

Base.@ccallable function propaq_noise_abi_version()::UInt32
    PROPAQ_NOISE_ABI_VERSION
end

Base.@ccallable function propaq_noise_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "damping", 0.001), parse_field(json, "drift_rate", 1e-5), Threads.Atomic{UInt64}(0))
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

Base.@ccallable function propaq_noise_damping_factor(ctx::Ptr{Cvoid}, term_weight::UInt32, active_modes::UInt32)::Cdouble
    c = unsafe_pointer_to_objref(ctx)::Ctx
    call_index = Threads.atomic_add!(c.call_index, UInt64(1))
    damping_factor(c.damping, c.drift_rate, term_weight, call_index)
end

Base.@ccallable function propaq_noise_damping_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                     active_modes::Ptr{UInt32}, out::Ptr{Cdouble}, n::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    # Reserve the whole chunk's worth of call indices with one atomic op
    # instead of one fetch-add per term.
    base = Threads.atomic_add!(c.call_index, UInt64(n))
    weights = unsafe_wrap(Array, term_weights, n)
    result = unsafe_wrap(Array, out, n)
    @inbounds for i in 1:n
        result[i] = damping_factor(c.damping, c.drift_rate, weights[i], base + UInt64(i - 1))
    end
    return Int32(0)
end
