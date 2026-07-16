# Example propaq native noise plugin, implemented in Julia (AOT-compiled via
# PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Stretched-exponential relaxation: damping_factor = exp(-(gamma * weight)^beta).
# beta = 1 reduces exactly to UniformNoiseModel's plain exponential; beta != 1
# gives a different decay shape (sub-/super-exponential in weight), which is
# not expressible by UniformNoiseModel's single-parameter formula.
#

const PROPAQ_NOISE_ABI_VERSION = UInt32(1)

mutable struct Ctx
    gamma::Float64
    beta::Float64
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

damping_factor(gamma::Float64, beta::Float64, term_weight::UInt32) =
    exp(-(gamma * Float64(term_weight))^beta)

Base.@ccallable function propaq_noise_abi_version()::UInt32
    PROPAQ_NOISE_ABI_VERSION
end

Base.@ccallable function propaq_noise_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(parse_field(json, "gamma", 0.001), parse_field(json, "beta", 1.0))
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
    damping_factor(c.gamma, c.beta, term_weight)
end

Base.@ccallable function propaq_noise_damping_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                     active_modes::Ptr{UInt32}, out::Ptr{Cdouble}, n::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    weights = unsafe_wrap(Array, term_weights, n)
    result = unsafe_wrap(Array, out, n)
    @inbounds for i in 1:n
        result[i] = damping_factor(c.gamma, c.beta, weights[i])
    end
    return Int32(0)
end
