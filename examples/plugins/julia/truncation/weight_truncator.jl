# Example propaq truncator plugin, implemented in Julia (AOT-compiled via
# PackageCompiler.jl into a C-ABI shared library; see the plugins README).
#
# Weight alone, so it declares no dependencies
#

const PROPAQ_TRUNCATOR_ABI_VERSION = UInt32(1)

mutable struct Ctx
    max_weight::UInt32
end

const CTX_STORE = Dict{Ptr{Cvoid},Ctx}()
const CTX_LOCK = ReentrantLock()

function parse_field(json::AbstractString, key::AbstractString, fallback::Float64)
    m = match(Regex("\"" * key * "\"\\s*:\\s*(-?[0-9.eE+-]+)"), json)
    m === nothing ? fallback : parse(Float64, m.captures[1])
end

Base.@ccallable function propaq_truncator_abi_version()::UInt32
    PROPAQ_TRUNCATOR_ABI_VERSION
end

Base.@ccallable function propaq_truncator_create(config_json::Ptr{Cchar})::Ptr{Cvoid}
    json = config_json == C_NULL ? "" : unsafe_string(config_json)
    ctx = Ctx(UInt32(parse_field(json, "max_weight", 4.0)))
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
    return weight <= c.max_weight ? Int32(1) : Int32(0)
end

Base.@ccallable function propaq_truncator_keep_batch(ctx::Ptr{Cvoid}, basis_kind::UInt32,
                                                     words::Ptr{UInt64}, n_words_per_term::Csize_t,
                                                     n_units::UInt32, weights::Ptr{UInt32},
                                                     coeff_magnitudes::Ptr{Cdouble},
                                                     layer_index::UInt32, n_layers::UInt32,
                                                     out_keep::Ptr{UInt8}, n_terms::Csize_t)::Int32
    c = unsafe_pointer_to_objref(ctx)::Ctx
    w = unsafe_wrap(Array, weights, n_terms)
    result = unsafe_wrap(Array, out_keep, n_terms)
    @inbounds for i in 1:n_terms
        result[i] = w[i] <= c.max_weight ? UInt8(1) : UInt8(0)
    end
    return Int32(0)
end
