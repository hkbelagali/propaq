# Example propaq native noise plugin, implemented in Julia.

const PROPAQ_NOISE_ABI_VERSION = UInt32(1)

# This example ignores `config_json` and hardcodes damping for brevity
const DAMPING = 0.001

Base.@ccallable function propaq_noise_abi_version()::UInt32
    PROPAQ_NOISE_ABI_VERSION
end

Base.@ccallable function propaq_noise_damping_factor(ctx::Ptr{Cvoid}, term_weight::UInt32, active_modes::UInt32)::Cdouble
    exp(-DAMPING * Float64(term_weight))
end

Base.@ccallable function propaq_noise_damping_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                     active_modes::Ptr{UInt32}, out::Ptr{Cdouble}, n::Csize_t)::Int32
    weights = unsafe_wrap(Array, term_weights, n)
    result = unsafe_wrap(Array, out, n)
    @inbounds for i in 1:n
        result[i] = exp(-DAMPING * Float64(weights[i]))
    end
    return Int32(0)
end
