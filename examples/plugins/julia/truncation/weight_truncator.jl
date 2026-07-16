# Example propaq native truncator plugin, implemented in Julia.

const PROPAQ_TRUNCATOR_ABI_VERSION = UInt32(1)

const MAX_WEIGHT = UInt32(4)

Base.@ccallable function propaq_truncator_abi_version()::UInt32
    PROPAQ_TRUNCATOR_ABI_VERSION
end

Base.@ccallable function propaq_truncator_keep(ctx::Ptr{Cvoid}, term_weight::UInt32,
                                                coeff_magnitude::Cdouble, active_modes::UInt32)::Int32
    return term_weight <= MAX_WEIGHT ? Int32(1) : Int32(0)
end

Base.@ccallable function propaq_truncator_keep_batch(ctx::Ptr{Cvoid}, term_weights::Ptr{UInt32},
                                                      coeff_magnitudes::Ptr{Cdouble}, active_modes::Ptr{UInt32},
                                                      out_keep::Ptr{UInt8}, n::Csize_t)::Int32
    weights = unsafe_wrap(Array, term_weights, n)
    result = unsafe_wrap(Array, out_keep, n)
    @inbounds for i in 1:n
        result[i] = weights[i] <= MAX_WEIGHT ? UInt8(1) : UInt8(0)
    end
    return Int32(0)
end
