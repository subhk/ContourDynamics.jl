module ContourDynamicsCUDAExt

using ContourDynamics
using CUDA
using Adapt
using KernelAbstractions

# Wire GPU() to CuArray
ContourDynamics.device_array(::ContourDynamics.GPU) = CuArray

ContourDynamics.device_zeros(::ContourDynamics.GPU, ::Type{T}, dims...) where {T} =
    CUDA.zeros(T, dims...)

ContourDynamics.to_device(::ContourDynamics.GPU, x) = adapt(CuArray, x)

ContourDynamics._ka_backend(::ContourDynamics.GPU) = CUDABackend()

# Adapt.jl integration for ContourProblem
function Adapt.adapt_structure(to, prob::ContourDynamics.ContourProblem)
    new_dev = to <: CUDA.CuArray ? ContourDynamics.GPU() : ContourDynamics.CPU()
    ContourDynamics.ContourProblem(prob.kernel, prob.domain, prob.contours; dev=new_dev)
end

end # module
