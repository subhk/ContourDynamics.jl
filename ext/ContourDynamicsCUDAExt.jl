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

# Adapt.jl integration for ContourProblem and MultiLayerContourProblem.
# Contour nodes stay on CPU (packed to GPU on each velocity evaluation).
# adapt_structure only switches the device tag for dispatch.
function Adapt.adapt_structure(to, prob::ContourDynamics.ContourProblem)
    new_dev = _detect_device(to)
    ContourDynamics.ContourProblem(prob.kernel, prob.domain, prob.contours; dev=new_dev)
end

function Adapt.adapt_structure(to, prob::ContourDynamics.MultiLayerContourProblem)
    new_dev = _detect_device(to)
    ContourDynamics.MultiLayerContourProblem(prob.kernel, prob.domain, prob.layers; dev=new_dev)
end

_detect_device(::CUDA.CuArrayAdaptor) = ContourDynamics.GPU()
_detect_device(::Type{T}) where {T<:CuArray} = ContourDynamics.GPU()
_detect_device(::Type{T}) where {T<:Array} = ContourDynamics.CPU()
_detect_device(::Any) = ContourDynamics.CPU()

end # module
