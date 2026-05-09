# CUDA extension.
#
# Loading CUDA activates the GPU() device tag by replacing the CPU fallback
# stubs in the parent package with CuArray allocation, transfer, and
# KernelAbstractions backend methods. The core package keeps GPU() available as
# a lightweight tag; this extension supplies the concrete CUDA implementation.
module ContourDynamicsCUDAExt

using ContourDynamics
using CUDA
using Adapt
using KernelAbstractions

# Wire GPU() to CuArray storage and the CUDA KernelAbstractions backend.
ContourDynamics.device_array(::ContourDynamics.GPU) = CuArray

ContourDynamics.device_zeros(::ContourDynamics.GPU, ::Type{T}, dims...) where {T} =
    CUDA.zeros(T, dims...)

ContourDynamics.to_device(::ContourDynamics.GPU, x) = adapt(CuArray, x)

ContourDynamics._ka_backend(::ContourDynamics.GPU) = CUDABackend()

# Adapt.jl integration for ContourProblem and MultiLayerContourProblem.
# Reconstruct through the explicit materialization boundary so GPU device state
# is authoritative and stale host shadows are not adapted back by accident.
function Adapt.adapt_structure(to, prob::ContourDynamics.ContourProblem)
    new_dev = _detect_device(to)
    host_contours = ContourDynamics.materialize_contours(prob)
    ContourDynamics.ContourProblem(prob.kernel, prob.domain, host_contours; dev=new_dev)
end

function Adapt.adapt_structure(to, prob::ContourDynamics.MultiLayerContourProblem)
    new_dev = _detect_device(to)
    host_layers = ContourDynamics.materialize_contours(prob)
    ContourDynamics.MultiLayerContourProblem(prob.kernel, prob.domain, host_layers; dev=new_dev)
end

# Detect GPU from Adapt.jl adaptors.  CuArrayAdaptor may not exist in all
# CUDA.jl 5.x releases, so wrap in a try/catch to avoid load-time errors.
try
    @eval _detect_device(::CUDA.CuArrayAdaptor) = ContourDynamics.GPU()
catch
    # CuArrayAdaptor not available in this CUDA version; fall through to catch-all.
end
_detect_device(::Type{T}) where {T<:CuArray} = ContourDynamics.GPU()
_detect_device(::Type{T}) where {T<:Array} = ContourDynamics.CPU()
_detect_device(::Any) = ContourDynamics.CPU()

end # module
