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

# Kernel launches are not synchronized individually (see `@_ka_launch`), so a
# host read must first drain the stream. `Array(::CuArray)` already waits for
# an unpinned destination; the explicit synchronize makes that contract
# independent of CUDA.jl's copy implementation.
function ContourDynamics.to_cpu(x::CuArray)
    CUDA.synchronize()
    return Array(x)
end

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

# The Problem wrapper moves its contour problem and rebuilds the stepper's
# buffers on the target device (Problem requires both on one device).
function Adapt.adapt_structure(to, prob::ContourDynamics.Problem)
    contour_problem = Adapt.adapt_structure(to, prob.contour_problem)
    stepper = prob.stepper
    stepper isa ContourDynamics.RK4Stepper || throw(ArgumentError(
        "cannot move a Problem with a $(typeof(stepper)) to another device"))
    new_stepper = ContourDynamics.RK4Stepper(
        stepper.dt, ContourDynamics.total_nodes(contour_problem); dev=contour_problem.dev)
    return ContourDynamics.Problem(contour_problem, new_stepper, prob.surgery_params)
end

# Detect GPU from Adapt.jl adaptors. `cu` adapts with `CuArrayAdaptor` in
# CUDA.jl 5.0 and `CuArrayKernelAdaptor` from 5.1 on; register whichever exist.
for adaptor in (:CuArrayAdaptor, :CuArrayKernelAdaptor)
    if isdefined(CUDA, adaptor)
        @eval _detect_device(::CUDA.$adaptor) = ContourDynamics.GPU()
    end
end
_detect_device(::Type{T}) where {T<:CuArray} = ContourDynamics.GPU()
_detect_device(::Type{T}) where {T<:Array} = ContourDynamics.CPU()
_detect_device(::Any) = ContourDynamics.CPU()

end # module
