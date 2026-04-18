# Device abstraction for CPU/GPU dispatch.
# GPU() methods are stubs that get overridden by ext/ContourDynamicsCUDAExt.jl.

using Adapt
using KernelAbstractions

abstract type AbstractDevice end

"""
    CPU()

Select CPU execution. This is the default device.
"""
struct CPU <: AbstractDevice end

"""
    GPU()

Select GPU execution. Requires `using CUDA` to activate the CUDA extension.
"""
struct GPU <: AbstractDevice end

"""
    device_array(dev)

Return the array constructor for the given device: `Array` for `CPU()`,
`CuArray` for `GPU()` (when CUDA extension is loaded).
"""
device_array(::CPU) = Array
device_array(::GPU) = error(
    "GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Allocate a zero-filled array on the given device."""
device_zeros(::CPU, ::Type{T}, dims...) where {T} = zeros(T, dims...)
device_zeros(::GPU, ::Type{T}, dims...) where {T} = error(
    "GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Transfer data to CPU. No-op for CPU arrays."""
to_cpu(x::Array) = x
to_cpu(x) = adapt(Array, x)

"""Transfer data to the given device. No-op for CPU."""
to_device(::CPU, x) = x
to_device(::GPU, x) = error(
    "GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Return the KernelAbstractions backend for the given device."""
_ka_backend(::CPU) = KernelAbstractions.CPU()
_ka_backend(::GPU) = error(
    "GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")
