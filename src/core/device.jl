# Device abstraction for CPU/GPU dispatch.
# GPU() methods are stubs that get overridden by ext/ContourDynamicsCUDAExt.jl.

using Adapt
using KernelAbstractions

"""
    AbstractDevice

Supertype for execution-device tags used to choose storage and
KernelAbstractions backends. Concrete public devices are [`CPU`](@ref) and
[`GPU`](@ref).
"""
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
device_array(::AbstractDevice) = error("GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Allocate a zero-filled array on the given device."""
device_zeros(::CPU, ::Type{T}, dims...) where {T} = zeros(T, dims...)
device_zeros(::AbstractDevice, ::Type{T}, dims...) where {T} = error("GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Transfer data to CPU. No-op for CPU arrays."""
to_cpu(x::Array) = x
to_cpu(x) = adapt(Array, x)

"""Transfer data to the given device. No-op for CPU."""
to_device(::CPU, x) = x
to_device(::AbstractDevice, x) = error("GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""Return the KernelAbstractions backend for the given device."""
_ka_backend(::CPU) = KernelAbstractions.CPU()
_ka_backend(::AbstractDevice) = error("GPU support requires CUDA.jl. Load it with `using CUDA` before using GPU().")

"""
    @_ka_launch dev ndrange kernel_builder(args...)

Build the KernelAbstractions backend for `dev`, instantiate `kernel_builder`,
launch it with `ndrange`, and return `nothing`.

The launch is not synchronized: successive launches on one backend are
stream-ordered, so device-to-device work may be queued back to back. Host
reads go through [`to_cpu`](@ref) or [`_device_synchronize`](@ref), which wait
for the queued work. (The KA CPU backend runs each launch to completion.)

This is intentionally small: it only removes the repeated KA launch boilerplate
while leaving data layout, bounds, and topology rules explicit at each call site.
"""
macro _ka_launch(dev, ndrange, call)
    @assert call isa Expr && call.head === :call
    backend = gensym(:backend)
    kernel = gensym(:kernel)
    builder = call.args[1]
    args = call.args[2:end]
    return quote
        local $backend = _ka_backend($(esc(dev)))
        local $kernel = $(esc(builder))($backend)
        $kernel($(map(esc, args)...); ndrange=$(esc(ndrange)))
        nothing
    end
end

"""
    _device_synchronize(dev)

Wait for all queued kernel launches on `dev` to finish. Call it before reading
device results into host memory by a route other than [`to_cpu`](@ref), e.g. a
`copyto!` into a preallocated host buffer.
"""
_device_synchronize(dev::AbstractDevice) = KernelAbstractions.synchronize(_ka_backend(dev))
