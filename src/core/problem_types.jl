# ── Problem Structs ──────────────────────────────────────

"""
    ContourProblem{K,D,T,Dev}(kernel, domain, contours; dev=CPU())

A single-layer contour-dynamics problem with a velocity `kernel`, computational
`domain`, a vector of [`PVContour`](@ref)s, and a target `dev`ice
([`CPU`](@ref) or [`GPU`](@ref)).
"""
struct ContourProblem{K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice, S}
    kernel::K
    domain::D
    storage::S
    dev::Dev
    workspace::ExecutionWorkspace{T}
    function ContourProblem(kernel::K, domain::D, contours::Vector{PVContour{T}};
                            dev::Dev=CPU(), workspace::ExecutionWorkspace{T}=ExecutionWorkspace(T)) where {K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice}
        _check_kernel_type(kernel, T)
        _check_domain_type(domain, T)
        _check_kernel_domain(kernel, domain)
        _check_gpu_support(kernel, domain, dev)
        storage = _storage(contours, dev)
        new{K, D, T, Dev, typeof(storage)}(kernel, domain, storage, dev, workspace)
    end
end

# Validation helpers keep constructor errors close to the user input. They are
# intentionally small single-dispatch functions so adding a new kernel/domain or
# device support path only requires adding methods here.
_check_gpu_support(::AbstractKernel, ::AbstractDomain, ::CPU) = nothing
_check_gpu_support(::EulerKernel, ::UnboundedDomain, ::GPU) = nothing
_check_gpu_support(::QGKernel, ::UnboundedDomain, ::GPU) = nothing
_check_gpu_support(::EulerKernel, ::PeriodicDomain, ::GPU) = nothing
_check_gpu_support(::QGKernel, ::PeriodicDomain, ::GPU) = nothing
_check_gpu_support(::SQGKernel, ::UnboundedDomain, ::GPU) = nothing
_check_gpu_support(::SQGKernel, ::PeriodicDomain, ::GPU) = nothing
_check_gpu_support(::BetaPlaneQGKernel, ::PeriodicDomain, ::GPU) = nothing
_check_gpu_support(kernel, domain, ::GPU) = throw(ArgumentError(
    "GPU velocity is supported for single-layer EulerKernel, QGKernel, and SQGKernel " *
    "on UnboundedDomain or PeriodicDomain, and BetaPlaneQGKernel on PeriodicDomain. " *
    "Got $(typeof(kernel)) on $(typeof(domain)). Use dev=CPU()."))

_check_kernel_type(::AbstractKernel, ::Type) = nothing
_check_kernel_type(::QGKernel{Tk}, ::Type{T}) where {Tk, T} =
    Tk !== T && throw(ArgumentError("QGKernel uses $Tk but contours use $T — construct the kernel with the same float type as the contours"))
_check_kernel_type(::BetaPlaneQGKernel{Tk}, ::Type{T}) where {Tk, T} =
    Tk !== T && throw(ArgumentError("BetaPlaneQGKernel uses $Tk but contours use $T — construct the kernel with the same float type as the contours"))
_check_kernel_type(::SQGKernel{Tk}, ::Type{T}) where {Tk, T} =
    Tk !== T && throw(ArgumentError("SQGKernel uses $Tk but contours use $T — construct the kernel with the same float type as the contours"))
_check_kernel_type(::MultiLayerQGKernel{N,M,Tk}, ::Type{T}) where {N,M,Tk,T} =
    Tk !== T && throw(ArgumentError("MultiLayerQGKernel uses $Tk but contours use $T — construct the kernel with the same float type as the contours"))

_check_domain_type(::AbstractDomain, ::Type) = nothing
_check_domain_type(::PeriodicDomain{Td}, ::Type{T}) where {Td, T} =
    Td !== T && throw(ArgumentError("PeriodicDomain uses $Td but contours use $T — construct the domain with the same float type as the contours"))

_check_kernel_domain(::AbstractKernel, ::AbstractDomain) = nothing
_check_kernel_domain(kernel::BetaPlaneQGKernel, domain::AbstractDomain) =
    _validate_beta_plane_reference(kernel, domain)

"""
    MultiLayerContourProblem{N,K,D,T,Dev}(kernel, domain, layers; dev=CPU())

An `N`-layer contour-dynamics problem.  Each element of the `layers` tuple
holds the contours for one layer.  The optional `dev` keyword selects the
target device ([`CPU`](@ref) or [`GPU`](@ref)) for buffer allocation.
"""
struct MultiLayerContourProblem{N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice, S}
    kernel::K
    domain::D
    storage::S
    dev::Dev
    workspace::ExecutionWorkspace{T}
    function MultiLayerContourProblem(kernel::K, domain::D, layers::NTuple{N, Vector{PVContour{T}}};
                                      dev::Dev=CPU(), workspace::ExecutionWorkspace{T}=ExecutionWorkspace(T)) where {N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice}
        _check_kernel_type(kernel, T)
        _check_domain_type(domain, T)
        storage = _storage(layers, dev)
        new{N, K, D, T, Dev, typeof(storage)}(kernel, domain, storage, dev, workspace)
    end
end

"""Return the number of layers in a multi-layer contour problem."""
nlayers(::MultiLayerContourProblem{N}) where {N} = N

"""
    total_nodes(prob)

Total number of nodes across all contours in a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
@inline function total_nodes(prob::ContourProblem)
    s = 0
    for c in _host_contours(prob)
        s += nnodes(c)
    end
    return s
end

@inline total_nodes(prob::ContourProblem{K,D,T,GPU,S}) where {
    K<:AbstractKernel,D<:AbstractDomain,T<:AbstractFloat,S
} =
    _device_state_nnodes(_device_state(prob))

@inline function total_nodes(prob::MultiLayerContourProblem{N}) where {N}
    s = 0
    for i in 1:N
        for c in _host_contours(prob)[i]
            s += nnodes(c)
        end
    end
    return s
end

@inline function total_nodes(prob::MultiLayerContourProblem{N,K,D,T,GPU,S}) where {
    N,K<:MultiLayerQGKernel{N},D<:AbstractDomain,T<:AbstractFloat,S
}
    s = 0
    for i in 1:N
        s += _device_state_nnodes(_device_state(prob)[i])
    end
    return s
end

const _ContourProblemTypes = Union{ContourProblem,MultiLayerContourProblem}
_active_storage(prob::_ContourProblemTypes) = getfield(prob, :storage)
_host_contours(prob::_ContourProblemTypes) = _borrow_contours(_active_storage(prob))
_device_state(prob::_ContourProblemTypes) = _device_storage(_active_storage(prob))

"""Borrow live CPU contours (or layer tuple). Mutations affect the problem."""
contours(prob::_ContourProblemTypes) = _host_contours(prob)

"""
    snapshot_contours(prob)

Return an owned CPU copy of current contours, including nodes and corner flags.
The result never aliases the live state, on either backend. Multi-layer problems
return a tuple of vectors. Use `contours(prob)` to explicitly borrow CPU state.
"""
snapshot_contours(prob::_ContourProblemTypes) = _snapshot_storage(_active_storage(prob))

"""
    materialize_contours(prob)

Legacy output accessor: borrows CPU contours and copies GPU contours. Use
`contours` for an explicit borrow or `snapshot_contours` for a stable owned copy.
"""
materialize_contours(prob::_ContourProblemTypes) = _materialize_storage(_active_storage(prob))
execution_workspace(prob::_ContourProblemTypes) = getfield(prob, :workspace)
clear_state_workspace_cache!(prob::_ContourProblemTypes) = clear_state_workspace_cache!(execution_workspace(prob))

# Preserve field-style inspection without keeping a stale host mirror on GPU.
# GPU .contours/.layers reads now materialize the active state; internal host
# algorithms use _host_contours and therefore reject accidental device access.
@inline function Base.getproperty(prob::ContourProblem, name::Symbol)
    name === :contours && return materialize_contours(prob)
    name === :device_state && return _device_state(prob)
    name === :velocity_scratch && return getfield(prob, :workspace).cpu
    return getfield(prob, name)
end
@inline function Base.getproperty(prob::MultiLayerContourProblem, name::Symbol)
    name === :layers && return materialize_contours(prob)
    name === :device_state && return _device_state(prob)
    name === :velocity_scratch && return getfield(prob, :workspace).cpu
    return getfield(prob, name)
end
Base.propertynames(::ContourProblem, private::Bool=false) =
    (:kernel, :domain, :contours, :dev, :device_state, :velocity_scratch, :storage, :workspace)
Base.propertynames(::MultiLayerContourProblem, private::Bool=false) =
    (:kernel, :domain, :layers, :dev, :device_state, :velocity_scratch, :storage, :workspace)
