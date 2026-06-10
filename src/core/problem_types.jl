# ── Problem Structs ──────────────────────────────────────

mutable struct _VelocityScratch{T<:AbstractFloat}
    contour_curvatures::Vector{Vector{T}}
    reference_curvatures::Vector{Vector{T}}
    layer_curvatures::Vector{Vector{Vector{T}}}
    offsets::Vector{Int}
    target_nodes::Vector{SVector{2,T}}
    mode_vel::Vector{SVector{2,T}}
    modal_matrix::Matrix{T}
    modal_matrix_inv::Matrix{T}
    energy_partial::Vector{T}
end

function _VelocityScratch{T}() where {T<:AbstractFloat}
    return _VelocityScratch{T}(Vector{T}[],
                               Vector{T}[],
                               Vector{Vector{T}}[],
                               Int[],
                               SVector{2,T}[],
                               SVector{2,T}[],
                               Matrix{T}(undef, 0, 0),
                               Matrix{T}(undef, 0, 0),
                               T[])
end

"""
    ContourProblem{K,D,T,Dev}(kernel, domain, contours; dev=CPU())

A single-layer contour-dynamics problem with a velocity `kernel`, computational
`domain`, a vector of [`PVContour`](@ref)s, and a target `dev`ice
([`CPU`](@ref) or [`GPU`](@ref)).
"""
struct ContourProblem{K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice, S}
    kernel::K
    domain::D
    contours::Vector{PVContour{T}}
    dev::Dev
    device_state::S
    velocity_scratch::_VelocityScratch{T}
    function ContourProblem(kernel::K, domain::D, contours::Vector{PVContour{T}};
                            dev::Dev=CPU()) where {K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice}
        _check_kernel_type(kernel, T)
        _check_domain_type(domain, T)
        _check_kernel_domain(kernel, domain)
        _check_gpu_support(kernel, domain, dev)
        state = _build_device_state(contours, dev)
        new{K, D, T, Dev, typeof(state)}(kernel, domain, contours, dev, state,
                                          _VelocityScratch{T}())
    end
end

"""Return the single-layer contour vector stored in a CPU problem."""
contours(prob::ContourProblem{<:Any, <:Any, <:Any, CPU}) = prob.contours
contours(prob::ContourProblem{<:Any, <:Any, <:Any, GPU}) = error(
    "GPU() problems keep contours in device state. Use materialize_contours(prob) " *
    "only at output, animation, plotting, or host-inspection boundaries.")
contours(prob::ContourProblem) = prob.contours

"""Materialize contours on CPU for output, animation, plotting, or inspection."""
materialize_contours(prob::ContourProblem{<:Any, <:Any, <:Any, CPU}) = prob.contours
materialize_contours(prob::ContourProblem{<:Any, <:Any, <:Any, GPU}) =
    materialize_contours(prob.device_state)

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
_check_gpu_support(kernel, domain, ::GPU) = throw(ArgumentError(
    "GPU velocity is supported for single-layer EulerKernel, QGKernel, and SQGKernel " *
    "on UnboundedDomain or PeriodicDomain. " *
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
    layers::NTuple{N, Vector{PVContour{T}}}
    dev::Dev
    device_state::S
    velocity_scratch::_VelocityScratch{T}
    function MultiLayerContourProblem(kernel::K, domain::D, layers::NTuple{N, Vector{PVContour{T}}};
                                      dev::Dev=CPU()) where {N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat, Dev<:AbstractDevice}
        _check_kernel_type(kernel, T)
        _check_domain_type(domain, T)
        state = _build_layer_device_state(layers, dev)
        new{N, K, D, T, Dev, typeof(state)}(kernel, domain, layers, dev, state,
                                             _VelocityScratch{T}())
    end
end

"""Return the per-layer contour tuple stored in a CPU problem."""
contours(prob::MultiLayerContourProblem{<:Any, <:Any, <:Any, <:Any, CPU}) = prob.layers
contours(prob::MultiLayerContourProblem{<:Any, <:Any, <:Any, <:Any, GPU}) = error(
    "GPU() problems keep contours in device state. Use materialize_contours(prob) " *
    "only at output, animation, plotting, or host-inspection boundaries.")
contours(prob::MultiLayerContourProblem) = prob.layers

"""Materialize per-layer contours on CPU for output, animation, plotting, or inspection."""
materialize_contours(prob::MultiLayerContourProblem{<:Any, <:Any, <:Any, <:Any, CPU}) = prob.layers
materialize_contours(prob::MultiLayerContourProblem{N, <:Any, <:Any, <:Any, GPU}) where {N} =
    ntuple(i -> materialize_contours(prob.device_state[i]), N)

"""Return the number of layers in a multi-layer contour problem."""
nlayers(::MultiLayerContourProblem{N}) where {N} = N

"""
    total_nodes(prob)

Total number of nodes across all contours in a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
@inline function total_nodes(prob::ContourProblem)
    s = 0
    for c in prob.contours
        s += nnodes(c)
    end
    return s
end

@inline total_nodes(prob::ContourProblem{<:Any, <:Any, <:Any, GPU}) =
    _device_state_nnodes(prob.device_state)

@inline function total_nodes(prob::MultiLayerContourProblem{N}) where {N}
    s = 0
    for i in 1:N
        for c in prob.layers[i]
            s += nnodes(c)
        end
    end
    return s
end

@inline function total_nodes(prob::MultiLayerContourProblem{N, <:Any, <:Any, <:Any, GPU}) where {N}
    s = 0
    for i in 1:N
        s += _device_state_nnodes(prob.device_state[i])
    end
    return s
end
