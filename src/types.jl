# Note: StaticArrays and LinearAlgebra are imported in the parent module (ContourDynamics.jl).
# No `using` statements needed here since this file is `include`d.

# ── Kernels ──────────────────────────────────────────────

abstract type AbstractKernel end

struct EulerKernel <: AbstractKernel end

struct QGKernel{T<:AbstractFloat} <: AbstractKernel
    Ld::T
    function QGKernel(Ld::T) where {T<:AbstractFloat}
        Ld > zero(T) || throw(ArgumentError("Deformation radius Ld must be positive, got $Ld"))
        new{T}(Ld)
    end
end

struct MultiLayerQGKernel{N, M, T<:AbstractFloat} <: AbstractKernel
    Ld::SVector{M, T}
    coupling::SMatrix{N, N, T}
    H::SVector{N, T}
    function MultiLayerQGKernel(Ld::SVector{M, T}, coupling::SMatrix{N, N, T}, H::SVector{N, T}) where {N, M, T<:AbstractFloat}
        M == N - 1 || throw(ArgumentError("Number of deformation radii M=$M must equal N-1=$(N-1)"))
        all(>(zero(T)), H) || throw(ArgumentError("Layer thicknesses must be positive"))
        new{N, M, T}(Ld, coupling, H)
    end
end

nlayers(::MultiLayerQGKernel{N}) where {N} = N

# ── Contours ─────────────────────────────────────────────

struct PVContour{T<:AbstractFloat}
    nodes::Vector{SVector{2, T}}
    pv::T
end

nnodes(c::PVContour) = length(c.nodes)

# ── Domains ──────────────────────────────────────────────

abstract type AbstractDomain end

struct UnboundedDomain <: AbstractDomain end

struct PeriodicDomain{T<:AbstractFloat} <: AbstractDomain
    Lx::T
    Ly::T
    function PeriodicDomain(Lx::T, Ly::T) where {T<:AbstractFloat}
        Lx > zero(T) && Ly > zero(T) || throw(ArgumentError("Domain half-widths must be positive"))
        new{T}(Lx, Ly)
    end
end

# ── Problem Structs ──────────────────────────────────────

struct ContourProblem{K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat}
    kernel::K
    domain::D
    contours::Vector{PVContour{T}}
end

struct MultiLayerContourProblem{N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat}
    kernel::K
    domain::D
    layers::NTuple{N, Vector{PVContour{T}}}
end

nlayers(::MultiLayerContourProblem{N}) where {N} = N

# Total node count across all contours
function total_nodes(prob::ContourProblem)
    s = 0
    for c in prob.contours
        s += nnodes(c)
    end
    return s
end

function total_nodes(prob::MultiLayerContourProblem{N}) where {N}
    s = 0
    for i in 1:N
        for c in prob.layers[i]
            s += nnodes(c)
        end
    end
    return s
end

# ── Surgery Parameters ───────────────────────────────────

struct SurgeryParams{T<:AbstractFloat}
    delta::T
    mu::T
    Delta_max::T
    area_min::T
    n_surgery::Int
    function SurgeryParams(delta::T, mu::T, Delta_max::T, area_min::T, n_surgery::Int) where {T<:AbstractFloat}
        delta > zero(T) || throw(ArgumentError("delta must be positive"))
        mu > zero(T) || throw(ArgumentError("mu must be positive"))
        Delta_max > mu || throw(ArgumentError("Delta_max must be greater than mu"))
        area_min > zero(T) || throw(ArgumentError("area_min must be positive"))
        n_surgery > 0 || throw(ArgumentError("n_surgery must be positive"))
        new{T}(delta, mu, Delta_max, area_min, n_surgery)
    end
end

# ── Time Steppers ────────────────────────────────────────

abstract type AbstractTimeStepper end

struct RK4Stepper{T<:AbstractFloat} <: AbstractTimeStepper
    dt::T
    k1::Vector{SVector{2, T}}
    k2::Vector{SVector{2, T}}
    k3::Vector{SVector{2, T}}
    k4::Vector{SVector{2, T}}
end

function RK4Stepper(dt::T, n::Int) where {T<:AbstractFloat}
    z = zero(SVector{2, T})
    RK4Stepper(dt, fill(z, n), fill(z, n), fill(z, n), fill(z, n))
end

mutable struct LeapfrogStepper{T<:AbstractFloat} <: AbstractTimeStepper
    dt::T
    nodes_prev::Vector{SVector{2, T}}
    initialized::Bool
end

function LeapfrogStepper(dt::T, n::Int) where {T<:AbstractFloat}
    z = zero(SVector{2, T})
    LeapfrogStepper(dt, fill(z, n), false)
end
