# Note: StaticArrays and LinearAlgebra are imported in the parent module (ContourDynamics.jl).
# No `using` statements needed here since this file is `include`d.

# ── Kernels ──────────────────────────────────────────────

abstract type AbstractKernel end

"""
    EulerKernel <: AbstractKernel

Kernel for 2-D Euler (barotropic) vortex dynamics, using the Green's function
`G(r) = -log(r) / (2π)`.
"""
struct EulerKernel <: AbstractKernel end

"""
    QGKernel{T}(Ld)

Kernel for single-layer quasi-geostrophic dynamics with deformation radius `Ld`.
Uses the modified Bessel function `K₀(r/Ld)` as the Green's function.
"""
struct QGKernel{T<:AbstractFloat} <: AbstractKernel
    Ld::T
    function QGKernel(Ld::T) where {T<:AbstractFloat}
        Ld > zero(T) || throw(ArgumentError("Deformation radius Ld must be positive, got $Ld"))
        new{T}(Ld)
    end
end

"""
    SQGKernel{T}(delta)

Kernel for surface quasi-geostrophic (SQG) dynamics using the fractional
Laplacian Green's function `G(r) = -1/(2πr)`.

The SQG velocity at a contour boundary is singular (the tangential component
diverges logarithmically), so a **regularization length** `delta > 0` is
required.  The kernel `1/r` is replaced by `1/√(r² + δ²)`.  A typical choice
is `delta ≈ μ`, the minimum segment length used for surgery.

# Sign convention

Positive PV (surface buoyancy) induces counter-clockwise circulation, matching
the `EulerKernel` convention.
"""
struct SQGKernel{T<:AbstractFloat} <: AbstractKernel
    delta::T
    function SQGKernel(delta::T) where {T<:AbstractFloat}
        delta > zero(T) || throw(ArgumentError("Regularization δ must be positive, got $delta"))
        new{T}(delta)
    end
end

"""
    MultiLayerQGKernel{N,M,T}(Ld, coupling, H)

Kernel for `N`-layer quasi-geostrophic dynamics with `M = N-1` deformation radii `Ld`,
layer coupling matrix `coupling`, and layer thicknesses `H`.  The constructor
eigen-decomposes the coupling matrix for efficient velocity evaluation.

!!! note
    The evolution and energy diagnostics derive modal deformation radii from the
    coupling matrix eigenvalues (`Ld_mode = 1/√|λ|`), **not** from the provided `Ld`.
    `Ld` is stored for user reference and introspection but does not affect dynamics.
    Ensure that the coupling matrix is consistent with your intended deformation radii.
"""
struct MultiLayerQGKernel{N, M, T<:AbstractFloat} <: AbstractKernel
    Ld::SVector{M, T}
    coupling::SMatrix{N, N, T}
    H::SVector{N, T}
    eigenvalues::SVector{N, T}
    eigenvectors::SMatrix{N, N, T}
    eigenvectors_inv::SMatrix{N, N, T}
    function MultiLayerQGKernel(Ld::SVector{M, T}, coupling::SMatrix{N, N, T}, H::SVector{N, T}) where {N, M, T<:AbstractFloat}
        M == N - 1 || throw(ArgumentError("Number of deformation radii M=$M must equal N-1=$(N-1)"))
        all(>(zero(T)), Ld) || throw(ArgumentError("Deformation radii Ld must all be positive"))
        all(>(zero(T)), H) || throw(ArgumentError("Layer thicknesses must be positive"))
        cmat = Matrix(coupling)
        issymmetric(cmat) || throw(ArgumentError("Coupling matrix must be symmetric; got asymmetry ‖C-Cᵀ‖ = $(norm(cmat - cmat', Inf))"))
        eig = eigen(Symmetric(cmat))
        eigenvalues = SVector{N,T}(eig.values)
        eigenvectors = SMatrix{N,N,T}(eig.vectors)
        # eigen(Symmetric(...)) returns orthonormal eigenvectors, so P⁻¹ = Pᵀ.
        # Using transpose is both faster and more numerically stable than inv().
        eigenvectors_inv = SMatrix{N,N,T}(eig.vectors')

        # Physical validation: coupling eigenvalues should be non-positive
        # (the stretching operator is dissipative in the QG energy norm)
        for (m, λ) in enumerate(eigenvalues)
            if λ > eps(T) * T(100)
                @warn "MultiLayerQGKernel: coupling eigenvalue λ[$m] = $λ is positive; " *
                      "physical coupling matrices should have non-positive eigenvalues" maxlog=1
                break
            end
        end

        # Warn if Ld appears inconsistent with the modal radii the evolution
        # will actually use, so silent mis-specification is caught early.
        modal_radii = T[one(T) / sqrt(abs(λ)) for λ in eigenvalues if abs(λ) > eps(T) * T(100)]
        if length(modal_radii) == M
            sorted_modal = sort(modal_radii)
            sorted_Ld = sort(Vector(Ld))
            max_ratio = maximum(max(sorted_modal[m] / sorted_Ld[m],
                                    sorted_Ld[m] / sorted_modal[m]) for m in 1:M)
            if max_ratio > T(10)
                @warn "MultiLayerQGKernel: provided Ld = $(Vector(Ld)) differs substantially " *
                      "(ratio > 10) from modal radii derived from coupling eigenvalues: " *
                      "$(round.(modal_radii; digits=4)). Note: the evolution uses modal radii " *
                      "from eigenvalues; provided Ld is stored but not used in dynamics." maxlog=1
            end
        end

        new{N, M, T}(Ld, coupling, H, eigenvalues, eigenvectors, eigenvectors_inv)
    end
end

"""
    nlayers(kernel_or_problem)

Return the number of layers in a [`MultiLayerQGKernel`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
nlayers(::MultiLayerQGKernel{N}) where {N} = N

# ── Contours ─────────────────────────────────────────────

"""
    PVContour{T}(nodes, pv[, wrap])

A piecewise-linear contour carrying a potential-vorticity jump `pv`.
`nodes` is the ordered list of vertices.  For contours that span a periodic
domain, `wrap` gives the shift vector that closes the final segment back to
`nodes[1]`; it defaults to zero for ordinary closed contours.
"""
struct PVContour{T<:AbstractFloat}
    nodes::Vector{SVector{2, T}}
    pv::T
    wrap::SVector{2, T}   # periodic shift for closing segment; (0,0) for closed contours
end

# Backward-compatible constructor for closed contours
PVContour(nodes::Vector{SVector{2, T}}, pv::T) where {T<:AbstractFloat} =
    PVContour(nodes, pv, zero(SVector{2, T}))

"""
    nnodes(c::PVContour)

Number of nodes (vertices) in contour `c`.
"""
nnodes(c::PVContour) = length(c.nodes)

"""
    is_spanning(c::PVContour) -> Bool

Return `true` if contour `c` spans the periodic domain (i.e. has a non-zero wrap vector).
"""
is_spanning(c::PVContour) = any(!iszero, c.wrap)

"""Get the next node after index `j`, handling periodic wrap for spanning contours."""
@inline function next_node(c::PVContour{T}, j::Int) where {T}
    j < length(c.nodes) ? c.nodes[j + 1] : c.nodes[1] + c.wrap
end

# ── Domains ──────────────────────────────────────────────

abstract type AbstractDomain end

"""
    UnboundedDomain <: AbstractDomain

An infinite, unbounded two-dimensional domain.
"""
struct UnboundedDomain <: AbstractDomain end

"""
    PeriodicDomain{T}(Lx, Ly)

A doubly-periodic rectangular domain with half-widths `Lx` and `Ly`,
i.e. the domain `[-Lx, Lx) × [-Ly, Ly)`.
"""
struct PeriodicDomain{T<:AbstractFloat} <: AbstractDomain
    Lx::T
    Ly::T
    function PeriodicDomain(Lx::T, Ly::T) where {T<:AbstractFloat}
        Lx > zero(T) && Ly > zero(T) || throw(ArgumentError("Domain half-widths must be positive"))
        new{T}(Lx, Ly)
    end
end

# ── Problem Structs ──────────────────────────────────────

"""
    ContourProblem{K,D,T}(kernel, domain, contours)

A single-layer contour-dynamics problem with a velocity `kernel`, computational
`domain`, and a vector of [`PVContour`](@ref)s.
"""
struct ContourProblem{K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat}
    kernel::K
    domain::D
    contours::Vector{PVContour{T}}
    function ContourProblem(kernel::K, domain::D, contours::Vector{PVContour{T}}) where {K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat}
        _check_kernel_type(kernel, T)
        new{K, D, T}(kernel, domain, contours)
    end
end

# Warn if kernel's floating-point type doesn't match contour type
_check_kernel_type(::AbstractKernel, ::Type) = nothing  # no-op for unparameterized kernels
_check_kernel_type(k::QGKernel{Tk}, ::Type{T}) where {Tk, T} =
    Tk !== T && @warn "ContourProblem: QGKernel uses $Tk but contours use $T — this may cause type instability" maxlog=1
_check_kernel_type(k::SQGKernel{Tk}, ::Type{T}) where {Tk, T} =
    Tk !== T && @warn "ContourProblem: SQGKernel uses $Tk but contours use $T — this may cause type instability" maxlog=1

"""
    MultiLayerContourProblem{N,K,D,T}(kernel, domain, layers)

An `N`-layer contour-dynamics problem.  Each element of the `layers` tuple
holds the contours for one layer.
"""
struct MultiLayerContourProblem{N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat}
    kernel::K
    domain::D
    layers::NTuple{N, Vector{PVContour{T}}}
end

nlayers(::MultiLayerContourProblem{N}) where {N} = N

"""
    total_nodes(prob)

Total number of nodes across all contours in a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
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

"""
    SurgeryParams{T}(delta, mu, Delta_max, area_min, n_surgery)

Parameters controlling contour surgery.

- `delta`: proximity threshold for detecting close contour segments.
- `mu`: minimum segment length after remeshing.
- `Delta_max`: maximum segment length after remeshing.
- `area_min`: minimum enclosed area; contours smaller than this are removed.
- `n_surgery`: number of time-steps between surgery passes.
"""
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
        if delta > mu
            @warn "SurgeryParams: delta ($delta) > mu ($mu); typically delta ≤ mu/4 for correct Dritschel surgery" maxlog=1
        end
        new{T}(delta, mu, Delta_max, area_min, n_surgery)
    end
end

# ── Time Steppers ────────────────────────────────────────

abstract type AbstractTimeStepper end

"""
    RK4Stepper{T}(dt, n)

Classical fourth-order Runge–Kutta time stepper with step size `dt`.
Allocates internal buffers for `n` nodes.
"""
struct RK4Stepper{T<:AbstractFloat} <: AbstractTimeStepper
    dt::T
    k1::Vector{SVector{2, T}}
    k2::Vector{SVector{2, T}}
    k3::Vector{SVector{2, T}}
    k4::Vector{SVector{2, T}}
    nodes_buf::Vector{SVector{2, T}}  # pre-allocated buffer for original node positions
    vel_bufs::Vector{Vector{SVector{2, T}}}  # reusable per-layer velocity buffers (multi-layer)
end

function RK4Stepper(dt::T, n::Int) where {T<:AbstractFloat}
    z = zero(SVector{2, T})
    RK4Stepper(dt, fill(z, n), fill(z, n), fill(z, n), fill(z, n), fill(z, n),
               Vector{Vector{SVector{2, T}}}())
end

"""
    LeapfrogStepper{T}(dt, n)

Leapfrog (second-order centred) time stepper with step size `dt`.
The first step is bootstrapped with a forward-Euler half-step.
"""
mutable struct LeapfrogStepper{T<:AbstractFloat} <: AbstractTimeStepper
    dt::T
    nodes_prev::Vector{SVector{2, T}}
    vel_buf::Vector{SVector{2, T}}      # pre-allocated velocity buffer
    nodes_buf::Vector{SVector{2, T}}    # pre-allocated current-nodes buffer
    vel_mid::Vector{SVector{2, T}}      # pre-allocated bootstrap midpoint velocity buffer
    initialized::Bool
    ra_coeff::T  # Robert-Asselin filter coefficient (0 = no filter)
    vel_bufs::Vector{Vector{SVector{2, T}}}  # reusable per-layer velocity buffers (multi-layer)
end

function LeapfrogStepper(dt::T, n::Int; ra_coeff::T=T(0.05)) where {T<:AbstractFloat}
    z = zero(SVector{2, T})
    LeapfrogStepper(dt, fill(z, n), fill(z, n), fill(z, n), fill(z, n), false, ra_coeff,
                    Vector{Vector{SVector{2, T}}}())
end
