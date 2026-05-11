# Beta-plane contour QG stores total-PV contours plus a frozen straight
# staircase reference. The inversion uses current contours minus that reference,
# plus the analytic velocity of the reference staircase relative to beta*y.

function _copy_pv_contour(c::PVContour{T}) where {T}
    return PVContour(copy(c.nodes), c.pv, c.wrap, copy(c.corners))
end

_copy_pv_contours(contours::Vector{PVContour{T}}) where {T} =
    [_copy_pv_contour(c) for c in contours]

_beta_plane_reference_contours(contours::Vector{PVContour{T}}) where {T} =
    [_copy_pv_contour(c) for c in contours if is_spanning(c)]

@inline function _beta_plane_isapprox(a::T, b::T, scale::T) where {T<:AbstractFloat}
    return isapprox(a, b; rtol=sqrt(eps(T)) * T(100),
                    atol=eps(T) * T(100) * max(one(T), abs(scale)))
end

"""
    BetaPlaneQGKernel(beta, Ld, reference_contours)

Contour-dynamics beta-plane QG kernel with deformation radius `Ld`.

The live contours represent full PV. `reference_contours` is the undeformed
straight beta staircase. The velocity comes from `current full PV - reference
beta staircase` plus the analytic zonal correction for
`reference beta staircase - beta*y`.
"""
struct BetaPlaneQGKernel{T<:AbstractFloat} <: AbstractKernel
    beta::T
    Ld::T
    reference_contours::Vector{PVContour{T}}
    function BetaPlaneQGKernel(beta::T, Ld::T,
                               reference_contours::Vector{PVContour{T}}) where {T<:AbstractFloat}
        Ld > zero(T) || throw(ArgumentError("Deformation radius Ld must be positive, got $Ld"))
        all(is_spanning, reference_contours) || throw(ArgumentError(
            "BetaPlaneQGKernel reference_contours must all be spanning beta-staircase contours."))
        new{T}(beta, Ld, _copy_pv_contours(reference_contours))
    end
end

BetaPlaneQGKernel(beta::Real, Ld::Real,
                  reference_contours::Vector{PVContour{T}}) where {T<:AbstractFloat} =
    BetaPlaneQGKernel(T(beta), T(Ld), reference_contours)

function _validate_beta_plane_reference(kernel::BetaPlaneQGKernel{T},
                                        domain::PeriodicDomain{T}) where {T<:AbstractFloat}
    reference = kernel.reference_contours
    isempty(reference) && throw(ArgumentError(
        "BetaPlaneQGKernel requires at least one reference beta-staircase contour."))

    expected_wrap = SVector{2,T}(2 * domain.Lx, zero(T))
    expected_dy = 2 * domain.Ly / T(length(reference))
    expected_pv = kernel.beta * expected_dy

    for (ci, c) in pairs(reference)
        nnodes(c) > 0 || throw(ArgumentError(
            "BetaPlaneQGKernel reference contour $ci has no nodes."))

        _beta_plane_isapprox(c.wrap[1], expected_wrap[1], expected_wrap[1]) &&
            _beta_plane_isapprox(c.wrap[2], expected_wrap[2], domain.Ly) ||
            throw(ArgumentError(
                "BetaPlaneQGKernel reference contour $ci has wrap $(c.wrap); " *
                "expected $(expected_wrap) for this periodic domain."))

        _beta_plane_isapprox(c.pv, expected_pv, expected_pv) || throw(ArgumentError(
            "BetaPlaneQGKernel reference contour $ci has PV jump $(c.pv); " *
            "expected beta * dy = $expected_pv for beta=$(kernel.beta)."))

        y0 = c.nodes[1][2]
        for node in c.nodes
            _beta_plane_isapprox(node[2], y0, domain.Ly) || throw(ArgumentError(
                "BetaPlaneQGKernel reference contour $ci is not a straight horizontal staircase contour."))
        end
    end

    y_levels = sort([c.nodes[1][2] for c in reference])
    for (k, y) in pairs(y_levels)
        expected_y = -domain.Ly + (T(k) - T(0.5)) * expected_dy
        _beta_plane_isapprox(y, expected_y, domain.Ly) || throw(ArgumentError(
            "BetaPlaneQGKernel reference y-level $k is $y; expected $expected_y. " *
            "Build reference contours with beta_staircase(beta, domain, n_beta)."))
    end
    return nothing
end

function _validate_beta_plane_reference(::BetaPlaneQGKernel, ::AbstractDomain)
    throw(ArgumentError("BetaPlaneQGKernel currently requires PeriodicDomain."))
end

function _attach_beta_plane_reference(kernel::BetaPlaneQGKernel{T},
                                      contours::Vector{PVContour{T}}) where {T}
    !isempty(kernel.reference_contours) && return kernel
    reference = _beta_plane_reference_contours(contours)
    isempty(reference) && throw(ArgumentError(
        "kernel=:beta_plane_qg requires spanning beta-staircase contours. " *
        "Build them with beta_staircase(beta, PeriodicDomain(Lx, Ly), n_beta)."))
    return BetaPlaneQGKernel(kernel.beta, kernel.Ld, reference)
end
