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

Contour-dynamics beta-plane QG kernel with finite, positive deformation radius
`Ld`.

The live contours represent full PV `q = q_s + q_r`. `reference_contours` is
the undeformed straight staircase `q_ref = q_r(t=0)`. The velocity comes from
`q - q_ref` plus the analytic zonal correction for `q_ref - beta*y`, so the
total inversion source is exactly `q - beta*y`.
"""
struct BetaPlaneQGKernel{T<:AbstractFloat} <: AbstractKernel
    beta::T
    Ld::T
    reference_contours::Vector{PVContour{T}}
    function BetaPlaneQGKernel(beta::T, Ld::T,
                               reference_contours::Vector{PVContour{T}}) where {T<:AbstractFloat}
        isfinite(beta) || throw(ArgumentError("Planetary PV gradient beta must be finite, got $beta"))
        isfinite(Ld) && Ld > zero(T) || throw(ArgumentError(
            "Deformation radius Ld must be finite and positive, got $Ld"))
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

"""
    _beta_sawtooth_u(beta, κ, dy, ξ)

Zonal velocity of the sawtooth jet that solves `(∂yy − κ²)ψ = −βξ` periodically
across one staircase step of height `dy`, with `u = −∂yψ` and `ξ ∈ [−dy/2, dy/2]`
the local step coordinate:

    u(ξ) = (β/κ²) [ z·cosh(κξ)/sinh(z) − 1 ],   z = κ·dy/2

Evaluated directly, that is a difference of two `O(β/κ²)` terms and cancels
catastrophically as `z → 0` — the near-barotropic regime where the deformation
radius exceeds the step spacing. For `κ·dy < 0.05` we therefore use the Taylor
expansion in `z` (whose leading term is the barotropic jet `β(ξ²/2 − dy²/24)`),
and the closed form above only where it is well conditioned. Both branches are
accurate to a relative `~1e-11` at the crossover, and the expansion has no `κ`
in a denominator, so the large-finite-`Ld` limit remains accurate.
"""

# Crossover in `z = κ·dy/2` between the Taylor branch and the closed form. The
# closed form's cancellation error grows like `eps(T)/z²` while the series'
# truncation error falls like `z⁶`; balancing the two gives `z ~ eps(T)^(1/8)`,
# so the cutoff must widen as the working precision drops. The Float32 and
# Float64 values are written out so the comparison folds to a constant inside
# GPU kernels.
@inline _beta_series_cutoff(::Type{Float64}) = 0.0275
@inline _beta_series_cutoff(::Type{Float32}) = 0.34f0
@inline _beta_series_cutoff(::Type{T}) where {T} = T(5) / 2 * eps(T)^(one(T) / 8)

@inline function _beta_sawtooth_u(beta::T, κ::T, dy::T, ξ::T) where {T}
    z = κ * dy / 2
    if abs(z) < _beta_series_cutoff(T)
        s = 2 * ξ / dy                       # step coordinate scaled to [-1, 1]
        s2 = s * s
        z2 = z * z
        return beta * dy^2 / 4 * (
            (s2 / 2 - T(1) / 6)
            + z2 * (s2 * s2 / 24 - s2 / 12 + T(7) / 360)
            + z2 * z2 * (s2 * s2 * s2 / 720 - s2 * s2 / 144 + 7 * s2 / 720 - T(31) / 15120))
    end
    return -beta / κ^2 + beta * dy * cosh(κ * ξ) / (2 * κ * sinh(z))
end
