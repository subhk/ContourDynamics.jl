# Domain-specific single-layer velocity kernels for `UnboundedDomain`.

"""
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from node `a`
to node `b` with unit PV jump, using the 2D Euler Green's function in an
unbounded domain.

Computes the contour dynamics integral analytically:
  v_seg = -(1/(4π)) * (bx-ax, by-ay) * ∫₀¹ log|x - a - t(b-a)|² dt

The velocity direction is along `ds = b - a`, not rotated.
"""
function segment_velocity(::EulerKernel, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return SVector{2,T}(_straight_euler_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), one(T) / (T(4) * T(π))))
end

function curved_segment_velocity(::EulerKernel, ::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    return SVector{2,T}(_curved_euler_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), κa, κb, one(T) / (T(4) * T(π))))
end

function curved_segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    return SVector{2,T}(_curved_qg_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), κa, κb, kernel.Ld,
        one(T) / (T(2) * T(π)), one(T) / (T(4) * T(π))))
end

function curved_segment_velocity(kernel::SQGKernel{T}, domain::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    return SVector{2,T}(_curved_sqg_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), κa, κb, kernel.δ, one(T) / (T(2) * T(π))))
end

@inline curved_segment_velocity(k::AbstractKernel, d::AbstractDomain,
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                ::T, ::T) where {T} = segment_velocity(k, d, x, a, b)

@inline curved_segment_velocity(k::AbstractKernel, d::AbstractDomain,
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                κa::T, κb::T, ewald) where {T} = curved_segment_velocity(k, d, x, a, b, κa, κb)

# Unbounded domains don't use Ewald caches; ignore the argument.
@inline segment_velocity(k::AbstractKernel, d::UnboundedDomain,
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} = segment_velocity(k, d, x, a, b)

"""
    segment_velocity(::QGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from `a` to `b`
using the QG Green's function G(r) = K₀(r/Ld) / (2π).

The contour dynamics velocity is:
  v_seg = (1/(2π)) ∫₀¹ K₀(|x-P(t)|/Ld) ds dt

Uses singular subtraction: the log singularity in K₀ is handled analytically
(matching the Euler kernel), and the smooth remainder [K₀(r/Ld) + log(r)]
is integrated with 5-point Gauss-Legendre quadrature.
"""
function segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return SVector{2,T}(_curved_qg_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), zero(T), zero(T), kernel.Ld,
        one(T) / (T(2) * T(π)), one(T) / (T(4) * T(π))))
end

"""
    segment_velocity(::SQGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a surface buoyancy patch contour segment from
node `a` to node `b` with unit buoyancy jump, using the regularized SQG
Green's function `G(r) = 1/(2π√(r²+δ²))`.

The contour integral is:
  v_seg = (1/(2π)) t̂ [F(u_a) - F(u_b)]

where `F(u) = log(u + √(u² + h_eff²))` and `h_eff² = h² + δ²`.

The `1/(2π)` prefactor reflects the different Green's function
normalisation: SQG uses `1/(2πr)` while Euler uses `log(r²)/(4π)`.
"""
function segment_velocity(kernel::SQGKernel{T}, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return SVector{2,T}(_curved_sqg_contribution_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2],
        one(T), zero(T), zero(T), kernel.δ, one(T) / (T(2) * T(π))))
end
