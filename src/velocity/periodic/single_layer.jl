# Single-layer periodic velocity kernels.
#
# Every periodic segment velocity uses singular subtraction:
#
#     v_periodic = v_base  +  ∮_segment [G_per - G_base] ds
#
# where `v_base` is a singularity-handling base velocity (analytic unbounded
# Euler/SQG, or the periodic-Euler velocity for QG) and the bracketed term is a
# smooth periodic correction integrated with 5-point Gauss-Legendre quadrature.
# The smooth correction for each kernel is computed at a single quadrature point
# by `_periodic_{euler,qg,sqg}_green_correction`; both the straight and curved
# segment paths call those helpers, so the correction math has exactly one home.

@inline function _nearest_periodic_segment_image(domain::PeriodicDomain{T},
                                                 x::SVector{2,T},
                                                 a::SVector{2,T},
                                                 b::SVector{2,T}) where {T}
    ax, ay, bx, by = _nearest_periodic_segment_image_scalar(
        x[1], x[2], a[1], a[2], b[1], b[2], domain.Lx, domain.Ly)
    return SVector{2,T}(ax, ay), SVector{2,T}(bx, by)
end

@inline function _periodic_euler_zero_mode(cache::EwaldCache{T},
                                           domain::PeriodicDomain{T}) where {T}
    return _periodic_euler_zero_mode_scalar(cache.α, domain.Lx, domain.Ly)
end

# The three periodic point kernels share one singular-subtraction template for
# both straight and curved segments; only two ingredients vary per kernel:
#
#   * `_periodic_base_velocity` / `_periodic_curved_base_velocity` — the
#     singularity-handling base term:
#       - Euler: analytic unbounded Euler segment velocity (log singularity).
#       - QG:    periodic-Euler Ewald velocity (the QG cache carries the Euler
#                coefficients too, so one cache serves both parts).
#       - SQG:   regularized unbounded SQG velocity (1/r handled via δ).
#   * `_periodic_green_correction` — the smooth periodic correction
#     `G_per - G_base` at one quadrature point (decomposition documented on
#     each kernel's method below).
const _PeriodicPointKernel{T} = Union{EulerKernel, QGKernel{T}, SQGKernel{T}}

@inline _periodic_base_velocity(::EulerKernel, ::PeriodicDomain{T},
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                ::EwaldCache{T}) where {T} =  segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)

@inline _periodic_base_velocity(::QGKernel{T}, domain::PeriodicDomain{T},
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                cache::EwaldCache{T}) where {T} = segment_velocity(EulerKernel(), domain, x, a, b, cache)

@inline _periodic_base_velocity(kernel::SQGKernel{T}, ::PeriodicDomain{T},
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                ::EwaldCache{T}) where {T} = segment_velocity(kernel, UnboundedDomain(), x, a, b)

"""
    segment_velocity(kernel, domain::PeriodicDomain, x, a, b[, cache])

Velocity at point `x` from segment `a→b` in a periodic domain, for the Euler,
QG, and SQG kernels.

Uses singular subtraction: a per-kernel base velocity handles the singularity
analytically (`_periodic_base_velocity`), and only the smooth periodic
correction (`_periodic_green_correction`) is integrated with 5-point
Gauss-Legendre quadrature.
"""
function segment_velocity(kernel::_PeriodicPointKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

@inline segment_velocity(kernel::_PeriodicPointKernel{T}, domain::PeriodicDomain{T},
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} = segment_velocity(kernel, domain, x, a, b)

function segment_velocity(kernel::_PeriodicPointKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    sqrt(ds[1]^2 + ds[2]^2) < eps(T) && return zero(SVector{2,T})

    # Singularity-handling base term (per-kernel; see _periodic_base_velocity).
    v_base = _periodic_base_velocity(kernel, domain, x, a, b, cache)

    # Smooth periodic correction, integrated by 5-point Gauss-Legendre.
    g_nodes, g_weights = _gl5_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2
    corr = zero(T)

    for q in eachindex(g_nodes)
        s_pt = mid + g_nodes[q] * half_ds
        corr += g_weights[q] * _periodic_green_correction(kernel, domain, cache, x, s_pt)
    end
    return v_base + half_ds * corr
end

"""
    _periodic_green_correction(kernel, domain, cache, x, s_pt)

Smooth periodic Green's-function correction `G_per - G_base` at one quadrature
point, per kernel.

Euler decomposition (Ewald):
- Central-image real-space: (1/(4π))[E₁(α²r²) + log(r²)] → (1/(4π))(-γ - 2ln α) as r→0
- Non-central real-space: (1/(4π)) Σ_{images≠0} E₁(α²|r+shift|²)  (smooth)
- Fourier space: Σ_{k≠0} coeff * cos(k·r)  (smooth)
"""
@inline function _periodic_green_correction(::EulerKernel, domain::PeriodicDomain{T},
                                          cache::EwaldCache{T},
                                          x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    return _periodic_euler_green_correction_scalar(
        x[1], x[2], s_pt[1], s_pt[2], cache.α, domain.Lx, domain.Ly,
        cache.n_images, cache.kx, cache.ky, cache.fourier_coeffs,
        one(T) / (T(4) * T(π)), T(Base.MathConstants.eulergamma))
end

# QG–Euler decomposition: G_QG_per = G_Euler_per - G_correction, where the
# correction is a smooth, rapidly convergent Fourier series
#   G_corr(r) = -(1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²)),  κ = 1/Ld.
# Coefficients decay as 1/k⁴, so the truncated sum converges without damping.
@inline function _periodic_green_correction(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                       cache::EwaldCache{T},
                                       x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    return _periodic_qg_green_correction_scalar(
        x[1], x[2], s_pt[1], s_pt[2], inv(kernel.Ld^2), T(4) * domain.Lx * domain.Ly,
        cache.kx, cache.ky, cache.corr_coeffs)
end

# SQG Ewald decomposition of the regularized kernel over every periodic image:
# - Central-image real-space: -(1/(2π)) erf(αr)/r, finite limit -2α/√π at r=0.
# - Non-central real-space: unregularized Ewald term plus (1/(2π))(1/r_δ - 1/r),
#   with r_δ = √(r² + δ²).
# - Fourier space: (1/(2π)) Σ c_k cos(k·r), c_k = (2π/|k|) erfc(|k|/(2α))/A.
@inline function _periodic_green_correction(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                        cache::EwaldCache{T},
                                        x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    return _periodic_sqg_green_correction_scalar(
        x[1], x[2], s_pt[1], s_pt[2], cache.α, kernel.δ^2, domain.Lx, domain.Ly,
        cache.n_images, cache.kx, cache.ky, cache.fourier_coeffs, one(T) / (T(2) * T(π)))
end


@inline _periodic_curved_base_velocity(::EulerKernel, ::PeriodicDomain{T},
                                       x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                       κa::T, κb::T, ::EwaldCache{T}) where {T} = curved_segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b, κa, κb)

@inline _periodic_curved_base_velocity(::QGKernel{T}, domain::PeriodicDomain{T},
                                       x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                       κa::T, κb::T, cache::EwaldCache{T}) where {T} = curved_segment_velocity(EulerKernel(), domain, x, a, b, κa, κb, cache)

@inline _periodic_curved_base_velocity(kernel::SQGKernel{T}, ::PeriodicDomain{T},
                                       x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                       κa::T, κb::T, ::EwaldCache{T}) where {T} = curved_segment_velocity(kernel, UnboundedDomain(), x, a, b, κa, κb)

function curved_segment_velocity(kernel::_PeriodicPointKernel{T}, domain::PeriodicDomain{T},
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    curved_segment_velocity(kernel, domain, x, a, b, κa, κb, 
                            _get_ewald_cache(domain, kernel))
end

function curved_segment_velocity(kernel::_PeriodicPointKernel{T}, domain::PeriodicDomain{T},
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T, cache::EwaldCache{T}) where {T}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})
    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return segment_velocity(kernel, domain, x, a, b, cache)

    g_nodes, g_weights = _gl5_nodes_weights(T)
    corr_integral = zero(SVector{2,T})
    
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        s = _cubic_segment_point(a, b, κa, κb, p)
        tangent = _cubic_segment_tangent(a, b, κa, κb, p)
        G_corr = _periodic_green_correction(kernel, domain, cache, x, s)
        corr_integral += (g_weights[q] / T(2)) * G_corr * tangent
    end

    return _periodic_curved_base_velocity(kernel, domain, x, a, b, κa, κb, cache) + corr_integral
end
