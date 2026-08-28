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
    Lx2 = T(2) * domain.Lx
    Ly2 = T(2) * domain.Ly
    mid = (a + b) / T(2)
    shift = SVector{2,T}(
        round((x[1] - mid[1]) / Lx2) * Lx2,
        round((x[2] - mid[2]) / Ly2) * Ly2)
    return a + shift, b + shift
end

@inline function _periodic_euler_zero_mode(cache::EwaldCache{T},
                                           domain::PeriodicDomain{T}) where {T}
    area = T(4) * domain.Lx * domain.Ly
    return one(T) / (T(4) * cache.α^2 * area)
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
    α = cache.α
    Lx, Ly = domain.Lx, domain.Ly
    inv4pi = one(T) / (4 * T(π))
    γ_euler = T(Base.MathConstants.eulergamma)
    zero_mode = _periodic_euler_zero_mode(cache, domain)

    r_vec0 = x - s_pt
    G_corr = zero(T)

    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            r_vec = r_vec0 - shift
            r2 = r_vec[1]^2 + r_vec[2]^2
            if px == 0 && py == 0
                if r2 > eps(T)
                    G_corr += inv4pi * (_expint_e1(α^2 * r2) + log(r2))
                else
                    G_corr += inv4pi * (-γ_euler - 2 * log(α))
                end
            elseif r2 > eps(T)
                G_corr += inv4pi * _expint_e1(α^2 * r2)
            end
        end
    end

    rx, ry = r_vec0[1], r_vec0[2]
    nkx = length(cache.kx)
    nky = length(cache.ky)

    @inbounds for mi in 1:nkx
        kxi = cache.kx[mi]
        cx = cos(kxi * rx)
        sx = sin(kxi * rx)
        for ni in 1:nky
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            kyi = cache.ky[ni]
            G_corr += coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
        end
    end

    return G_corr - zero_mode
end

# QG–Euler decomposition: G_QG_per = G_Euler_per - G_correction, where the
# correction is a smooth, rapidly convergent Fourier series
#   G_corr(r) = -(1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²)),  κ = 1/Ld.
# Coefficients decay as 1/k⁴, so the truncated sum converges without damping.
@inline function _periodic_green_correction(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                       cache::EwaldCache{T},
                                       x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    # The correction coefficients κ²/(k²(k²+κ²)A) are precomputed in `cache`
    # (zero for the k=0 mode); read them rather than recomputing a division per
    # wavenumber. Coefficients decay as 1/k⁴, so the truncated sum converges.
    r_vec = x - s_pt
    rx, ry = r_vec[1], r_vec[2]
    G_corr = zero(T)
    corr_coeffs = cache.corr_coeffs
    nkx = length(cache.kx)
    nky = length(cache.ky)

    @inbounds for mi in 1:nkx
        kxi = cache.kx[mi]
        cx = cos(kxi * rx)
        sx = sin(kxi * rx)
        for ni in 1:nky
            coeff = corr_coeffs[mi, ni]
            iszero(coeff) && continue
            kyi = cache.ky[ni]
            G_corr -= coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
        end
    end

    return G_corr
end

# SQG Ewald decomposition of the regularized kernel over every periodic image:
# - Central-image real-space: -(1/(2π)) erf(αr)/r, finite limit -2α/√π at r=0.
# - Non-central real-space: unregularized Ewald term plus (1/(2π))(1/r_δ - 1/r),
#   with r_δ = √(r² + δ²).
# - Fourier space: (1/(2π)) Σ c_k cos(k·r), c_k = (2π/|k|) erfc(|k|/(2α))/A.
@inline function _periodic_green_correction(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                        cache::EwaldCache{T},
                                        x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    α = cache.α
    δ_sq = kernel.δ^2
    Lx, Ly = domain.Lx, domain.Ly
    inv2pi = one(T) / (2 * T(π))

    r_vec0 = x - s_pt
    G_corr = zero(T)

    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            r_vec = r_vec0 - shift
            r2 = r_vec[1]^2 + r_vec[2]^2
            if px == 0 && py == 0
                G_corr -= inv2pi * _sqg_erf_over_r(α, r2)
            elseif r2 > eps(T)
                r = sqrt(r2)
                r_reg = sqrt(r2 + δ_sq)
                # erfc(αr)/r + (1/rδ - 1/r), written so the softening
                # adjustment does not lose precision when δ ≪ r.
                softening = -δ_sq / (r * r_reg * (r + r_reg))
                G_corr += inv2pi * (erfc(α * r) / r + softening)
            end
        end
    end

    rx, ry = r_vec0[1], r_vec0[2]
    nkx = length(cache.kx)
    nky = length(cache.ky)
    @inbounds for mi in 1:nkx
        kxi = cache.kx[mi]
        cx = cos(kxi * rx)
        sx = sin(kxi * rx)
        for ni in 1:nky
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            kyi = cache.ky[ni]
            G_corr += inv2pi * coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
        end
    end

    return G_corr
end

@inline function _sqg_erf_over_r(α::T, r2::T) where {T}
    if r2 > eps(T)^2
        r = sqrt(r2)
        return erf(α * r) / r
    end
    return T(2) * α / sqrt(T(π))
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
