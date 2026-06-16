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
    return one(T) / (T(4) * cache.alpha^2 * area)
end

"""
    segment_velocity(kernel::EulerKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using Ewald summation.

Uses singular subtraction: the log singularity is handled analytically by the
unbounded Euler segment velocity, and only the smooth periodic correction
`G_per - G_∞` is integrated with 5-point Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: (1/(4π))[E₁(α²r²) + log(r²)] → (1/(4π))(-γ - 2ln α) as r→0
- Non-central real-space: (1/(4π)) Σ_{images≠0} E₁(α²|r+shift|²)  (smooth)
- Fourier space: Σ_{k≠0} coeff * cos(k·r)  (smooth)
"""
function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

@inline segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} =
    segment_velocity(kernel, domain, x, a, b)

function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    sqrt(ds[1]^2 + ds[2]^2) < eps(T) && return zero(SVector{2,T})

    # Analytic unbounded Euler contribution (handles the log singularity exactly).
    v_unbounded = segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)

    # Smooth periodic correction G_per - G_∞, integrated by 5-point Gauss-Legendre.
    # The per-node correction lives in `_periodic_euler_green_correction` so the
    # straight and curved segment paths share one implementation.
    g_nodes, g_weights = _gl5_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2
    corr = zero(T)
    for q in eachindex(g_nodes)
        s_pt = mid + g_nodes[q] * half_ds
        corr += g_weights[q] * _periodic_euler_green_correction(kernel, domain, cache, x, s_pt)
    end
    return v_unbounded + half_ds * corr
end

"""
    segment_velocity(kernel::QGKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using the QG kernel.

Decomposes the periodic QG Green's function as:
  G_QG_per = G_Euler_per - G_correction
where the Euler part is handled by the validated Ewald summation, and the
correction is a smooth, rapidly convergent Fourier series:
  G_corr(r) = -(1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²))
with κ = 1/Ld.  Coefficients decay as 1/k⁴, so the truncated sum converges
without Gaussian damping.
"""
function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

@inline segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} =
    segment_velocity(kernel, domain, x, a, b)

function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    sqrt(ds[1]^2 + ds[2]^2) < eps(T) && return zero(SVector{2,T})

    # Euler periodic part handles the log singularity via Ewald. The QG cache
    # carries the Euler coefficients too, so it serves both parts.
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b, cache)

    # Smooth QG–Euler correction (Fourier series), integrated by Gauss-Legendre.
    # The per-node correction lives in `_periodic_qg_green_correction` so the
    # straight and curved segment paths share one implementation.
    g_nodes, g_weights = _gl5_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2
    corr = zero(T)
    for q in eachindex(g_nodes)
        s_pt = mid + g_nodes[q] * half_ds
        corr += g_weights[q] * _periodic_qg_green_correction(kernel, domain, cache, x, s_pt)
    end
    return v_euler + half_ds * corr
end

"""
    segment_velocity(kernel::SQGKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using the SQG kernel.

Uses singular subtraction: the regularized unbounded SQG velocity (exact arcsinh
antiderivative with ``\\delta > 0``) handles the ``1/r`` singularity analytically, and the
smooth periodic correction ``G_{\\text{per}} - G_\\infty`` is integrated with 5-point
Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: ``-(1/(2\\pi))\\operatorname{erf}(\\alpha r_\\delta)/r_\\delta``,
  where ``r_\\delta = \\sqrt{r^2+\\delta^2}``
- Non-central real-space: ``(1/(2\\pi)) \\operatorname{erfc}(\\alpha|\\mathbf{r}-\\mathbf{R}_n|)/|\\mathbf{r}-\\mathbf{R}_n|``
- Fourier space: ``(1/(2\\pi)) \\sum c_k \\cos(\\mathbf{k}\\cdot\\mathbf{r})``
  with ``c_k = (2\\pi/|k|) \\operatorname{erfc}(|k|/(2\\alpha))/A``
"""
function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

@inline segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} =
    segment_velocity(kernel, domain, x, a, b)

function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    sqrt(ds[1]^2 + ds[2]^2) < eps(T) && return zero(SVector{2,T})

    # Exact unbounded regularized SQG velocity (handles the 1/r singularity via δ).
    v_unbounded = segment_velocity(kernel, UnboundedDomain(), x, a, b)

    # Smooth periodic correction G_per - G_∞, integrated by Gauss-Legendre.
    # The per-node correction lives in `_periodic_sqg_green_correction` so the
    # straight and curved segment paths share one implementation.
    g_nodes, g_weights = _gl5_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2
    corr = zero(T)
    for q in eachindex(g_nodes)
        s_pt = mid + g_nodes[q] * half_ds
        corr += g_weights[q] * _periodic_sqg_green_correction(kernel, domain, cache, x, s_pt)
    end
    return v_unbounded + half_ds * corr
end

@inline function _periodic_euler_green_correction(::EulerKernel, domain::PeriodicDomain{T},
                                          cache::EwaldCache{T},
                                          x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    inv4pi = one(T) / (4 * T(π))
    gamma_euler = T(Base.MathConstants.eulergamma)
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
                    G_corr += inv4pi * (_expint_e1(alpha^2 * r2) + log(r2))
                else
                    G_corr += inv4pi * (-gamma_euler - 2 * log(alpha))
                end
            elseif r2 > eps(T)
                G_corr += inv4pi * _expint_e1(alpha^2 * r2)
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

@inline function _periodic_qg_green_correction(kernel::QGKernel{T}, domain::PeriodicDomain{T},
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

@inline function _periodic_sqg_green_correction(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                        cache::EwaldCache{T},
                                        x::SVector{2,T}, s_pt::SVector{2,T}) where {T}
    alpha = cache.alpha
    delta_sq = kernel.delta^2
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
                r_reg = sqrt(r2 + delta_sq)
                G_corr -= inv2pi * erf(alpha * r_reg) / r_reg
            elseif r2 > eps(T)
                r = sqrt(r2)
                G_corr += inv2pi * erfc(alpha * r) / r
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

function curved_segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    curved_segment_velocity(kernel, domain, x, a, b, κa, κb, _get_ewald_cache(domain, kernel))
end

function curved_segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
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
        G_corr = _periodic_euler_green_correction(kernel, domain, cache, x, s)
        corr_integral += (g_weights[q] / T(2)) * G_corr * tangent
    end

    return curved_segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b, κa, κb) +
           corr_integral
end

function curved_segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    curved_segment_velocity(kernel, domain, x, a, b, κa, κb, _get_ewald_cache(domain, kernel))
end

function curved_segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
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
        G_corr = _periodic_qg_green_correction(kernel, domain, cache, x, s)
        corr_integral += (g_weights[q] / T(2)) * G_corr * tangent
    end

    return curved_segment_velocity(EulerKernel(), domain, x, a, b, κa, κb, cache) +
           corr_integral
end

function curved_segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    curved_segment_velocity(kernel, domain, x, a, b, κa, κb, _get_ewald_cache(domain, kernel))
end

function curved_segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
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
        G_corr = _periodic_sqg_green_correction(kernel, domain, cache, x, s)
        corr_integral += (g_weights[q] / T(2)) * G_corr * tangent
    end

    return curved_segment_velocity(kernel, UnboundedDomain(), x, a, b, κa, κb) +
           corr_integral
end
