# Domain-specific single-layer periodic velocity kernels.

"""
    segment_velocity(kernel::EulerKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using Ewald summation.

Uses singular subtraction: the log singularity is handled analytically by the
unbounded Euler segment velocity, and only the smooth periodic correction
`G_per - G_∞` is integrated with 3-point Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: (1/(4π))[E₁(α²r²) + log(r²)] → (1/(4π))(-γ - 2ln α) as r→0
- Non-central real-space: (1/(4π)) Σ_{images≠0} E₁(α²|r+shift|²)  (smooth)
- Fourier space: Σ_{k≠0} coeff * cos(k·r)  (smooth)
"""
function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Analytic unbounded Euler contribution (handles the log singularity exactly)
    v_unbounded = segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)

    # Smooth periodic correction: G_per(r) - G_∞(r)
    # G_∞(r) = -(1/(4π)) log(r²), so we subtract it from the central Ewald image.
    # All terms in the correction are bounded at r=0 → GL quadrature is accurate.
    g_nodes, g_weights = _gl3_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2

    inv4pi = one(T) / (4 * T(π))
    gamma_euler = T(Base.MathConstants.eulergamma)

    corr_integral = zero(T)

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        r_vec0_raw = x - s_pt
        # Minimum-image wrap so the central image (px=0,py=0) is always the
        # closest periodic copy, reducing the n_images required for convergence.
        r_vec0 = SVector{2,T}(
            r_vec0_raw[1] - round(r_vec0_raw[1] / (2 * Lx)) * (2 * Lx),
            r_vec0_raw[2] - round(r_vec0_raw[2] / (2 * Ly)) * (2 * Ly))
        G_corr = zero(T)

        # Real-space Ewald sum with central-image singularity subtracted
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = r_vec0 - shift
                r2 = r_vec[1]^2 + r_vec[2]^2
                if px == 0 && py == 0
                    # Central image: compute E₁(α²r²) + log(r²) which is smooth at r=0.
                    # Limit as r→0: E₁(x) + log(x) → -γ, and log(r²) = log(x/α²),
                    # so E₁(α²r²) + log(r²) → -γ - 2*log(α).
                    if r2 > eps(T)
                        G_corr += inv4pi * (_expint_e1(alpha^2 * r2) + log(r2))
                    else
                        G_corr += inv4pi * (-gamma_euler - 2 * log(alpha))
                    end
                else
                    # Non-central images: already smooth
                    if r2 > eps(T)
                        G_corr += inv4pi * _expint_e1(alpha^2 * r2)
                    end
                end
            end
        end

        # Fourier-space sum (smooth).
        # r_vec0 is minimum-image wrapped (for the real-space sum above), but
        # cos(k·r) is periodic with the same periods as the domain, so the
        # wrapping has no effect on the Fourier sum.
        # Use cos(kx*rx + ky*ry) = cos(kx*rx)*cos(ky*ry) - sin(kx*rx)*sin(ky*ry)
        # to reduce trig calls from O(nk²) to O(nk).
        rx, ry = r_vec0[1], r_vec0[2]
        nkx = length(cache.kx)
        nky = length(cache.ky)
        for mi in 1:nkx
            kxi = cache.kx[mi]
            cx = cos(kxi * rx)
            sx = sin(kxi * rx)
            for ni in 1:nky
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                kyi = cache.ky[ni]
                # cos(kx*rx + ky*ry) = cx*cos(ky*ry) - sx*sin(ky*ry)
                G_corr += coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
            end
        end

        corr_integral += g_weights[q] * G_corr
    end

    # v_periodic = v_unbounded + ds * ∫₀¹ [G_per - G_∞] dt
    # The ∫₀¹ dt = (1/2) ∫_{-1}^{1} dt' gives the half_ds factor.
    return v_unbounded + half_ds * corr_integral
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
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, EulerKernel()))
end

function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           euler_cache::EwaldCache{T}) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Euler periodic part (handles the log singularity via Ewald)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b, euler_cache)

    # Smooth QG–Euler correction via Fourier sum.
    # G_QG_per - G_Euler_per = -(1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²))
    # The sign is negative because the QG kernel (K₀) is screened relative
    # to the Euler kernel (log): F{K₀(κr)/(2π)} = 1/(k²+κ²) < 1/k² = F{-log(r)/(2π)}.
    # NOTE: These coefficients are computed inline (without Gaussian damping)
    # because the correction is smooth and converges without Ewald splitting.
    # The precomputed QG fourier_coeffs from build_ewald_cache include Gaussian
    # damping and are therefore not used here.
    kappa2 = one(T) / kernel.Ld^2
    area = 4 * domain.Lx * domain.Ly

    g_nodes, g_weights = _gl3_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2

    nkx = length(euler_cache.kx)
    nky = length(euler_cache.ky)

    corr_integral = zero(T)
    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        r_vec = x - s_pt
        rx, ry = r_vec[1], r_vec[2]
        G_corr = zero(T)
        @inbounds for mi in 1:nkx
            kxi = euler_cache.kx[mi]
            cx = cos(kxi * rx)
            sx = sin(kxi * rx)
            for ni in 1:nky
                kyi = euler_cache.ky[ni]
                k2 = kxi^2 + kyi^2
                k2 < eps(T) && continue
                coeff = kappa2 / (k2 * (k2 + kappa2) * area)
                G_corr -= coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
            end
        end
        corr_integral += g_weights[q] * G_corr
    end

    return v_euler + half_ds * corr_integral
end

"""
    segment_velocity(kernel::SQGKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using the SQG kernel.

Uses singular subtraction: the regularized unbounded SQG velocity (exact arcsinh
antiderivative with ``\\delta > 0``) handles the ``1/r`` singularity analytically, and the
smooth periodic correction ``G_{\\text{per}} - G_\\infty`` is integrated with 3-point
Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: ``-(1/(2\\pi))[\\operatorname{erfc}(\\alpha r)/r - 1/\\sqrt{r^2+\\delta^2}]``
  (finite at GL quadrature points since ``r > 0``)
- Non-central real-space: ``-(1/(2\\pi)) \\operatorname{erfc}(\\alpha|\\mathbf{r}-\\mathbf{R}_n|)/|\\mathbf{r}-\\mathbf{R}_n|``
- Fourier space: ``-(1/(2\\pi)) \\sum c_k \\cos(\\mathbf{k}\\cdot\\mathbf{r})``
  with ``c_k = (2\\pi/|k|) e^{-k^2/(4\\alpha^2)}/A``
"""
function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    alpha = cache.alpha
    delta = kernel.delta
    delta_sq = delta * delta
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Exact unbounded regularized SQG velocity (handles the 1/r singularity via δ)
    v_unbounded = segment_velocity(kernel, UnboundedDomain(), x, a, b)

    # Smooth periodic correction: G_per_SQG(r) - G_∞_SQG_reg(r)
    # G_∞_SQG_reg(r) = -(1/(2π))/√(r²+δ²) is the regularized unbounded Green's function.
    # All correction terms are finite at GL quadrature points (r > 0).
    g_nodes, g_weights = _gl3_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2

    inv2pi = one(T) / (2 * T(π))

    corr_integral = zero(T)

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        r_vec0_raw = x - s_pt
        # Minimum-image wrap for the real-space sum
        r_vec0 = SVector{2,T}(
            r_vec0_raw[1] - round(r_vec0_raw[1] / (2 * Lx)) * (2 * Lx),
            r_vec0_raw[2] - round(r_vec0_raw[2] / (2 * Ly)) * (2 * Ly))
        G_corr = zero(T)

        # Real-space Ewald sum with central-image singularity subtracted
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = r_vec0 - shift
                r2 = r_vec[1]^2 + r_vec[2]^2
                if px == 0 && py == 0
                    # Central image: evaluate Ewald real-space at √(r²+δ²) instead
                    # of r so the subtraction erfc(α·r_reg)/r_reg - 1/r_reg
                    # = -erf(α·r_reg)/r_reg is bounded by 1/δ for all r ≥ 0.
                    # This is an O(δ²/r) perturbation, within the existing
                    # regularization error. Avoids 1/r divergence when contour
                    # nodes from different PV levels pass very close together.
                    r_reg = sqrt(r2 + delta_sq)
                    G_corr += inv2pi * erf(alpha * r_reg) / r_reg
                else
                    # Non-central images: -(1/(2π)) erfc(α|r|)/|r|  (smooth)
                    if r2 > eps(T)
                        r = sqrt(r2)
                        G_corr -= inv2pi * erfc(alpha * r) / r
                    end
                end
            end
        end

        # Fourier-space sum: -(1/(2π)) Σ fourier_coeffs cos(k·r)
        # Factored trig: cos(kx*rx + ky*ry) = cos(kx*rx)*cos(ky*ry) - sin(kx*rx)*sin(ky*ry)
        rx, ry = r_vec0[1], r_vec0[2]
        nkx = length(cache.kx)
        nky = length(cache.ky)
        for mi in 1:nkx
            kxi = cache.kx[mi]
            cx = cos(kxi * rx)
            sx = sin(kxi * rx)
            for ni in 1:nky
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                kyi = cache.ky[ni]
                G_corr -= inv2pi * coeff * (cx * cos(kyi * ry) - sx * sin(kyi * ry))
            end
        end

        corr_integral += g_weights[q] * G_corr
    end

    return v_unbounded + half_ds * corr_integral
end
