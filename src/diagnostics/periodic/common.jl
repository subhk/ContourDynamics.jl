# Periodic-domain helper routines shared by diagnostics.

"""
    _eval_ewald_greens(r_vec, cache, domain)

Evaluate the periodic Green's function at separation `r_vec` using Ewald summation.
Returns `G_per(r)` = real-space Ewald sum + Fourier-space sum.

!!! note
    The central-image real-space term (E₁(α²r²)) diverges at `r = 0`.
    This function silently skips that term when `r² < eps(T)`, so the
    returned value is *not* valid at zero separation.  Callers that need
    the self-interaction limit must handle `r = 0` separately (see
    `_energy_contour_pair_periodic_green` for an example).
"""
function _eval_ewald_greens(r_vec::SVector{2,T}, cache::EwaldCache{T},
                            domain::PeriodicDomain{T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    inv4pi = one(T) / (4 * T(π))
    G_val = zero(T)

    # Real-space sum
    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            rv = r_vec - shift
            r2 = rv[1]^2 + rv[2]^2
            if r2 > eps(T)
                G_val += inv4pi * _expint_e1(alpha^2 * r2)
            end
        end
    end

    # Fourier-space sum
    for (mi, kxi) in enumerate(cache.kx)
        for (ni, kyi) in enumerate(cache.ky)
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            phase = kxi * r_vec[1] + kyi * r_vec[2]
            G_val += coeff * cos(phase)
        end
    end

    return G_val
end

function _energy_contour_pair_periodic_green(ci::PVContour{T}, cj::PVContour{T},
                                             cache::EwaldCache{T},
                                             domain::PeriodicDomain{T};
                                             _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Double contour integral for periodic Euler energy. The self-pair case
    # subtracts the logarithmic singularity analytically and adds back the smooth
    # periodic correction through quadrature.
    is_self = ci.nodes === cj.nodes
    Lx2, Ly2 = _period_lengths(domain.Lx, domain.Ly)

    # Assign the r→0 correction exactly once. Mutating it in place (`-=`) would
    # force Julia to heap-box the variable so the @_energy_segment_loop
    # Threads.@threads closure could capture it, and every boxed read inside the
    # O(N²) segment loop below would then allocate (~350 KB for N=64).
    corr_at_zero = is_self ? _periodic_euler_corr_at_zero(cache, domain) : zero(T)

    # Distinct segments are smooth after minimum-image wrapping, so the periodic
    # Green's function can be sampled directly. Replace log(r²)/2 with its
    # periodic equivalent -2π G_per, since log(r²)/2 = -2π G_∞ for unbounded Euler.
    Φ = rv -> begin
        # Minimum-image wrap for Ewald convergence (matches velocity path)
        r_vec = SVector{2,T}(rv[1] - round(rv[1] / Lx2) * Lx2,
                             rv[2] - round(rv[2] / Ly2) * Ly2)
        -2 * T(π) * _eval_ewald_greens(r_vec, cache, domain)
    end
    # Smooth periodic correction [-2π G_per(r) - log(r²)/2]; unwrapped, since
    # the self pair's quadrature points are already within one image.
    Φ_corr = rv -> begin
        r2 = rv[1]^2 + rv[2]^2
        # qi == qj: fall back to the precomputed finite limit
        r2 > eps(T) || return corr_at_zero
        return -2 * T(π) * _eval_ewald_greens(rv, cache, domain) - log(r2) / 2
    end
    self_quad = (mid, half_ds, g_nodes, g_weights) ->
        _log_self_seg_quad(half_ds) +
        _gl3_pair_quad(mid, half_ds, mid, half_ds, g_nodes, g_weights, Φ_corr)
    return _energy_contour_pair(ci, cj, Φ, self_quad; _partial=_partial)
end

function _energy_contour_pair_euler_periodic(ci::PVContour{T}, cj::PVContour{T},
                                              cache::EwaldCache{T},
                                              domain::PeriodicDomain{T};
                                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # For G_k = 1/(A|k|²), choose φ_k = -1/(A|k|⁴), so Δφ = G.
    # The shared normalization then requires the contour integrand 4πφ.
    # This k⁻⁴ series is smooth, including for coincident segments.
    area = T(4) * domain.Lx * domain.Ly
    kx, ky = cache.kx, cache.ky
    Φ = rv -> begin
        val = zero(T)
        for kxi in kx
            cx = cos(kxi * rv[1])
            sx = sin(kxi * rv[1])
            for kyi in ky
                k2 = kxi * kxi + kyi * kyi
                k2 < eps(T) && continue
                phase_cos = cx * cos(kyi * rv[2]) - sx * sin(kyi * rv[2])
                val -= T(4) * T(π) * phase_cos / (area * k2 * k2)
            end
        end
        val
    end
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

"""
Radial potential whose 2-D Laplacian is `erfc(alpha * r) / r`, up to an
irrelevant additive constant.  The constant is chosen so `phi(0) = 0`, which
reduces cancellation in closed-contour double integrals.
"""
@inline function _sqg_ewald_real_potential(r::T, alpha::T) where {T}
    r <= zero(T) && return zero(T)

    # Near zero, log(r) + E1(alpha^2 r^2)/2 is finite but suffers cancellation.
    # Use the series form there and the direct expression elsewhere.
    inv_alpha_sqrtpi = one(T) / (alpha * sqrt(T(π)))
    ar = alpha * r
    z = ar * ar
    gamma_euler = T(Base.MathConstants.eulergamma)

    log_plus_half_e1 = if z < T(0.25)
        s = -log(alpha) - gamma_euler / 2
        term = one(T)
        max_terms = max(60, ceil(Int, -2 * log(eps(T))))
        for n in 1:max_terms
            term *= -z / T(n)
            incr = -term / (2 * T(n))
            s += incr
            abs(incr) < eps(T) * max(one(T), abs(s)) && break
        end
        s
    else
        log(r) + _expint_e1(z) / 2
    end

    zero_limit = (-one(T) - gamma_euler / 2 - log(alpha)) * inv_alpha_sqrtpi
    return r * erfc(ar) - exp(-z) * inv_alpha_sqrtpi +
        log_plus_half_e1 * inv_alpha_sqrtpi - zero_limit
end

function _eval_sqg_periodic_energy_potential(r_vec::SVector{2,T},
                                             cache::EwaldCache{T},
                                             domain::PeriodicDomain{T},
                                             delta::T) where {T}
    # Apply the SQG softening to every periodic image.  For each image,
    #
    #   Φ_real(r) + Φδ(r) - r
    #
    # has Laplacian erfc(αr)/r + 1/sqrt(r²+δ²) - 1/r.  Together with the
    # unregularized Fourier part this is exactly the Ewald decomposition used by
    # the periodic velocity kernel.
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    phi = zero(T)

    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            rv = r_vec - shift
            r2 = rv[1]^2 + rv[2]^2
            r = sqrt(r2)
            phi += _sqg_ewald_real_potential(r, alpha) +
                   _sqg_regularized_energy_potential_scalar(r2, delta) - r
        end
    end

    for (mi, kxi) in enumerate(cache.kx)
        for (ni, kyi) in enumerate(cache.ky)
            k2 = kxi^2 + kyi^2
            k2 < eps(T) && continue
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            phase = kxi * r_vec[1] + kyi * r_vec[2]
            phi -= coeff * cos(phase) / k2
        end
    end

    return phi
end

function _energy_contour_pair_sqg_periodic(ci::PVContour{T}, cj::PVContour{T},
                                           cache::EwaldCache{T},
                                           domain::PeriodicDomain{T},
                                           delta::T;
                                           _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Periodic SQG pair energy has no special self-segment branch here because
    # delta regularization keeps the potential finite at coincident quadrature
    # points.
    Lx2, Ly2 = _period_lengths(domain.Lx, domain.Ly)
    Φ = rv -> begin
        r_vec = SVector{2,T}(rv[1] - round(rv[1] / Lx2) * Lx2,
                             rv[2] - round(rv[2] / Ly2) * Ly2)
        _eval_sqg_periodic_energy_potential(r_vec, cache, domain, delta)
    end
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end
