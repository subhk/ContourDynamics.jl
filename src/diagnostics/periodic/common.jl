# Periodic-domain helper routines shared by diagnostics.

function _energy_contour_pair_euler_periodic(ci::PVContour{T}, cj::PVContour{T},
                                              cache::EwaldCache{T},
                                              domain::PeriodicDomain{T};
                                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # For G_k = 1/(A|k|²), choose φ_k = -1/(A|k|⁴), so Δφ = G.
    # The shared normalization then requires the contour integrand 4πφ.
    # This k⁻⁴ series is smooth, including for coincident segments.
    area = T(4) * domain.Lx * domain.Ly
    Φ = rv -> _periodic_energy_potential_scalar(
        rv[1], rv[2], zero(T), area, cache.kx, cache.ky)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function _energy_contour_pair_qg_periodic(ci::PVContour{T}, cj::PVContour{T},
                                          cache::EwaldCache{T},
                                          domain::PeriodicDomain{T}, Ld::T;
                                          _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # For G_k=1/[A(k²+κ²)], k≠0, the contour potential required by
    # the shared normalization is -4π cos(k·r)/[A k²(k²+κ²)].
    # The spatially constant k=0 energy is added by the problem-level caller.
    area = T(4) * domain.Lx * domain.Ly
    kappa2 = one(T) / (Ld * Ld)
    Φ = rv -> _periodic_energy_potential_scalar(
        rv[1], rv[2], kappa2, area, cache.kx, cache.ky)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

"""
Radial potential whose 2-D Laplacian is `erfc(α * r) / r`, up to an
irrelevant additive constant.  The constant is chosen so `phi(0) = 0`, which
reduces cancellation in closed-contour double integrals.
"""
@inline function _sqg_ewald_real_potential(r::T, α::T) where {T}
    r <= zero(T) && return zero(T)

    # Near zero, log(r) + E1(α^2 r^2)/2 is finite but suffers cancellation.
    # Use the series form there and the direct expression elsewhere.
    inv_α_sqrtpi = one(T) / (α * sqrt(T(π)))
    ar = α * r
    z = ar * ar
    γ_euler = T(Base.MathConstants.eulergamma)

    log_plus_half_e1 = if z < T(0.25)
        s = -log(α) - γ_euler / 2
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

    zero_limit = (-one(T) - γ_euler / 2 - log(α)) * inv_α_sqrtpi
    return r * erfc(ar) - exp(-z) * inv_α_sqrtpi +
        log_plus_half_e1 * inv_α_sqrtpi - zero_limit
end

function _eval_sqg_periodic_energy_potential(r_vec::SVector{2,T},
                                             cache::EwaldCache{T},
                                             domain::PeriodicDomain{T},
                                             δ::T) where {T}
    return _sqg_periodic_energy_potential_scalar(
        r_vec[1], r_vec[2], cache.α, domain.Lx, domain.Ly, δ,
        cache.n_images, cache.kx, cache.ky, cache.fourier_coeffs)
end

function _energy_contour_pair_sqg_periodic(ci::PVContour{T}, cj::PVContour{T},
                                           cache::EwaldCache{T},
                                           domain::PeriodicDomain{T},
                                           δ::T;
                                           _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Periodic SQG pair energy has no special self-segment branch here because
    # δ regularization keeps the potential finite at coincident quadrature
    # points.
    Lx2, Ly2 = _period_lengths(domain.Lx, domain.Ly)
    Φ = rv -> begin
        r_vec = SVector{2,T}(rv[1] - round(rv[1] / Lx2) * Lx2,
                             rv[2] - round(rv[2] / Ly2) * Ly2)
        _eval_sqg_periodic_energy_potential(r_vec, cache, domain, δ)
    end
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end
