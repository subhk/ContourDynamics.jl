# Periodic-domain helper routines shared by diagnostics.

function _energy_contour_pair_euler_periodic(ci::PVContour{T}, cj::PVContour{T},
                                              cache::EwaldCache{T},
                                              domain::PeriodicDomain{T};
                                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # For G_k = 1/(A|k|²), φ_k = -1/(A|k|⁴) satisfies Δφ = G; the shared
    # normalization requires the contour integrand 4πφ, Ewald split as the
    # κ → 0 limit of the scaled QG correction (numerics/periodic_ewald.jl).
    energy_table = _required_ewald_table(cache, cache.energy_cos, :energy_coeffs)
    Φ = rv -> _periodic_energy_potential_scalar(
        rv[1], rv[2], zero(T), cache.α, domain.Lx, domain.Ly, cache.n_images,
        cache.dkx, cache.dky, cache.fourier_cos, energy_table)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function _energy_contour_pair_qg_periodic(ci::PVContour{T}, cj::PVContour{T},
                                          cache::EwaldCache{T},
                                          domain::PeriodicDomain{T}, Ld::T;
                                          _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # For G_k=1/[A(k²+κ²)], k≠0, the contour potential required by the shared
    # normalization is -4π cos(k·r)/[A k²(k²+κ²)] = 4πĜ.
    # The spatially constant k=0 energy is added by the problem-level caller.
    energy_table = _required_ewald_table(cache, cache.energy_cos, :energy_coeffs)
    Φ = rv -> _periodic_energy_potential_scalar(
        rv[1], rv[2], one(T) / (Ld * Ld), cache.α, domain.Lx, domain.Ly,
        cache.n_images, cache.dkx, cache.dky, cache.fourier_cos, energy_table)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function _energy_contour_pair_sqg_periodic(ci::PVContour{T}, cj::PVContour{T},
                                           cache::EwaldCache{T},
                                           domain::PeriodicDomain{T},
                                           δ::T;
                                           _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    # Periodic SQG pair energy has no special self-segment branch here because
    # δ regularization keeps the potential finite at coincident quadrature
    # points.
    energy_table = _required_ewald_table(cache, cache.energy_cos, :energy_coeffs)
    Φ = rv -> _sqg_periodic_energy_potential_scalar(
        rv[1], rv[2], cache.α, domain.Lx, domain.Ly, δ, cache.n_images,
        cache.dkx, cache.dky, energy_table)
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end
