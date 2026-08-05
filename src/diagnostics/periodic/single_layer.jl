# Periodic-domain single-layer diagnostics.

function energy(prob::ContourProblem{EulerKernel, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv *
             _energy_contour_pair_euler_periodic(ci, cj, cache, prob.domain; _partial=partial)
    end
    return _normalize_energy(E)
end

function energy(prob::ContourProblem{QGKernel{T}, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv * _energy_contour_pair_qg_periodic(
            ci, cj, cache, prob.domain, prob.kernel.Ld; _partial=partial)
    end
    area = T(4) * prob.domain.Lx * prob.domain.Ly
    zero_mode = circulation(prob)^2 * prob.kernel.Ld^2 / (T(2) * area)
    return _normalize_energy(E) + zero_mode
end

"""QG-Euler correction for periodic energy: smooth Fourier series with -κ²/(k²(k²+κ²)) coefficients (precomputed in `cache.corr_coeffs`)."""
function _energy_contour_pair_qg_correction(ci::PVContour{T}, cj::PVContour{T},
                                             cache::EwaldCache{T};
                                             _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    corr_coeffs = cache.corr_coeffs
    kx = cache.kx
    ky = cache.ky
    nkx = length(kx)
    nky = length(ky)
    # Smooth everywhere — a pure Fourier series, so no self-segment branch.
    Φ = rv -> begin
        G_corr = zero(T)
        for mi in 1:nkx
            kxi = kx[mi]
            for ni in 1:nky
                coeff = corr_coeffs[mi, ni]
                iszero(coeff) && continue
                phase = kxi * rv[1] + ky[ni] * rv[2]
                G_corr -= coeff * cos(phase)
            end
        end
        -2 * T(π) * G_corr
    end
    return _energy_contour_pair(ci, cj, Φ; _partial=_partial)
end

function energy(prob::ContourProblem{SQGKernel{T}, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    δ = prob.kernel.δ
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv *
             _energy_contour_pair_sqg_periodic(ci, cj, cache, prob.domain, δ; _partial=partial)
    end
    # Remove the constant Ewald coefficient: periodic SQG inversion acts on the
    # mean-free scalar, so k=0 is not part of the Hamiltonian.
    zero_mode = _sqg_periodic_ewald_zero_mode(cache, prob.domain, δ)
    return _normalize_energy(E) - zero_mode * circulation(prob)^2 / T(2)
end
