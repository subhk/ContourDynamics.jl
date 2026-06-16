# Periodic-domain single-layer diagnostics.

function energy(prob::ContourProblem{EulerKernel, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    inv4pi = one(T) / (4 * T(π))
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv *
             _energy_contour_pair_euler_periodic(ci, cj, cache, prob.domain; _partial=partial)
    end
    return -inv4pi * E / 2
end

function energy(prob::ContourProblem{QGKernel{T}, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    # Decompose: G_QG_per = G_Euler_per + G_correction, where the signed
    # correction has coefficients -κ²/(k²(k²+κ²)). The QG cache carries both the
    # Euler periodic coefficients and the precomputed correction coefficients.
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    inv4pi = one(T) / (4 * T(π))
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        pair_E = _energy_contour_pair_euler_periodic(ci, cj, cache, prob.domain; _partial=partial)
        pair_E += _energy_contour_pair_qg_correction(ci, cj, cache; _partial=partial)
        E += ci.pv * cj.pv * pair_E
    end
    return -inv4pi * E / 2
end

"""QG-Euler correction for periodic energy: smooth Fourier series with -κ²/(k²(k²+κ²)) coefficients (precomputed in `cache.corr_coeffs`)."""
function _energy_contour_pair_qg_correction(ci::PVContour{T}, cj::PVContour{T},
                                             cache::EwaldCache{T};
                                             _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    corr_coeffs = cache.corr_coeffs
    kx = cache.kx
    ky = cache.ky
    nkx = length(kx)
    nky = length(ky)
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    return @_energy_segment_loop partial _partial nci for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            quad = zero(T)
            for qi in 1:3
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:3
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    G_corr = zero(T)
                    for mi in 1:nkx
                        kxi = kx[mi]
                        for ni in 1:nky
                            coeff = corr_coeffs[mi, ni]
                            iszero(coeff) && continue
                            phase = kxi * dx + ky[ni] * dy
                            G_corr -= coeff * cos(phase)
                        end
                    end
                    quad += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_corr)
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
end

function energy(prob::ContourProblem{SQGKernel{T}, PeriodicDomain{T}, T}) where {T}
    prob.dev isa CPU || return _ka_energy(prob, prob.dev)
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    delta = prob.kernel.delta
    E = zero(T)
    @_valid_contour_pairs ci cj partial contours prob.velocity_scratch.energy_partial begin
        E += ci.pv * cj.pv *
             _energy_contour_pair_sqg_periodic(ci, cj, cache, prob.domain, delta; _partial=partial)
    end
    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end
