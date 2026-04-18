# Periodic-domain multi-layer QG energy diagnostics.

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T}) where {N, K, T}
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    E = zero(T)

    euler_cache = _get_ewald_cache(domain, EulerKernel())
    area = 4 * domain.Lx * domain.Ly
    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        for li in 1:N
            wi = P_inv[mode, li]
            abs(wi) < eps(T) && continue
            for lj in 1:N
                wj = P_inv[mode, lj]
                abs(wj) < eps(T) && continue
                for ci in prob.layers[li]
                    nci = nnodes(ci)
                    nci < 3 && continue
                    is_spanning(ci) && continue
                    for cj in prob.layers[lj]
                        ncj = nnodes(cj)
                        ncj < 3 && continue
                        is_spanning(cj) && continue
                        if abs(lam) < eps(T) * 100
                            pair_E = _energy_contour_pair_euler_periodic(ci, cj, euler_cache, domain; _partial=_partial)
                        else
                            kappa2 = abs(lam)
                            pair_E = _energy_contour_pair_euler_periodic(ci, cj, euler_cache, domain; _partial=_partial)
                            pair_E += _energy_contour_pair_qg_correction(ci, cj, euler_cache, kappa2, area; _partial=_partial)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end
