# Periodic-domain multi-layer QG energy diagnostics.

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T}) where {N, K, T}
    # Periodic multi-layer energy follows the unbounded modal decomposition, but
    # each modal pair integral uses the periodic Euler Ewald cache plus the QG
    # Fourier correction for nonzero coupling eigenvalues.
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    E = zero(T)

    euler_cache = _get_ewald_cache(domain, EulerKernel())
    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        # Each mode behaves like an Euler (λ≈0) or QG mode. QG modes use a cache
        # whose correction coefficients are precomputed for this mode's κ² = |λ|
        # (i.e. Ld = 1/√|λ|); the Euler periodic part reads the same cache.
        is_euler_mode = abs(lam) < eps(T) * 100
        mode_cache = is_euler_mode ? euler_cache :
                     _get_ewald_cache(domain, QGKernel(one(T) / sqrt(abs(lam))))
        # The inverse eigenvector weights convert physical-layer PV jumps into
        # modal PV jumps before the contour-pair Green's-function integral.
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
                        pair_E = if is_euler_mode
                            _energy_contour_pair_euler_periodic(
                                ci, cj, mode_cache, domain; _partial=_partial)
                        else
                            _energy_contour_pair_periodic_green(
                                ci, cj, mode_cache, domain; _partial=_partial)
                        end
                        if !is_euler_mode
                            pair_E += _energy_contour_pair_qg_correction(ci, cj, mode_cache; _partial=_partial)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    return _normalize_energy(E)
end
