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
    E_zero = zero(T)
    area = T(4) * domain.Lx * domain.Ly
    layer_circulation = ntuple(li -> sum(
        c.pv * vortex_area(c) for c in prob.layers[li]
        if nnodes(c) >= 3 && !is_spanning(c); init=zero(T)), Val(N))

    euler_cache = _get_ewald_cache(domain, EulerKernel())
    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        # Each mode behaves like an Euler (λ≈0) or QG mode. QG modes use a cache
        # whose correction coefficients are precomputed for this mode's κ² = |λ|
        # (i.e. Ld = 1/√|λ|); the Euler periodic part reads the same cache.
        is_euler_mode = _is_barotropic_mode(kernel, lam)
        mode_cache = is_euler_mode ? euler_cache :
                     _get_ewald_cache(domain, QGKernel(one(T) / sqrt(abs(lam))))
        if !is_euler_mode
            gamma_mode = sum(P_inv[mode, li] * layer_circulation[li] for li in 1:N)
            E_zero += gamma_mode^2 / (T(2) * area * abs(lam))
        end
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
                        pair_E = is_euler_mode ?
                            _energy_contour_pair_euler_periodic(
                                ci, cj, mode_cache, domain; _partial=_partial) :
                            _energy_contour_pair_qg_periodic(
                                ci, cj, mode_cache, domain,
                                one(T) / sqrt(abs(lam)); _partial=_partial)
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    return _normalize_energy(E) + E_zero
end
