# Unbounded-domain multi-layer QG energy diagnostics.

function energy(prob::MultiLayerContourProblem{N, K, UnboundedDomain, T}) where {N, K, T}
    # Diagonalize the vertical coupling and evaluate energy mode-by-mode. The
    # barotropic zero mode uses Euler energy; nonzero modes use QG energy with
    # modal deformation radius 1/sqrt(abs(lambda)).
    kernel = prob.kernel
    evals = kernel.eigenvalues
    P_inv = kernel.physical_to_modal
    E = zero(T)

    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        # The weighted physical-to-modal map projects layer PV jumps into modal
        # amplitudes. The
        # pair loops remain over physical contours so existing single-layer
        # pair-integral helpers can be reused.
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
                        if _is_barotropic_mode(kernel, lam)
                            pair_E = _energy_contour_pair_euler(ci, cj; _partial=_partial)
                        else
                            Ld_mode = one(T) / sqrt(abs(lam))
                            pair_E = _energy_contour_pair_qg(ci, cj, Ld_mode; _partial=_partial)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    return _normalize_energy(E)
end
