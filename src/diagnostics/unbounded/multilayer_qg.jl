# Unbounded-domain multi-layer QG energy diagnostics.

function energy(prob::MultiLayerContourProblem{N, K, UnboundedDomain, T}) where {N, K, T}
    kernel = prob.kernel
    E = zero(T)
    partial = zeros(T, maximum(
        nnodes(c) for layer in prob.layers for c in layer
        if _valid_energy_contour(c); init=0))

    for (mode, λ) in pairs(kernel.eigenvalues)
        E += _dispatch_qg_mode(
            _multilayer_mode_pair_energy, kernel, λ, prob, mode, partial)
    end

    return _normalize_energy(E)
end
