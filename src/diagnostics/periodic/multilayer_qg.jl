# Periodic-domain multi-layer QG energy diagnostics.

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T}) where {N, K, T}
    kernel = prob.kernel
    domain = prob.domain
    E = zero(T)
    E_zero = zero(T)
    area = T(4) * domain.Lx * domain.Ly
    layer_circulation = ntuple(layer -> sum(
        c.pv * vortex_area(c) for c in prob.layers[layer]
        if _valid_energy_contour(c); init=zero(T)), Val(N))
            
    partial = zeros(T, maximum(
        nnodes(c) for layer in prob.layers for c in layer
        if _valid_energy_contour(c); init=0))

    for (mode, λ) in pairs(kernel.eigenvalues)
        mode_pair, mode_zero = _dispatch_qg_mode(
            _periodic_multilayer_mode_energy, kernel, λ, prob, mode,
            partial, layer_circulation, area)
        E += mode_pair
        E_zero += mode_zero
    end

    return _normalize_energy(E) + E_zero
end
