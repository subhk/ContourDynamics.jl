# Shared multi-layer QG energy orchestration.

@inline _modal_energy_cache(::UnboundedDomain, ::AbstractKernel) = nothing
@inline _modal_energy_cache(domain::PeriodicDomain, mode_kernel::AbstractKernel) = _get_ewald_cache(domain, mode_kernel)

@inline function _modal_pair_energy(ci, cj, ::EulerKernel,
                                    ::UnboundedDomain, ::Nothing, partial)
    return _energy_contour_pair_euler(ci, cj; _partial=partial)
end

@inline function _modal_pair_energy(ci, cj, mode_kernel::QGKernel,
                                    ::UnboundedDomain, ::Nothing, partial)
    return _energy_contour_pair_qg(ci, cj, mode_kernel.Ld; _partial=partial)
end

@inline function _modal_pair_energy(ci, cj, ::EulerKernel,
                                    domain::PeriodicDomain, cache, partial)
    return _energy_contour_pair_euler_periodic(ci, cj, cache, domain; _partial=partial)
end

@inline function _modal_pair_energy(ci, cj, mode_kernel::QGKernel,
                                    domain::PeriodicDomain, cache, partial)
    return _energy_contour_pair_qg_periodic(ci, cj, cache, domain, mode_kernel.Ld; _partial=partial)
end

function _multilayer_mode_pair_energy(
        mode_kernel::MK, prob::MultiLayerContourProblem{N,K,D,T},
        mode::Int, partial::Vector{T}) where {N,K,D,T,MK}
    to_modal = prob.kernel.physical_to_modal
    cache = _modal_energy_cache(prob.domain, mode_kernel)
    result = zero(T)

    for source_layer in 1:N
        source_weight = to_modal[mode, source_layer]
        abs(source_weight) < eps(T) && continue
        for target_layer in 1:N
            target_weight = to_modal[mode, target_layer]
            abs(target_weight) < eps(T) && continue
            modal_weight = source_weight * target_weight
            for source in prob.layers[source_layer]
                _valid_energy_contour(source) || continue
                for target in prob.layers[target_layer]
                    _valid_energy_contour(target) || continue
                    pair_energy = _modal_pair_energy(
                        source, target, mode_kernel, prob.domain, cache, partial)
                    result += modal_weight * source.pv * target.pv * pair_energy
                end
            end
        end
    end
    return result
end

@inline _periodic_modal_zero_energy(::EulerKernel, to_modal, mode, layer_circulation, area) = zero(area)

@inline function _periodic_modal_zero_energy(mode_kernel::QGKernel, to_modal, mode, layer_circulation, area)
    circulation = sum(to_modal[mode, layer] * layer_circulation[layer]
                      for layer in eachindex(layer_circulation))

    kappa2 = inv(mode_kernel.Ld * mode_kernel.Ld)

    return circulation * circulation / (2 * area * kappa2)
end

function _periodic_multilayer_mode_energy(mode_kernel, prob, mode, partial, layer_circulation, area)
    pair_energy = _multilayer_mode_pair_energy(mode_kernel, prob, mode, partial)
    zero_energy = _periodic_modal_zero_energy(mode_kernel, prob.kernel.physical_to_modal, mode, layer_circulation, area)
    return pair_energy, zero_energy
end
