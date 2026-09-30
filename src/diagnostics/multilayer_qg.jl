# Shared multi-layer QG energy orchestration.

@inline _modal_energy_cache(::UnboundedDomain, ::AbstractKernel) = nothing
@inline _modal_energy_cache(domain::PeriodicDomain, mode_kernel::AbstractKernel) = _get_ewald_cache(domain, mode_kernel)

# Fourier part of a periodic mode's energy (by Parseval) and the real-space
# cache for its contour pairs; unbounded modes have neither.
@inline _modal_energy_split(::Nothing, mode_kernel, layers, to_modal, mode, ::Type{T}) where {T} =
    zero(T), nothing

function _modal_energy_split(cache::EwaldCache{T}, mode_kernel, layers::NTuple{N},
                             to_modal, mode::Int, ::Type{T}) where {N, T}
    groups = ((layers[layer], T(to_modal[mode, layer])) for layer in 1:N
              if abs(to_modal[mode, layer]) >= eps(T))
    fourier = _parseval_energy(_energy_far_coefficients(cache, mode_kernel), cache, groups)
    return fourier, _real_space_cache(cache)
end

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
    layers = _host_contours(prob)
    result, cache = _modal_energy_split(_modal_energy_cache(prob.domain, mode_kernel),
                                        mode_kernel, layers, to_modal, mode, T)

    # The pair integrand is symmetric, so each unordered pair of (layer,
    # contour) sources is visited once and pairs of distinct contours count
    # twice, as in `@_valid_contour_pairs`.
    for source_layer in 1:N
        source_weight = to_modal[mode, source_layer]
        abs(source_weight) < eps(T) && continue
        for (si, source) in pairs(layers[source_layer])
            _valid_energy_contour(source) || continue
            for target_layer in source_layer:N
                target_weight = to_modal[mode, target_layer]
                abs(target_weight) < eps(T) && continue
                targets = layers[target_layer]
                first_target = target_layer == source_layer ? si : firstindex(targets)
                for ti in first_target:lastindex(targets)
                    target = targets[ti]
                    _valid_energy_contour(target) || continue
                    mult = target_layer == source_layer && ti == si ? 1 : 2
                    pair_energy = _modal_pair_energy(
                        source, target, mode_kernel, prob.domain, cache, partial)
                    result += mult * source_weight * target_weight *
                              source.pv * target.pv * pair_energy
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
