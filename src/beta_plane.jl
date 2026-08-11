# Contour-dynamics beta-plane QG velocity.
#
# Live contours encode the full PV `q = q_s + q_r` in the notation of Lam &
# Dritschel (2001). The frozen straight beta staircase `q_ref = q_r(t=0)` stored
# in `BetaPlaneQGKernel.reference_contours` is subtracted from the contour sum,
# then an analytic zonal correction adds the velocity induced by
# `q_ref - beta*y`. By linearity the resulting inversion source is
#
#     (q - q_ref) + (q_ref - beta*y) = q - beta*y
#
# and hence the computed velocity is the total flow from `psi_s + psi_r`, not
# the regular component alone. This realizes the paper's decomposed inversion
# without introducing a grid solve.

@inline _qg_kernel(kernel::BetaPlaneQGKernel{T}) where {T} = QGKernel(kernel.Ld)

@inline _prefetch_ewald(domain::PeriodicDomain, kernel::BetaPlaneQGKernel) =
    _prefetch_ewald(domain, _qg_kernel(kernel))

# The beta-plane velocity path consumes the QG-keyed Ewald cache (via the
# delegation above), so the public cache API must accept BetaPlaneQGKernel too:
# `setup_ewald_cache!`/`build_ewald_cache` configure the same entry the
# velocity evaluation reads.
@inline _cache_key(domain::PeriodicDomain, kernel::BetaPlaneQGKernel) =
    _cache_key(domain, _qg_kernel(kernel))
@inline _kernel_value_precision(kernel::BetaPlaneQGKernel) =
    _kernel_value_precision(_qg_kernel(kernel))
build_ewald_cache(domain::PeriodicDomain{T}, kernel::BetaPlaneQGKernel{T};
                  n_fourier::Int=8, n_images::Int=2) where {T} =
    build_ewald_cache(domain, _qg_kernel(kernel);
                      n_fourier=n_fourier, n_images=n_images)
function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::BetaPlaneQGKernel{T};
                            n_fourier::Int=8,
                            n_images::Int=2) where {T<:Union{Float64, Float32}}
    return setup_ewald_cache!(domain, _qg_kernel(kernel);
                              n_fourier=n_fourier, n_images=n_images)
end
function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::BetaPlaneQGKernel{T};
                            n_fourier::Int=8,
                            n_images::Int=2) where {T<:AbstractFloat}
    return setup_ewald_cache!(domain, _qg_kernel(kernel);
                              n_fourier=n_fourier, n_images=n_images)
end

@inline function _beta_plane_sawtooth_velocity(kernel::BetaPlaneQGKernel{T},
                                               domain::PeriodicDomain{T},
                                               x::SVector{2,T}) where {T}
    n_beta = length(kernel.reference_contours)
    dy = 2 * domain.Ly / T(n_beta)
    κ = inv(kernel.Ld)
    ξ = mod(x[2] + domain.Ly + dy / 2, dy) - dy / 2

    return SVector{2,T}(_beta_sawtooth_u(kernel.beta, κ, dy, ξ), zero(T))
end

@inline function _beta_plane_velocity_at(kernel::BetaPlaneQGKernel{T},
                                         domain::PeriodicDomain{T},
                                         x::SVector{2,T},
                                         contours::Vector{PVContour{T}},
                                         contour_curvatures,
                                         reference_curvatures,
                                         ewald) where {T}
    # Both the live-contour and frozen-reference sums are the same pv-weighted
    # segment sweep the generic direct path uses; only the sources differ.
    qg = _qg_kernel(kernel)
    current = _accumulate_node_velocity(qg, domain, contours,
                                        contour_curvatures, ewald, x)
    reference = _accumulate_node_velocity(qg, domain, kernel.reference_contours,
                                          reference_curvatures, ewald, x)
    return current - reference + _beta_plane_sawtooth_velocity(kernel, domain, x)
end

function _direct_velocity!(vel::Vector{SVector{2,T}},
                           prob::ContourProblem{BetaPlaneQGKernel{T}, D, T, CPU}) where {T, D<:PeriodicDomain{T}}
    kernel = prob.kernel
    domain = prob.domain
    contours = prob.contours
    N = _validate_velocity_buffer!(vel, prob)

    ewald = _prefetch_ewald(domain, kernel)
    scratch = prob.velocity_scratch
    contour_curvatures = _prepare_curvature_buffers!(scratch.contour_curvatures, contours)
    reference_curvatures = _prepare_curvature_buffers!(scratch.reference_curvatures,
                                                       kernel.reference_contours)

    return _direct_velocity_loop!(vel, prob, N,
        xi -> _beta_plane_velocity_at(kernel, domain, xi, contours,
                                      contour_curvatures, reference_curvatures,
                                      ewald))
end

function velocity(prob::ContourProblem{BetaPlaneQGKernel{T}, D, T, CPU},
                  x::SVector{2,T}) where {T, D<:PeriodicDomain{T}}
    kernel = prob.kernel
    ewald = _prefetch_ewald(prob.domain, kernel)
    scratch = prob.velocity_scratch
    contour_curvatures = _prepare_curvature_buffers!(scratch.contour_curvatures,
                                                     prob.contours)
    reference_curvatures = _prepare_curvature_buffers!(scratch.reference_curvatures,
                                                       kernel.reference_contours)
    return _beta_plane_velocity_at(kernel, prob.domain, x, prob.contours,
                                   contour_curvatures, reference_curvatures, ewald)
end

function velocity(prob::ContourProblem{BetaPlaneQGKernel{T},D,T,CPU},
                  x::SVector{2,S}) where {T,S,D<:PeriodicDomain{T}}
    return velocity(prob, SVector{2,T}(x))
end
