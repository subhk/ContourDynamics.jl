# Contour-dynamics beta-plane QG velocity.
#
# Live contours encode full PV.  The frozen straight beta staircase stored in
# `BetaPlaneQGKernel.reference_contours` is subtracted from the QG inversion,
# leaving only regular PV to drive the flow.

@inline _qg_kernel(kernel::BetaPlaneQGKernel{T}) where {T} = QGKernel(kernel.Ld)

@inline _prefetch_ewald(domain::PeriodicDomain, kernel::BetaPlaneQGKernel) =
    _prefetch_ewald(domain, _qg_kernel(kernel))

function _sum_qg_contour_velocity(kernel::QGKernel{T},
                                  domain::AbstractDomain,
                                  x::SVector{2,T},
                                  contours::Vector{PVContour{T}},
                                  curvatures,
                                  ewald) where {T}
    v = zero(SVector{2,T})
    for (ci, c) in pairs(contours)
        nc = nnodes(c)
        nc < 2 && continue
        κ = curvatures[ci]
        @inbounds for j in 1:nc
            v += c.pv * _segment_velocity_with_geometry(
                kernel, domain, x, c.nodes[j], next_node(c, j),
                κ[j], κ[mod1(j + 1, nc)], ewald)
        end
    end
    return v
end

@inline function _beta_plane_velocity_at(kernel::BetaPlaneQGKernel{T},
                                         domain::AbstractDomain,
                                         x::SVector{2,T},
                                         contours::Vector{PVContour{T}},
                                         contour_curvatures,
                                         reference_curvatures,
                                         ewald) where {T}
    qg = _qg_kernel(kernel)
    current = _sum_qg_contour_velocity(qg, domain, x, contours, contour_curvatures, ewald)
    reference = _sum_qg_contour_velocity(qg, domain, x, kernel.reference_contours,
                                         reference_curvatures, ewald)
    return current - reference
end

function _direct_velocity!(vel::Vector{SVector{2,T}},
                           prob::ContourProblem{BetaPlaneQGKernel{T}, D, T, CPU}) where {T, D<:AbstractDomain}
    kernel = prob.kernel
    domain = prob.domain
    contours = prob.contours
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))

    ewald = _prefetch_ewald(domain, kernel)
    contour_curvatures = [_signed_node_curvatures(c) for c in contours]
    reference_curvatures = [_signed_node_curvatures(c) for c in kernel.reference_contours]

    if _should_thread_velocity(N)
        n_contours = length(contours)
        offsets = Vector{Int}(undef, n_contours + 1)
        offsets[1] = 0
        for ci in 1:n_contours
            offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
        end
        Threads.@threads for i in 1:N
            ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
            ci = clamp(ci, 1, n_contours)
            local_i = i - offsets[ci]
            (1 <= local_i <= nnodes(contours[ci])) || throw(BoundsError(contours[ci].nodes, local_i))
            vel[i] = _beta_plane_velocity_at(kernel, domain, contours[ci].nodes[local_i],
                                             contours, contour_curvatures,
                                             reference_curvatures, ewald)
        end
    else
        idx = 1
        for target_contour in contours
            @inbounds for local_i in 1:nnodes(target_contour)
                vel[idx] = _beta_plane_velocity_at(kernel, domain, target_contour.nodes[local_i],
                                                   contours, contour_curvatures,
                                                   reference_curvatures, ewald)
                idx += 1
            end
        end
    end

    return vel
end

function velocity(prob::ContourProblem{BetaPlaneQGKernel{T}, D, T, CPU},
                  x::SVector{2,T}) where {T, D<:AbstractDomain}
    kernel = prob.kernel
    ewald = _prefetch_ewald(prob.domain, kernel)
    contour_curvatures = [_signed_node_curvatures(c) for c in prob.contours]
    reference_curvatures = [_signed_node_curvatures(c) for c in kernel.reference_contours]
    return _beta_plane_velocity_at(kernel, prob.domain, x, prob.contours,
                                   contour_curvatures, reference_curvatures, ewald)
end
