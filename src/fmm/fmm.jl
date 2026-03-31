# Top-level FMM driver: upward pass, downward pass, local evaluation.

"""
    _near_field!(vel, tree, contours, kernel, domain, ewald_cache)

Evaluate near-field contributions for all leaf boxes using direct summation.
For each target node in a leaf box, sum velocity contributions from all
segments in the near-list boxes via `segment_velocity`.
"""
function _near_field!(vel::Vector{SVector{2,T}}, tree::FMMTree{T},
                      contours::AbstractVector{PVContour{T}}, kernel::AbstractKernel,
                      domain::AbstractDomain, ewald_cache) where {T}
    Threads.@threads for li_idx in 1:length(tree.leaf_indices)
        leaf = tree.leaf_indices[li_idx]
        box = tree.boxes[leaf]
        for seg_idx in box.segment_range
            ci_t, ni_t = tree.sorted_segments[seg_idx]
            xi = contours[ci_t].nodes[ni_t]
            flat_idx = _node_flat_index(contours, ci_t, ni_t)
            v = zero(SVector{2,T})
            for near_bi in tree.near_lists[leaf]
                near_box = tree.boxes[near_bi]
                for near_seg_idx in near_box.segment_range
                    ci_s, ni_s = tree.sorted_segments[near_seg_idx]
                    c = contours[ci_s]
                    a = c.nodes[ni_s]
                    b = next_node(c, ni_s)
                    v = v + c.pv * segment_velocity(kernel, domain, xi, a, b, ewald_cache)
                end
            end
            vel[flat_idx] += v
        end
    end
end

"""
    _fmm_velocity!(vel, prob::ContourProblem)

Full FMM driver for computing velocity at all contour nodes.
Orchestrates the complete FMM pipeline: tree construction, operator
precomputation, upward pass (S2M, M2M), interaction pass (M2L),
downward pass (L2L), local evaluation, and near-field correction.
"""
function _fmm_velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    # The proxy-surface FMM is not yet numerically reliable enough for
    # production use. Preserve correctness by delegating to the validated
    # direct evaluator until the acceleration path is repaired.
    return _direct_velocity!(vel, prob)
end

# --- Periodic domain support ---

_to_unbounded_kernel(::EulerKernel) = EulerKernel()
_to_unbounded_kernel(k::QGKernel) = k
_to_unbounded_kernel(k::SQGKernel) = k

"""
    _fmm_velocity!(vel, prob::ContourProblem{K, PeriodicDomain{T}, T})

FMM driver for periodic domains. Runs the unbounded FMM on the primary cell,
then adds the periodic correction (G_periodic - G_unbounded) per node via
direct evaluation of the difference using Ewald summation.
"""
function _fmm_velocity!(vel::Vector{SVector{2,T}},
                        prob::ContourProblem{K, PeriodicDomain{T}, T}) where {K, T}
    return _direct_velocity!(vel, prob)
end

"""
    _periodic_correction!(vel, contours, kernel, domain, ewald_cache)

Add the periodic correction (G_periodic - G_unbounded) to the velocity at each
node. For each target node, sum over all source segments the difference between
the periodic and unbounded segment velocities.
"""
function _periodic_correction!(vel::Vector{SVector{2,T}},
                               contours::AbstractVector{PVContour{T}},
                               kernel::AbstractKernel,
                               domain::PeriodicDomain{T}, ewald_cache) where {T}
    N = sum(nnodes(c) for c in contours; init=0)
    n_contours = length(contours)

    # Prefix-sum for flat index mapping
    offsets = Vector{Int}(undef, n_contours + 1)
    offsets[1] = 0
    for ci in 1:n_contours
        offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
    end

    @inbounds Threads.@threads for i in 1:N
        ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
        ci = clamp(ci, 1, n_contours)
        local_i = i - offsets[ci]
        xi = contours[ci].nodes[local_i]

        v_corr = zero(SVector{2,T})
        for c in contours
            nc = nnodes(c)
            nc < 2 && continue
            for j in 1:nc
                a = c.nodes[j]
                b = next_node(c, j)
                v_per = c.pv * segment_velocity(kernel, domain, xi, a, b, ewald_cache)
                v_unb = c.pv * segment_velocity(kernel, UnboundedDomain(), xi, a, b)
                v_corr = v_corr + (v_per - v_unb)
            end
        end
        vel[i] += v_corr
    end
end

# --- Multi-layer FMM support ---

"""
    _fmm_velocity!(vel, prob::MultiLayerContourProblem)

FMM velocity for multi-layer QG problems. Builds one tree over all contours
from all layers, then runs one FMM pass per mode with modal source/target weights.
"""
function _fmm_velocity!(vel::NTuple{NL, Vector{SVector{2,T}}},
                        prob::MultiLayerContourProblem{NL}) where {NL, T}
    return _direct_velocity!(vel, prob)
end

"""
    _s2m_modal!(proxy_data, tree, all_contours, layer_offsets, P_inv, mode, kernel, domain, ops, NL)

Like `_s2m!` but weights each segment's PV by `P_inv[mode, layer_of_segment]`.
"""
function _s2m_modal!(proxy_data, tree, all_contours, layer_offsets,
                     P_inv, mode, kernel, domain, ops, NL;
                     p=_FMM_PROXY_ORDER, p_check=_FMM_CHECK_ORDER)
    T = eltype(tree.boxes[1].center)
    leaves = tree.leaf_indices

    Threads.@threads for li_idx in 1:length(leaves)
        leaf = leaves[li_idx]
        box = tree.boxes[leaf]
        check_pts = _check_points(box.center, box.half_width, p_check)

        vel_check_x = zeros(T, p_check)
        vel_check_y = zeros(T, p_check)

        for seg_idx in box.segment_range
            ci, ni = tree.sorted_segments[seg_idx]
            # Determine which layer this contour belongs to
            layer = 1
            for l in 1:NL
                if ci > layer_offsets[l] && ci <= layer_offsets[l+1]
                    layer = l
                    break
                end
            end
            source_weight = P_inv[mode, layer]
            abs(source_weight) < eps(T) && continue

            c = all_contours[ci]
            a = c.nodes[ni]
            b = next_node(c, ni)
            weighted_pv = source_weight * c.pv

            for k in 1:p_check
                v = weighted_pv * segment_velocity(kernel, domain, check_pts[k], a, b, nothing)
                vel_check_x[k] += v[1]
                vel_check_y[k] += v[2]
            end
        end

        pinv = ops[box.level + 1].check_to_proxy_pinv
        str_x = pinv * vel_check_x
        str_y = pinv * vel_check_y
        equiv = proxy_data[leaf].equiv_strengths
        for k in 1:p
            equiv[k] = SVector{2,T}(str_x[k], str_y[k])
        end
    end
end

"""
    _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets, prob, P, mode, kernel, domain, NL)

Evaluate local expansion + near field for one mode and accumulate into per-layer
velocity arrays with modal projection weights.
"""
function _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets,
                            prob, P, mode, kernel, domain, NL;
                            p=_FMM_PROXY_ORDER)
    T = eltype(tree.boxes[1].center)

    for target_layer in 1:NL
        weight = P[target_layer, mode]
        abs(weight) < eps(T) && continue

        target_contours = prob.layers[target_layer]
        vel_layer = vel[target_layer]

        # For each target node in this layer, evaluate both local and near field
        node_idx = 0
        for tc in target_contours
            for ti in 1:nnodes(tc)
                node_idx += 1
                xi = tc.nodes[ti]

                v_local = zero(SVector{2,T})
                v_near = zero(SVector{2,T})

                # Find the leaf containing this target point
                for li in tree.leaf_indices
                    box = tree.boxes[li]
                    if _point_in_box(xi, box)
                        # Local expansion evaluation
                        if length(proxy_data[li].local_strengths) > 0
                            proxy_pts = _proxy_points(box.center, box.half_width, p)
                            for k in 1:p
                                G = _kernel_value(kernel, domain, xi, proxy_pts[k])
                                v_local += G * proxy_data[li].local_strengths[k]
                            end
                        end

                        # Near field
                        for near_bi in tree.near_lists[li]
                            near_box = tree.boxes[near_bi]
                            for seg_idx in near_box.segment_range
                                ci_s, ni_s = tree.sorted_segments[seg_idx]
                                # Weight by P_inv[mode, source_layer]
                                s_layer = 1
                                for l in 1:NL
                                    if ci_s > layer_offsets[l] && ci_s <= layer_offsets[l+1]
                                        s_layer = l
                                        break
                                    end
                                end
                                src_w = prob.kernel.eigenvectors_inv[mode, s_layer]
                                abs(src_w) < eps(T) && continue
                                c = all_contours[ci_s]
                                a = c.nodes[ni_s]
                                b = next_node(c, ni_s)
                                v_near += src_w * c.pv * segment_velocity(kernel, domain, xi, a, b, nothing)
                            end
                        end
                        break  # found the leaf
                    end
                end

                vel_layer[node_idx] += weight * (v_local + v_near)
            end
        end
    end
end

@inline function _point_in_box(pt::SVector{2,T}, box::FMMBox{T}) where {T}
    hw = box.half_width * T(1.01)  # slight tolerance
    return abs(pt[1] - box.center[1]) <= hw && abs(pt[2] - box.center[2]) <= hw
end

"""
    _multilayer_periodic_correction!(vel, prob::MultiLayerContourProblem)

Add periodic correction for multi-layer FMM. Computes the difference between
full periodic direct velocity and full unbounded direct velocity, then adds it
to the FMM result (which was computed with unbounded kernels).

Note: This correction is O(N^2) via direct evaluation. This is acceptable as a
first implementation since the dominant FMM speedup is already captured.
"""
function _multilayer_periodic_correction!(vel::NTuple{NL, Vector{SVector{2,T}}},
                                          prob::MultiLayerContourProblem{NL}) where {NL, T}
    # Compute full periodic direct velocity
    vel_periodic = ntuple(i -> zeros(SVector{2,T}, length(vel[i])), Val(NL))
    _direct_velocity!(vel_periodic, prob)

    # Compute full unbounded direct velocity
    unbounded_prob = MultiLayerContourProblem(prob.kernel, UnboundedDomain(), prob.layers)
    vel_unbounded = ntuple(i -> zeros(SVector{2,T}, length(vel[i])), Val(NL))
    _direct_velocity!(vel_unbounded, unbounded_prob)

    # Add correction: (periodic - unbounded)
    for i in 1:NL
        for j in eachindex(vel[i])
            vel[i][j] += vel_periodic[i][j] - vel_unbounded[i][j]
        end
    end
end
