# Top-level FMM driver: upward pass, downward pass, local evaluation.

"""Opening-angle parameter for the production treecode path."""
const _TREECODE_THETA = 0.15

@inline function _treecode_accepts(target_box::FMMBox{T}, source_box::FMMBox{T},
                                   theta::T = T(_TREECODE_THETA)) where {T}
    dx = source_box.center[1] - target_box.center[1]
    dy = source_box.center[2] - target_box.center[2]
    dist = sqrt(dx * dx + dy * dy)
    dist <= eps(T) && return false
    return (target_box.half_width + source_box.half_width) / dist <= theta
end

function _box_direct_velocity(tree::FMMTree{T},
                              contours::AbstractVector{PVContour{T}},
                              box::FMMBox{T},
                              kernel::AbstractKernel,
                              domain::AbstractDomain,
                              x::SVector{2,T},
                              ewald_cache) where {T}
    v = zero(SVector{2,T})
    for seg_idx in box.segment_range
        ci, ni = tree.sorted_segments[seg_idx]
        c = contours[ci]
        a = c.nodes[ni]
        b = next_node(c, ni)
        v = v + c.pv * segment_velocity(kernel, domain, x, a, b, ewald_cache)
    end
    return v
end

function _treecode_box_to_leaf!(vel::Vector{SVector{2,T}},
                                tree::FMMTree{T},
                                target_leaf_idx::Int,
                                source_box_idx::Int,
                                contours::AbstractVector{PVContour{T}},
                                flat_indices::Vector{Int},
                                kernel::AbstractKernel,
                                domain::AbstractDomain,
                                ewald_cache) where {T}
    target_box = tree.boxes[target_leaf_idx]
    source_box = tree.boxes[source_box_idx]
    center = target_box.center

    if source_box_idx in tree.near_lists[target_leaf_idx] || !_treecode_accepts(target_box, source_box)
        for target_seg_idx in target_box.segment_range
            flat_idx = flat_indices[target_seg_idx]
            ci_t, ni_t = tree.sorted_segments[target_seg_idx]
            x = contours[ci_t].nodes[ni_t]
            vel[flat_idx] = vel[flat_idx] + _box_direct_velocity(
                tree, contours, source_box, kernel, domain, x, ewald_cache)
        end
        return nothing
    end

    h = max(target_box.half_width / 2, sqrt(eps(T)))
    ex = SVector{2,T}(h, zero(T))
    ey = SVector{2,T}(zero(T), h)

    v0 = _box_direct_velocity(tree, contours, source_box, kernel, domain, center, ewald_cache)
    vxp = _box_direct_velocity(tree, contours, source_box, kernel, domain, center + ex, ewald_cache)
    vxm = _box_direct_velocity(tree, contours, source_box, kernel, domain, center - ex, ewald_cache)
    vyp = _box_direct_velocity(tree, contours, source_box, kernel, domain, center + ey, ewald_cache)
    vym = _box_direct_velocity(tree, contours, source_box, kernel, domain, center - ey, ewald_cache)

    j11 = (vxp[1] - vxm[1]) / (2h)
    j21 = (vxp[2] - vxm[2]) / (2h)
    j12 = (vyp[1] - vym[1]) / (2h)
    j22 = (vyp[2] - vym[2]) / (2h)

    for target_seg_idx in target_box.segment_range
        flat_idx = flat_indices[target_seg_idx]
        ci_t, ni_t = tree.sorted_segments[target_seg_idx]
        x = contours[ci_t].nodes[ni_t]
        dx = x - center
        vel[flat_idx] = vel[flat_idx] + SVector{2,T}(
            v0[1] + j11 * dx[1] + j12 * dx[2],
            v0[2] + j21 * dx[1] + j22 * dx[2],
        )
    end

    return nothing
end

function _treecode_velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    contours = prob.contours
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    fill!(vel, zero(SVector{2,T}))
    isempty(contours) && return vel

    tree = build_fmm_tree(contours)
    kernel = prob.kernel
    domain = prob.domain
    ewald = _prefetch_ewald(domain, kernel)

    offsets = Vector{Int}(undef, length(contours) + 1)
    offsets[1] = 0
    for ci in eachindex(contours)
        offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
    end

    flat_indices = Vector{Int}(undef, length(tree.sorted_segments))
    for seg_idx in eachindex(tree.sorted_segments)
        ci, ni = tree.sorted_segments[seg_idx]
        flat_indices[seg_idx] = offsets[ci] + ni
    end

    Threads.@threads for li_idx in eachindex(tree.leaf_indices)
        target_leaf_idx = tree.leaf_indices[li_idx]
        stack = Int[1]
        while !isempty(stack)
            source_box_idx = pop!(stack)
            source_box = tree.boxes[source_box_idx]
            if source_box.is_leaf || source_box_idx in tree.near_lists[target_leaf_idx] ||
               _treecode_accepts(tree.boxes[target_leaf_idx], source_box)
                _treecode_box_to_leaf!(vel, tree, target_leaf_idx, source_box_idx,
                                       contours, flat_indices, kernel, domain, ewald)
            else
                for child_idx in source_box.children
                    child_idx == 0 || push!(stack, child_idx)
                end
            end
        end
    end

    return vel
end

"""
    _near_field!(vel, tree, contours, kernel, domain, ewald_cache)

Evaluate near-field contributions for all leaf boxes using direct summation.
For each target node in a leaf box, sum velocity contributions from all
segments in the near-list boxes via `segment_velocity`.
"""
function _near_field!(vel::Vector{SVector{2,T}}, tree::FMMTree{T},
                      contours::AbstractVector{PVContour{T}}, kernel::AbstractKernel,
                      domain::AbstractDomain, ewald_cache,
                      flat_indices::Vector{Int}) where {T}
    Threads.@threads for li_idx in 1:length(tree.leaf_indices)
        leaf = tree.leaf_indices[li_idx]
        box = tree.boxes[leaf]
        for seg_idx in box.segment_range
            ci_t, ni_t = tree.sorted_segments[seg_idx]
            xi = contours[ci_t].nodes[ni_t]
            flat_idx = flat_indices[seg_idx]
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
    # The proxy-surface FMM pipeline is structurally complete but not yet
    # numerically validated to production accuracy. Gate behind the
    # _FMM_ACCELERATION_ENABLED flag; when disabled, delegate to direct.
    if !_FMM_ACCELERATION_ENABLED
        return _direct_velocity!(vel, prob)
    end

    contours = prob.contours
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    fill!(vel, zero(SVector{2,T}))
    isempty(contours) && return vel

    tree = build_fmm_tree(contours)
    kernel = prob.kernel
    domain = prob.domain
    ewald = _prefetch_ewald(domain, kernel)

    p = _FMM_PROXY_ORDER
    p_check = _FMM_CHECK_ORDER
    nboxes = length(tree.boxes)

    # Allocate proxy data for all boxes
    proxy_data = [ProxyData(zeros(SVector{2,T}, p), SVector{2,T}[]) for _ in 1:nboxes]

    # Precompute operators
    ops = precompute_level_operators(tree, kernel, domain; p, p_check)
    m2l_op = precompute_m2l_operators(tree, kernel, domain, ops; p, p_check)

    # Precompute flat indices for O(1) velocity writes
    offsets = Vector{Int}(undef, length(contours) + 1)
    offsets[1] = 0
    for ci in eachindex(contours)
        offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
    end
    flat_indices = Vector{Int}(undef, length(tree.sorted_segments))
    for seg_idx in eachindex(tree.sorted_segments)
        ci, ni = tree.sorted_segments[seg_idx]
        flat_indices[seg_idx] = offsets[ci] + ni
    end

    # Full FMM pipeline
    _s2m!(proxy_data, tree, contours, kernel, domain, ops, ewald; p, p_check)
    _m2m_upward!(proxy_data, tree, ops; p)
    _m2l!(proxy_data, tree, m2l_op; p)
    _l2l_downward!(proxy_data, tree, ops; p)
    _local_eval!(vel, tree, proxy_data, contours, kernel, domain, flat_indices; p)
    _near_field!(vel, tree, contours, kernel, domain, ewald, flat_indices)

    return vel
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

    # Pin BLAS to one thread to avoid nested threading with Julia's @threads
    # (the least-squares solves below dispatch to LAPACK).
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)

    Threads.@threads for li_idx in 1:length(leaves)
        leaf = leaves[li_idx]
        box = tree.boxes[leaf]

        # Dual check surfaces (consistent with _s2m!)
        check_pts_inner = _check_points(box.center, box.half_width, p_check)
        check_pts_outer = _check_points(box.center, box.half_width, p_check; radius_ratio=T(4))
        check_pts = vcat(check_pts_inner, check_pts_outer)
        n_check = length(check_pts)

        vel_check_x = Vector{T}(undef, n_check)
        vel_check_y = Vector{T}(undef, n_check)

        @inbounds for ic in 1:n_check
            xc = check_pts[ic]
            vx = zero(T)
            vy = zero(T)
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
                v = source_weight * c.pv * segment_velocity(kernel, domain, xc, a, b, nothing)
                vx += v[1]
                vy += v[2]
            end
            vel_check_x[ic] = vx
            vel_check_y[ic] = vy
        end

        # Fit equivalent-source strengths (same approach as _s2m!)
        proxy_pts = _proxy_points(box.center, box.half_width, p)
        K_cp = _build_kernel_matrix(kernel, domain, check_pts, proxy_pts)

        if kernel isa EulerKernel
            K_aug = vcat(K_cp, reshape(fill(one(T), p), 1, p))
            rhs_x = vcat(vel_check_x, zero(T))
            rhs_y = vcat(vel_check_y, zero(T))
        else
            K_aug = K_cp
            rhs_x = vel_check_x
            rhs_y = vel_check_y
        end
        str_x = K_aug \ rhs_x
        str_y = K_aug \ rhs_y

        equiv = proxy_data[leaf].equiv_strengths
        for k in 1:p
            equiv[k] = SVector{2,T}(str_x[k], str_y[k])
        end
    end

    BLAS.set_num_threads(prev_blas_threads)
end

"""
    _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets, prob, P, mode, kernel, domain, NL)

Evaluate local expansion + near field for one mode and accumulate into per-layer
velocity arrays with modal projection weights.
"""
function _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets,
                            prob, P, mode, kernel, domain, NL,
                            node_to_leaf::Dict{Tuple{Int,Int}, Int};
                            p=_FMM_PROXY_ORDER)
    T = eltype(tree.boxes[1].center)

    for target_layer in 1:NL
        weight = P[target_layer, mode]
        abs(weight) < eps(T) && continue

        target_contours = prob.layers[target_layer]
        vel_layer = vel[target_layer]

        # Map from layer-local contour index to global (all_contours) contour index
        ci_offset = layer_offsets[target_layer]

        # For each target node in this layer, evaluate both local and near field
        node_idx = 0
        for (tci, tc) in enumerate(target_contours)
            global_ci = ci_offset + tci
            for ti in 1:nnodes(tc)
                node_idx += 1
                xi = tc.nodes[ti]

                v_local = zero(SVector{2,T})
                v_near = zero(SVector{2,T})

                # Find the leaf containing this target point via precomputed mapping
                li = get(node_to_leaf, (global_ci, ti), 0)
                if li > 0
                    box = tree.boxes[li]

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
                end

                vel_layer[node_idx] += weight * (v_local + v_near)
            end
        end
    end
end

"""
    _build_node_to_leaf(tree) -> Dict{Tuple{Int,Int}, Int}

Build a mapping from `(contour_idx, node_idx)` to leaf box index using
the tree's segment assignment. O(N) construction, O(1) lookup.
"""
function _build_node_to_leaf(tree::FMMTree{T}) where {T}
    mapping = Dict{Tuple{Int,Int}, Int}()
    for leaf_idx in tree.leaf_indices
        box = tree.boxes[leaf_idx]
        for seg_idx in box.segment_range
            ci, ni = tree.sorted_segments[seg_idx]
            mapping[(ci, ni)] = leaf_idx
        end
    end
    return mapping
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
