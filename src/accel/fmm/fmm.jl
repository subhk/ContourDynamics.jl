# Top-level FMM driver: upward pass, downward pass, local evaluation.

"""
Opening-angle parameter for the production treecode path.

The treecode uses a first-order Taylor expansion with finite-difference Jacobians
(5 evaluations per source box). The tight θ=0.15 compensates for the low-order
expansion, giving ~2e-3 relative error.

Performance/accuracy tradeoff:
- θ=0.15 + 1st order: ~2e-3 error, conservative (many direct evaluations)
- θ=0.3-0.5 + 2nd order: would give similar error with ~4x fewer direct evals
- Analytic Jacobians would eliminate the 5x FD overhead per accepted box

TODO: implement analytic kernel Jacobians and/or second-order expansion to allow
relaxing theta, significantly improving performance for large problems.
"""
const _TREECODE_THETA = 0.15

@inline function _treecode_accepts(target_box::FMMBox{T}, source_box::FMMBox{T},
                                   theta::T = T(_TREECODE_THETA)) where {T}
    dx = source_box.center[1] - target_box.center[1]
    dy = source_box.center[2] - target_box.center[2]
    dist = sqrt(dx * dx + dy * dy)
    dist <= eps(T) && return false
    return (target_box.half_width + source_box.half_width) / dist <= theta
end

"""
    _build_treecode_worklists(tree) -> (direct_lists, approx_lists)

Precompute the source-box worklists for each target leaf used by the production
treecode. `direct_lists[i]` contains source boxes that must be evaluated by
direct segment summation for target leaf `tree.leaf_indices[i]`; `approx_lists[i]`
contains accepted source boxes handled by the linearized far-field model.

This moves the adaptive DFS traversal out of the hot velocity loop and turns the
runtime treecode path into regular iteration over preclassified source-box lists.
"""
function _build_treecode_worklists(tree::FMMTree{T}) where {T}
    nleaves = length(tree.leaf_indices)
    direct_lists = [Int[] for _ in 1:nleaves]
    approx_lists = [Int[] for _ in 1:nleaves]

    for li_idx in 1:nleaves
        target_leaf_idx = tree.leaf_indices[li_idx]
        target_box = tree.boxes[target_leaf_idx]
        near = tree.near_lists[target_leaf_idx]
        direct = direct_lists[li_idx]
        approx = approx_lists[li_idx]

        stack = Int[1]
        while !isempty(stack)
            source_box_idx = pop!(stack)
            source_box = tree.boxes[source_box_idx]
            is_near = source_box_idx in near
            accepted = !is_near && _treecode_accepts(target_box, source_box)

            if source_box.is_leaf || is_near || accepted
                if is_near || !accepted
                    push!(direct, source_box_idx)
                else
                    push!(approx, source_box_idx)
                end
            else
                for child_idx in source_box.children
                    child_idx == 0 || push!(stack, child_idx)
                end
            end
        end
    end

    return direct_lists, approx_lists
end

"""
    _build_flat_indices(tree, contours) -> Vector{Int}

Map each segment in `tree.sorted_segments` back to the flat node index used by
the public velocity arrays. This lets the treecode and proxy paths accumulate
directly into flat buffers without repeatedly reconstructing contour offsets.
"""
function _build_flat_indices(tree::FMMTree, contours)
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
    return flat_indices
end

"""
    _build_segment_layers(tree, contour_layers) -> Vector{Int}

Expand contour-level layer ids to one layer id per sorted segment. The modal
multi-layer paths use this to avoid re-deriving a segment's source layer in the
inner loops.
"""
function _build_segment_layers(tree::FMMTree, contour_layers::AbstractVector{Int})
    segment_layers = Vector{Int}(undef, length(tree.sorted_segments))
    for seg_idx in eachindex(tree.sorted_segments)
        ci, _ = tree.sorted_segments[seg_idx]
        segment_layers[seg_idx] = contour_layers[ci]
    end
    return segment_layers
end

"""
    _build_leaf_proxy_geometry(tree; p, p_check, include_proxy_geometry)

Precompute the per-leaf proxy and dual-check surfaces used by the proxy FMM.
The treecode also carries the resulting arrays in `TreeEvalPlan`, but leaves
them empty unless proxy geometry is explicitly requested.
"""
function _build_leaf_proxy_geometry(tree::FMMTree{T}; p::Int = _FMM_PROXY_ORDER,
                                    p_check::Int = _FMM_CHECK_ORDER,
                                    include_proxy_geometry::Bool = false) where {T}
    nboxes = length(tree.boxes)
    leaf_proxy_points = [SVector{2,T}[] for _ in 1:nboxes]
    leaf_check_points = [SVector{2,T}[] for _ in 1:nboxes]
    include_proxy_geometry || return leaf_proxy_points, leaf_check_points

    for leaf_idx in tree.leaf_indices
        box = tree.boxes[leaf_idx]
        proxy_pts = _proxy_points(box.center, box.half_width, p)
        check_pts_inner = _check_points(box.center, box.half_width, p_check)
        check_pts_outer = _check_points(box.center, box.half_width, p_check; radius_ratio=T(4))
        check_pts = Vector{SVector{2,T}}(undef, length(check_pts_inner) + length(check_pts_outer))
        copyto!(check_pts, 1, check_pts_inner, 1, length(check_pts_inner))
        copyto!(check_pts, length(check_pts_inner) + 1, check_pts_outer, 1, length(check_pts_outer))
        leaf_proxy_points[leaf_idx] = proxy_pts
        leaf_check_points[leaf_idx] = check_pts
    end

    return leaf_proxy_points, leaf_check_points
end

"""
    _build_leaf_check_to_proxy(tree, kernel, domain, leaf_proxy_points, leaf_check_points;
                               include_proxy_geometry)

Precompute the dense leaf-local operators used by the proxy solve:

- `K(check, proxy)` for general kernels
- the Euler-augmented matrix with a final zero-sum row
- QR factorizations for both systems so the runtime leaf solve is just
  "fill the RHS, apply the cached least-squares operator"
"""
function _build_leaf_check_to_proxy(
    tree::FMMTree{T},
    kernel::AbstractKernel,
    domain::AbstractDomain,
    leaf_proxy_points::Vector{Vector{SVector{2,T}}},
    leaf_check_points::Vector{Vector{SVector{2,T}}};
    include_proxy_geometry::Bool = false,
) where {T}
    nboxes = length(tree.boxes)
    leaf_check_to_proxy = [Matrix{T}(undef, 0, 0) for _ in 1:nboxes]
    leaf_augmented_check_to_proxy = [Matrix{T}(undef, 0, 0) for _ in 1:nboxes]
    empty_qr = qr(Matrix{T}(undef, 0, 0))
    leaf_check_to_proxy_qr = [empty_qr for _ in 1:nboxes]
    leaf_augmented_check_to_proxy_qr = [empty_qr for _ in 1:nboxes]
    include_proxy_geometry || return leaf_check_to_proxy, leaf_augmented_check_to_proxy,
                                    leaf_check_to_proxy_qr, leaf_augmented_check_to_proxy_qr

    for leaf_idx in tree.leaf_indices
        proxy_pts = leaf_proxy_points[leaf_idx]
        check_pts = leaf_check_points[leaf_idx]
        K_cp = _build_kernel_matrix(kernel, domain, check_pts, proxy_pts)
        leaf_check_to_proxy[leaf_idx] = K_cp
        leaf_check_to_proxy_qr[leaf_idx] = qr(K_cp)
        if kernel isa EulerKernel
            K_aug = Matrix{T}(undef, size(K_cp, 1) + 1, size(K_cp, 2))
            copyto!(K_aug, 1, K_cp, 1, length(K_cp))
            @inbounds for j in axes(K_aug, 2)
                K_aug[end, j] = one(T)
            end
            leaf_augmented_check_to_proxy[leaf_idx] = K_aug
            leaf_augmented_check_to_proxy_qr[leaf_idx] = qr(K_aug)
        end
    end

    return leaf_check_to_proxy, leaf_augmented_check_to_proxy,
           leaf_check_to_proxy_qr, leaf_augmented_check_to_proxy_qr
end

"""
    _build_tree_eval_plan(tree, contours, contour_layers=fill(1, length(contours)); ...)

Build the shared evaluation plan used by the treecode and proxy-FMM code.

The plan intentionally groups together:
- geometry-independent traversal metadata (`direct_lists`, `approx_lists`)
- flat scatter/gather mappings (`flat_indices`, `node_to_leaf`)
- multi-layer bookkeeping (`segment_layers`)
- optional proxy-FMM geometry and dense operators

Keeping these pieces together makes the hot evaluation paths read as a sequence
of numerical stages instead of a mix of traversal setup and physics.
"""
function _build_tree_eval_plan(tree::FMMTree{T}, contours,
                               contour_layers::AbstractVector{Int}=fill(1, length(contours));
                               p::Int = _FMM_PROXY_ORDER,
                               p_check::Int = _FMM_CHECK_ORDER,
                               include_proxy_geometry::Bool = false,
                               kernel::AbstractKernel = EulerKernel(),
                               domain::AbstractDomain = UnboundedDomain()) where {T}
    direct_lists, approx_lists = _build_treecode_worklists(tree)
    flat_indices = _build_flat_indices(tree, contours)
    node_to_leaf = _build_node_to_leaf(tree)
    segment_layers = _build_segment_layers(tree, contour_layers)
    leaf_proxy_points, leaf_check_points = _build_leaf_proxy_geometry(
        tree; p, p_check, include_proxy_geometry)
    leaf_check_to_proxy, leaf_augmented_check_to_proxy,
    leaf_check_to_proxy_qr, leaf_augmented_check_to_proxy_qr = _build_leaf_check_to_proxy(
        tree, kernel, domain, leaf_proxy_points, leaf_check_points;
        include_proxy_geometry)
    return TreeEvalPlan(flat_indices, direct_lists, approx_lists, node_to_leaf,
                        segment_layers, leaf_proxy_points, leaf_check_points,
                        leaf_check_to_proxy, leaf_augmented_check_to_proxy,
                        leaf_check_to_proxy_qr, leaf_augmented_check_to_proxy_qr)
end

"""
    _apply_treecode_worklists!(vel, tree, contours, plan, kernel, domain, ewald_cache)

Execute the production treecode using the preclassified worklists stored in
`plan`. Each target leaf is processed independently, first by direct near-field
boxes and then by accepted far-field boxes evaluated with the linearized model.
"""
function _apply_treecode_worklists!(
    vel::Vector{SVector{2,T}},
    tree::FMMTree{T},
    contours::AbstractVector{PVContour{T}},
    plan::TreeEvalPlan{T},
    kernel::AbstractKernel,
    domain::AbstractDomain,
    ewald_cache,
    dev::AbstractDevice=CPU(),
) where {T}
    function process_leaf(li_idx::Int)
        target_leaf_idx = tree.leaf_indices[li_idx]
        for source_box_idx in plan.direct_lists[li_idx]
            _treecode_direct_to_leaf!(vel, tree, target_leaf_idx, source_box_idx,
                                      contours, plan, kernel, domain, ewald_cache, dev)
        end
        for source_box_idx in plan.approx_lists[li_idx]
            _treecode_linearized_to_leaf!(vel, tree, target_leaf_idx, source_box_idx,
                                          contours, plan, kernel, domain, ewald_cache, dev)
        end
        return nothing
    end

    if _should_thread_accelerator(length(tree.leaf_indices))
        Threads.@threads for li_idx in eachindex(tree.leaf_indices)
            process_leaf(li_idx)
        end
    else
        for li_idx in eachindex(tree.leaf_indices)
            process_leaf(li_idx)
        end
    end

    return nothing
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

@inline function _treecode_direct_ka_available(::AbstractDevice, ::AbstractKernel, ::AbstractDomain)
    return false
end

@inline function _treecode_direct_ka_available(dev::GPU,
                                               ::Union{EulerKernel,QGKernel,SQGKernel},
                                               ::Union{UnboundedDomain,PeriodicDomain})
    try
        _ka_backend(dev)
        return true
    catch
        return false
    end
end

function _ka_treecode_direct_to_leaf!(vel::Vector{SVector{2,T}},
                                      tree::FMMTree{T},
                                      target_leaf_idx::Int,
                                      source_box_idx::Int,
                                      contours::AbstractVector{PVContour{T}},
                                      plan::TreeEvalPlan,
                                      kernel::AbstractKernel,
                                      domain::AbstractDomain,
                                      dev::AbstractDevice) where {T}
    source_box = tree.boxes[source_box_idx]
    n_source = length(source_box.segment_range)
    n_source == 0 && return nothing

    ax = Vector{T}(undef, n_source)
    ay = Vector{T}(undef, n_source)
    bx = Vector{T}(undef, n_source)
    by = Vector{T}(undef, n_source)
    pv = Vector{T}(undef, n_source)
    k = 1
    for seg_idx in source_box.segment_range
        ci_s, ni_s = tree.sorted_segments[seg_idx]
        c = contours[ci_s]
        a = c.nodes[ni_s]
        b = next_node(c, ni_s)
        ax[k] = a[1]
        ay[k] = a[2]
        bx[k] = b[1]
        by[k] = b[2]
        pv[k] = c.pv
        k += 1
    end

    target_box = tree.boxes[target_leaf_idx]
    n_targets = length(target_box.segment_range)
    n_targets == 0 && return nothing
    tx = Vector{T}(undef, n_targets)
    ty = Vector{T}(undef, n_targets)
    flat_idxs = Vector{Int}(undef, n_targets)
    k = 1
    for target_seg_idx in target_box.segment_range
        flat_idxs[k] = plan.flat_indices[target_seg_idx]
        ci_t, ni_t = tree.sorted_segments[target_seg_idx]
        x = contours[ci_t].nodes[ni_t]
        tx[k] = x[1]
        ty[k] = x[2]
        k += 1
    end

    vx, vy = _ka_box_direct_velocity_at_targets(ax, ay, bx, by, pv, tx, ty, kernel, domain, dev)
    @inbounds for i in 1:n_targets
        vel[flat_idxs[i]] = vel[flat_idxs[i]] + SVector{2,T}(vx[i], vy[i])
    end
    return nothing
end

function _ka_box_direct_velocity_at_targets(ax::Vector{T}, ay::Vector{T}, bx::Vector{T},
                                            by::Vector{T}, pv::Vector{T},
                                            tx::Vector{T}, ty::Vector{T},
                                            kernel::AbstractKernel,
                                            domain::AbstractDomain,
                                            dev::AbstractDevice) where {T}
    n_targets = length(tx)
    seg = SegmentData(to_device(dev, ax), to_device(dev, ay),
                      to_device(dev, bx), to_device(dev, by),
                      to_device(dev, pv))
    dev_tx = to_device(dev, tx)
    dev_ty = to_device(dev, ty)
    dev_vx = device_zeros(dev, T, n_targets)
    dev_vy = device_zeros(dev, T, n_targets)

    _ka_velocity_subset!(dev_vx, dev_vy, dev_tx, dev_ty, seg, kernel, domain, dev)
    return to_cpu(dev_vx), to_cpu(dev_vy)
end

function _treecode_direct_to_leaf!(vel::Vector{SVector{2,T}},
                                   tree::FMMTree{T},
                                   target_leaf_idx::Int,
                                   source_box_idx::Int,
                                   contours::AbstractVector{PVContour{T}},
                                   plan::TreeEvalPlan,
                                   kernel::AbstractKernel,
                                   domain::AbstractDomain,
                                   ewald_cache,
                                   dev::AbstractDevice=CPU()) where {T}
    if _treecode_direct_ka_available(dev, kernel, domain)
        return _ka_treecode_direct_to_leaf!(vel, tree, target_leaf_idx, source_box_idx,
                                            contours, plan, kernel, domain, dev)
    end

    target_box = tree.boxes[target_leaf_idx]
    source_box = tree.boxes[source_box_idx]
    for target_seg_idx in target_box.segment_range
        flat_idx = plan.flat_indices[target_seg_idx]
        ci_t, ni_t = tree.sorted_segments[target_seg_idx]
        x = contours[ci_t].nodes[ni_t]
        vel[flat_idx] = vel[flat_idx] + _box_direct_velocity(
            tree, contours, source_box, kernel, domain, x, ewald_cache)
    end
    return nothing
end

function _treecode_linearized_to_leaf!(vel::Vector{SVector{2,T}},
                                       tree::FMMTree{T},
                                       target_leaf_idx::Int,
                                       source_box_idx::Int,
                                       contours::AbstractVector{PVContour{T}},
                                       plan::TreeEvalPlan,
                                       kernel::AbstractKernel,
                                       domain::AbstractDomain,
                                       ewald_cache,
                                       dev::AbstractDevice=CPU()) where {T}
    target_box = tree.boxes[target_leaf_idx]
    source_box = tree.boxes[source_box_idx]
    center = target_box.center

    h = max(target_box.half_width / 2, sqrt(eps(T)))
    ex = SVector{2,T}(h, zero(T))
    ey = SVector{2,T}(zero(T), h)

    if _treecode_direct_ka_available(dev, kernel, domain)
        n_source = length(source_box.segment_range)
        ax = Vector{T}(undef, n_source)
        ay = Vector{T}(undef, n_source)
        bx = Vector{T}(undef, n_source)
        by = Vector{T}(undef, n_source)
        pv = Vector{T}(undef, n_source)
        k = 1
        for seg_idx in source_box.segment_range
            ci_s, ni_s = tree.sorted_segments[seg_idx]
            c = contours[ci_s]
            a = c.nodes[ni_s]
            b = next_node(c, ni_s)
            ax[k] = a[1]
            ay[k] = a[2]
            bx[k] = b[1]
            by[k] = b[2]
            pv[k] = c.pv
            k += 1
        end

        tx = T[center[1], center[1] + ex[1], center[1] - ex[1], center[1] + ey[1], center[1] - ey[1]]
        ty = T[center[2], center[2] + ex[2], center[2] - ex[2], center[2] + ey[2], center[2] - ey[2]]
        vx, vy = _ka_box_direct_velocity_at_targets(ax, ay, bx, by, pv, tx, ty, kernel, domain, dev)
        v0  = SVector{2,T}(vx[1], vy[1])
        vxp = SVector{2,T}(vx[2], vy[2])
        vxm = SVector{2,T}(vx[3], vy[3])
        vyp = SVector{2,T}(vx[4], vy[4])
        vym = SVector{2,T}(vx[5], vy[5])
    else
        v0 = _box_direct_velocity(tree, contours, source_box, kernel, domain, center, ewald_cache)
        vxp = _box_direct_velocity(tree, contours, source_box, kernel, domain, center + ex, ewald_cache)
        vxm = _box_direct_velocity(tree, contours, source_box, kernel, domain, center - ex, ewald_cache)
        vyp = _box_direct_velocity(tree, contours, source_box, kernel, domain, center + ey, ewald_cache)
        vym = _box_direct_velocity(tree, contours, source_box, kernel, domain, center - ey, ewald_cache)
    end

    j11 = (vxp[1] - vxm[1]) / (2h)
    j21 = (vxp[2] - vxm[2]) / (2h)
    j12 = (vyp[1] - vym[1]) / (2h)
    j22 = (vyp[2] - vym[2]) / (2h)

    for target_seg_idx in target_box.segment_range
        flat_idx = plan.flat_indices[target_seg_idx]
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
    plan = _build_tree_eval_plan(tree, contours)
    kernel = prob.kernel
    domain = prob.domain
    ewald = _prefetch_ewald(domain, kernel)

    _apply_treecode_worklists!(vel, tree, contours, plan, kernel, domain, ewald, prob.dev)

    return vel
end

"""
    _treecode_velocity!(vel, prob::MultiLayerContourProblem)

Treecode O(N log N) velocity for multi-layer QG problems using modal
decomposition.  Builds one tree over all contours from all layers (the
geometry is shared across modes), then for each mode creates weighted
contours with effective PV = P_inv[mode, layer] * pv and runs the
single-layer treecode.  Results are projected back to per-layer velocity
arrays with the eigenvector weights.
"""
function _treecode_velocity!(vel::NTuple{NL, Vector{SVector{2,T}}},
                             prob::MultiLayerContourProblem{NL}) where {NL, T}
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    for i in 1:NL
        fill!(vel[i], zero(SVector{2,T}))
    end

    # Collect all contours across layers into a flat list
    all_contours = PVContour{T}[]
    contour_layer = Int[]
    for li in 1:NL
        for c in prob.layers[li]
            push!(all_contours, c)
            push!(contour_layer, li)
        end
    end
    Ntot = sum(nnodes(c) for c in all_contours; init=0)
    Ntot == 0 && return vel

    # Build tree once — geometry is identical for all modes
    tree = build_fmm_tree(all_contours)
    plan = _build_tree_eval_plan(tree, all_contours, contour_layer)

    # Per-layer flat offsets for scattering results back
    layer_flat_offset = Vector{Int}(undef, NL)
    offset = 0
    for li in 1:NL
        layer_flat_offset[li] = offset
        offset += sum(nnodes(c) for c in prob.layers[li]; init=0)
    end

    mode_vel = Vector{SVector{2,T}}(undef, Ntot)

    for mode in 1:NL
        lam = evals[mode]
        mode_kernel = abs(lam) < eps(T) * 100 ? EulerKernel() :
                      QGKernel(one(T) / sqrt(abs(lam)))
        ewald = _prefetch_ewald(domain, mode_kernel)

        # Create weighted contours: effective PV = P_inv[mode, layer] * pv
        weighted = Vector{PVContour{T}}(undef, length(all_contours))
        for ci in eachindex(all_contours)
            c = all_contours[ci]
            w = P_inv[mode, contour_layer[ci]]
            weighted[ci] = PVContour(c.nodes, w * c.pv, c.wrap)
        end

        # Run treecode on weighted contours using the shared tree
        fill!(mode_vel, zero(SVector{2,T}))
        _apply_treecode_worklists!(mode_vel, tree, weighted, plan, mode_kernel, domain, ewald, prob.dev)

        # Scatter mode velocity to per-layer arrays with projection weights
        for li in 1:NL
            pw = P[li, mode]
            abs(pw) < eps(T) && continue
            n_layer = sum(nnodes(c) for c in prob.layers[li]; init=0)
            base = layer_flat_offset[li]
            @inbounds for k in 1:n_layer
                vel[li][k] = vel[li][k] + pw * mode_vel[base + k]
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
                      plan::TreeEvalPlan) where {T}
    if _should_thread_accelerator(length(tree.leaf_indices))
        Threads.@threads for li_idx in 1:length(tree.leaf_indices)
            leaf = tree.leaf_indices[li_idx]
            box = tree.boxes[leaf]
            for seg_idx in box.segment_range
                ci_t, ni_t = tree.sorted_segments[seg_idx]
                xi = contours[ci_t].nodes[ni_t]
                flat_idx = plan.flat_indices[seg_idx]
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
    else
        for li_idx in 1:length(tree.leaf_indices)
            leaf = tree.leaf_indices[li_idx]
            box = tree.boxes[leaf]
            for seg_idx in box.segment_range
                ci_t, ni_t = tree.sorted_segments[seg_idx]
                xi = contours[ci_t].nodes[ni_t]
                flat_idx = plan.flat_indices[seg_idx]
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
end

"""
    _experimental_fmm_velocity!(vel, prob::ContourProblem)

Experimental proxy FMM driver for computing velocity at all contour nodes.
Orchestrates the complete FMM pipeline: tree construction, operator
precomputation, upward pass (S2M, M2M), interaction pass (M2L),
downward pass (L2L), local evaluation, and near-field correction.
"""
function _experimental_fmm_velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    contours = prob.contours
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    fill!(vel, zero(SVector{2,T}))
    isempty(contours) && return vel

    tree = build_fmm_tree(contours)
    kernel = prob.kernel
    domain = prob.domain
    # The current proxy-surface translation operators only handle same-level
    # M2L interactions. On an unbalanced adaptive tree, conservative fallback
    # to the production treecode is safer than silently dropping coarse-leaf
    # colleague contributions.
    _has_unhandled_coarse_leaf_interactions(tree) && return _treecode_velocity!(vel, prob)
    p = _FMM_PROXY_ORDER
    p_check = _FMM_CHECK_ORDER
    plan = _build_tree_eval_plan(tree, contours; p, p_check, include_proxy_geometry=true,
                                 kernel, domain)
    ewald = _prefetch_ewald(domain, kernel)
    nboxes = length(tree.boxes)

    # Allocate proxy data for all boxes
    proxy_data = [ProxyData(zeros(SVector{2,T}, p), SVector{2,T}[]) for _ in 1:nboxes]

    # Precompute operators
    ops = precompute_level_operators(tree, kernel, domain; p, p_check)
    m2l_op = precompute_m2l_operators(tree, kernel, domain, ops; p, p_check)

    # Full FMM pipeline
    _s2m!(proxy_data, tree, contours, plan, kernel, domain, ops, ewald; p, p_check)
    _m2m_upward!(proxy_data, tree, ops; p)
    _m2l!(proxy_data, tree, m2l_op; p)
    _l2l_downward!(proxy_data, tree, ops; p)
    _local_eval!(vel, tree, proxy_data, contours, kernel, domain, plan; p)
    _near_field!(vel, tree, contours, kernel, domain, ewald, plan)

    return vel
end

"""
    _fmm_velocity!(vel, prob::ContourProblem)

Conservative public wrapper for the experimental proxy FMM. The runtime uses
the proxy path only for unbounded single-layer problems when explicitly
enabled. Otherwise it falls back to the validated direct evaluator.

This wrapper should not be read as "full FMM support for all `ContourProblem`
variants". Periodic and multi-layer `_fmm_velocity!` methods below are still
conservative fallbacks, not production FMM implementations.
"""
function _fmm_velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    if !_FMM_ACCELERATION_ENABLED
        return _direct_velocity!(vel, prob)
    end
    return _experimental_fmm_velocity!(vel, prob)
end

# --- Periodic domain support ---

_to_unbounded_kernel(::EulerKernel) = EulerKernel()
_to_unbounded_kernel(k::QGKernel) = k
_to_unbounded_kernel(k::SQGKernel) = k

"""
    _fmm_velocity!(vel, prob::ContourProblem{K, PeriodicDomain{T}, T})

Conservative periodic fallback.

Periodic proxy FMM is not implemented yet. This method currently delegates to
the validated direct periodic evaluator, even if `_FMM_ACCELERATION_ENABLED` is
true elsewhere.
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

    if _should_thread_velocity(N)
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
    else
        @inbounds for i in 1:N
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
end

# --- Multi-layer FMM support ---

"""
    _fmm_velocity!(vel, prob::MultiLayerContourProblem)

Conservative multi-layer fallback.

Multi-layer proxy FMM is not implemented as a public production path yet. This
method currently delegates to the validated direct multi-layer evaluator, even
if `_FMM_ACCELERATION_ENABLED` is true elsewhere.
"""
function _fmm_velocity!(vel::NTuple{NL, Vector{SVector{2,T}}},
                        prob::MultiLayerContourProblem{NL}) where {NL, T}
    return _direct_velocity!(vel, prob)
end

"""
    _s2m_modal!(proxy_data, tree, all_contours, plan, P_inv, mode, kernel, domain, ops, NL)

Like `_s2m!` but weights each segment's PV by `P_inv[mode, layer_of_segment]`.
The actual least-squares solve is still delegated to `_solve_leaf_proxy_strengths!`,
so the modal path and the single-layer path share the same cached leaf operators.
"""
function _s2m_modal!(proxy_data, tree, all_contours, plan::TreeEvalPlan,
                     P_inv, mode, kernel, domain, ops, NL;
                     p=_FMM_PROXY_ORDER, p_check=_FMM_CHECK_ORDER)
    T = eltype(tree.boxes[1].center)
    leaves = tree.leaf_indices
    workspaces = _build_proxy_workspace(plan)

    # Pin BLAS to one thread to avoid nested threading with Julia's @threads
    # (the least-squares solves below dispatch to LAPACK).
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try

    if _should_thread_accelerator(length(leaves))
        Threads.@threads for li_idx in 1:length(leaves)
            leaf = leaves[li_idx]
            box = tree.boxes[leaf]
            work = workspaces[Threads.threadid()]

            check_pts = plan.leaf_check_points[leaf]
            n_check = length(check_pts)

            vel_check_x = work.vel_check_x
            vel_check_y = work.vel_check_y

            @inbounds for ic in 1:n_check
                xc = check_pts[ic]
                vx = zero(T)
                vy = zero(T)
                for seg_idx in box.segment_range
                    ci, ni = tree.sorted_segments[seg_idx]
                    layer = plan.segment_layers[seg_idx]
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

            equiv = proxy_data[leaf].equiv_strengths
            _solve_leaf_proxy_strengths!(equiv, plan, leaf, kernel, work, n_check; p)
        end
    else
        for li_idx in 1:length(leaves)
            leaf = leaves[li_idx]
            box = tree.boxes[leaf]
            work = workspaces[1]

            check_pts = plan.leaf_check_points[leaf]
            n_check = length(check_pts)

            vel_check_x = work.vel_check_x
            vel_check_y = work.vel_check_y

            @inbounds for ic in 1:n_check
                xc = check_pts[ic]
                vx = zero(T)
                vy = zero(T)
                for seg_idx in box.segment_range
                    ci, ni = tree.sorted_segments[seg_idx]
                    layer = plan.segment_layers[seg_idx]
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

            equiv = proxy_data[leaf].equiv_strengths
            _solve_leaf_proxy_strengths!(equiv, plan, leaf, kernel, work, n_check; p)
        end
    end

    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
end

"""
    _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets, prob, P, mode, kernel, domain, NL, plan)

Evaluate local expansion + near field for one mode and accumulate into per-layer
velocity arrays with modal projection weights.
"""
function _modal_accumulate!(vel, tree, proxy_data, all_contours, layer_offsets,
                            prob, P, mode, kernel, domain, NL,
                            plan::TreeEvalPlan;
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
                li = get(plan.node_to_leaf, (global_ci, ti), 0)
                if li > 0
                    box = tree.boxes[li]

                    # Local expansion evaluation
                    if length(proxy_data[li].local_strengths) > 0
                        proxy_pts = plan.leaf_proxy_points[li]
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
                            s_layer = plan.segment_layers[seg_idx]
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
