# FMM translation operators: M2M, M2L, L2L, local evaluation.

# ── Types ───────────────────────────────────────────────────

"""
    M2LOperators{T}

Precomputed M2L translation matrices for one level, keyed by displacement
vector `(dx, dy)` in units of box width.
"""
struct M2LOperators{T<:AbstractFloat}
    operators::Dict{Tuple{Int,Int}, Matrix{T}}
end

# ── M2L precomputation ─────────────────────────────────────

"""
    precompute_m2l_operators(tree, kernel, domain, ops; p, p_check)

For each level of the quadtree, precompute M2L matrices for all valid
interaction-list displacement vectors.

A displacement `(dx, dy)` (in units of `2*hw`, the box width at that level)
is a valid interaction-list displacement when `max(|dx|, |dy|) ∈ {2,3}` but
excluding near neighbors (`|dx| <= 1` AND `|dy| <= 1`).

For each displacement the combined operator is:
    `pinv * K(target_check, source_proxy)`
mapping source equivalent strengths to target local strengths.

Returns a `Vector{M2LOperators{T}}` of length `tree.max_level + 1`.
"""
function precompute_m2l_operators(
    tree::FMMTree{T},
    kernel::AbstractKernel,
    domain::AbstractDomain = UnboundedDomain(),
    ops::Vector{LevelOperators{T}} = precompute_level_operators(tree, kernel, domain);
    p::Int = _FMM_PROXY_ORDER,
    p_check::Int = _FMM_CHECK_ORDER,
) where {T}
    max_level = tree.max_level
    root_hw = tree.boxes[1].half_width
    m2l_ops = Vector{M2LOperators{T}}(undef, max_level + 1)

    origin = SVector{2,T}(zero(T), zero(T))

    for level in 0:max_level
        hw = root_hw / T(2)^level
        box_width = 2 * hw  # full width of a box at this level

        # Target is a canonical box at the origin; use its check surface
        target_check_pts = _check_points(origin, hw, p_check)

        # Pseudoinverse for the target box
        pinv_mat = ops[level + 1].check_to_proxy_pinv

        # Enumerate all valid interaction-list displacements
        op_dict = Dict{Tuple{Int,Int}, Matrix{T}}()

        for dx in -3:3
            for dy in -3:3
                # Skip near neighbors (including self): |dx|<=1 AND |dy|<=1
                if abs(dx) <= 1 && abs(dy) <= 1
                    continue
                end
                # Only keep displacements within the interaction-list range
                # max(|dx|, |dy|) must be 2 or 3
                if max(abs(dx), abs(dy)) > 3
                    continue
                end

                # Source box center is displaced by (dx, dy) box widths
                source_center = SVector{2,T}(T(dx) * box_width, T(dy) * box_width)
                source_proxy_pts = _proxy_points(source_center, hw, p)

                # Kernel matrix: target check points evaluated at source proxy sources
                K_tc_sp = _build_kernel_matrix(kernel, domain, target_check_pts, source_proxy_pts)

                # Combined M2L operator: pinv * K
                op_dict[(dx, dy)] = Matrix{T}(pinv_mat * K_tc_sp)
            end
        end

        m2l_ops[level + 1] = M2LOperators{T}(op_dict)
    end

    return m2l_ops
end

# ── M2M upward pass ────────────────────────────────────────

"""
    _m2m_upward!(proxy_data, tree, ops; p)

Bottom-up M2M merge: propagate equivalent strengths from children to parents.

For each non-leaf box from the deepest level upward, accumulate child
equivalent strengths into the parent using the precomputed `child_to_parent`
operators (applied component-wise to x and y).
"""
function _m2m_upward!(
    proxy_data::Vector{ProxyData{T}},
    tree::FMMTree{T},
    ops::Vector{LevelOperators{T}};
    p::Int = _FMM_PROXY_ORDER,
) where {T}
    boxes = tree.boxes

    # Process levels bottom-up: from max_level-1 down to 0
    # (leaves at max_level have their strengths set by S2M;
    #  internal nodes merge their children's strengths)
    for level in (tree.max_level - 1):-1:0
        for bi in 1:length(boxes)
            box = boxes[bi]
            box.level == level || continue
            box.is_leaf && continue

            # Initialize parent equiv_strengths if empty
            parent_eq = proxy_data[bi].equiv_strengths
            if length(parent_eq) == 0
                resize!(parent_eq, p)
                fill!(parent_eq, zero(SVector{2,T}))
            end

            for q in 1:4
                child_idx = box.children[q]
                child_idx == 0 && continue

                child_eq = proxy_data[child_idx].equiv_strengths
                # Skip children with no strengths
                all(s -> s == zero(SVector{2,T}), child_eq) && continue

                # M2M operator for this child quadrant
                # The child is at level+1, so we use ops[level+2] which stores
                # child_to_parent for children at child's level
                m2m_mat = ops[level + 2].child_to_parent[q]

                # Extract x and y components
                child_x = [s[1] for s in child_eq]
                child_y = [s[2] for s in child_eq]

                parent_x = m2m_mat * child_x
                parent_y = m2m_mat * child_y

                # Accumulate into parent
                @inbounds for k in 1:p
                    parent_eq[k] = parent_eq[k] + SVector{2,T}(parent_x[k], parent_y[k])
                end
            end
        end
    end

    return nothing
end

# ── M2L interaction ────────────────────────────────────────

"""
    _m2l!(proxy_data, tree, m2l_ops; p)

For each box with a non-empty interaction list, apply the precomputed M2L
operator to translate source equivalent strengths into target local strengths.
"""
function _m2l!(
    proxy_data::Vector{ProxyData{T}},
    tree::FMMTree{T},
    m2l_ops::Vector{M2LOperators{T}};
    p::Int = _FMM_PROXY_ORDER,
) where {T}
    boxes = tree.boxes

    for bi in 1:length(boxes)
        ilist = tree.interaction_lists[bi]
        isempty(ilist) && continue

        box = boxes[bi]
        level = box.level
        hw = box.half_width
        box_width = 2 * hw

        # Initialize local_strengths if empty
        local_s = proxy_data[bi].local_strengths
        if length(local_s) == 0
            resize!(local_s, p)
            fill!(local_s, zero(SVector{2,T}))
        end

        level_m2l = m2l_ops[level + 1]

        for src_idx in ilist
            source_box = boxes[src_idx]
            source_eq = proxy_data[src_idx].equiv_strengths

            # Skip sources with no strengths
            all(s -> s == zero(SVector{2,T}), source_eq) && continue

            # Compute displacement in box-width units
            dx = round(Int, (source_box.center[1] - box.center[1]) / box_width)
            dy = round(Int, (source_box.center[2] - box.center[2]) / box_width)

            # Look up precomputed M2L matrix
            m2l_mat = level_m2l.operators[(dx, dy)]

            # Extract components
            src_x = [s[1] for s in source_eq]
            src_y = [s[2] for s in source_eq]

            local_x = m2l_mat * src_x
            local_y = m2l_mat * src_y

            # Accumulate into target local strengths
            @inbounds for k in 1:p
                local_s[k] = local_s[k] + SVector{2,T}(local_x[k], local_y[k])
            end
        end
    end

    return nothing
end

# ── L2L downward pass ──────────────────────────────────────

"""
    _l2l_downward!(proxy_data, tree, ops; p)

Top-down L2L push: propagate local strengths from parents to children.

Uses the transpose of the M2M `child_to_parent` operator as the L2L operator:
    `L2L_q = child_to_parent[q]'`
"""
function _l2l_downward!(
    proxy_data::Vector{ProxyData{T}},
    tree::FMMTree{T},
    ops::Vector{LevelOperators{T}};
    p::Int = _FMM_PROXY_ORDER,
) where {T}
    boxes = tree.boxes

    # Process levels top-down: from 0 to max_level-1
    for level in 0:(tree.max_level - 1)
        for bi in 1:length(boxes)
            box = boxes[bi]
            box.level == level || continue
            box.is_leaf && continue

            parent_local = proxy_data[bi].local_strengths
            # Skip parents with no local strengths
            (length(parent_local) == 0 || all(s -> s == zero(SVector{2,T}), parent_local)) && continue

            # Extract parent local components
            parent_x = [s[1] for s in parent_local]
            parent_y = [s[2] for s in parent_local]

            for q in 1:4
                child_idx = box.children[q]
                child_idx == 0 && continue

                # Initialize child local_strengths if empty
                child_local = proxy_data[child_idx].local_strengths
                if length(child_local) == 0
                    resize!(child_local, p)
                    fill!(child_local, zero(SVector{2,T}))
                end

                # L2L operator is transpose of M2M child_to_parent
                l2l_mat = ops[level + 2].child_to_parent[q]'

                child_x = l2l_mat * parent_x
                child_y = l2l_mat * parent_y

                # Accumulate into child local strengths
                @inbounds for k in 1:p
                    child_local[k] = child_local[k] + SVector{2,T}(child_x[k], child_y[k])
                end
            end
        end
    end

    return nothing
end

# ── Local evaluation ───────────────────────────────────────

"""
    _node_flat_index(contours, ci, ni)

Map `(contour_idx, node_idx)` to a flat velocity-array index.
The flat index is the sum of `nnodes` for contours `1..ci-1`, plus `ni`.
"""
function _node_flat_index(contours::AbstractVector{PVContour{T}}, ci::Int, ni::Int) where {T}
    idx = 0
    for i in 1:(ci - 1)
        idx += nnodes(contours[i])
    end
    return idx + ni
end

"""
    _local_eval!(vel, tree, proxy_data, contours, kernel, domain; p)

For each leaf box with nonzero local strengths, evaluate the local expansion
at every target node (segment midpoint's corresponding node) in the box.

The local expansion is represented by proxy charges on the proxy surface;
the velocity contribution is obtained by evaluating the kernel between each
target node and each proxy point, weighted by local strengths.
"""
function _local_eval!(
    vel::Vector{SVector{2,T}},
    tree::FMMTree{T},
    proxy_data::Vector{ProxyData{T}},
    contours::AbstractVector{PVContour{T}},
    kernel::AbstractKernel,
    domain::AbstractDomain;
    p::Int = _FMM_PROXY_ORDER,
) where {T}
    boxes = tree.boxes

    for leaf_idx in tree.leaf_indices
        box = boxes[leaf_idx]
        local_s = proxy_data[leaf_idx].local_strengths

        # Skip leaves with no local strengths
        (length(local_s) == 0 || all(s -> s == zero(SVector{2,T}), local_s)) && continue

        # Compute proxy points for this leaf
        proxy_pts = _proxy_points(box.center, box.half_width, p)

        # For each segment (target node) in this leaf
        for si in box.segment_range
            ci, ni = tree.sorted_segments[si]
            c = contours[ci]
            # Target position is the node itself (not the midpoint)
            x = c.nodes[ni]

            # Evaluate kernel at target from each proxy point
            vx = zero(T)
            vy = zero(T)
            @inbounds for k in 1:p
                G = _kernel_value(kernel, domain, x, proxy_pts[k])
                vx += G * local_s[k][1]
                vy += G * local_s[k][2]
            end

            # Add to velocity at the correct flat index
            flat_idx = _node_flat_index(contours, ci, ni)
            vel[flat_idx] = vel[flat_idx] + SVector{2,T}(vx, vy)
        end
    end

    return nothing
end
