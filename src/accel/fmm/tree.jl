# Adaptive quadtree construction for the kernel-independent FMM.

# ── Constants ───────────────────────────────────────────────

"""Threshold on total segment count above which a large-problem accelerator may be used."""
const _FMM_THRESHOLD = 1024

"""Whether the proxy-surface FMM acceleration path is enabled for runtime use."""
const _FMM_ACCELERATION_ENABLED = false

"""Maximum number of segments in a leaf box before subdivision."""
const _FMM_MAX_PER_LEAF = 40

"""Maximum depth of the quadtree."""
const _FMM_MAX_DEPTH = 20

# ── Types ───────────────────────────────────────────────────

struct FMMBox{T<:AbstractFloat}
    center::SVector{2,T}
    half_width::T
    level::Int
    parent::Int                   # index into boxes array; 0 = root
    children::SVector{4,Int}      # 0 = no child; order: SW, SE, NW, NE
    segment_range::UnitRange{Int} # range into sorted_segments
    is_leaf::Bool
end

struct FMMTree{T<:AbstractFloat}
    boxes::Vector{FMMBox{T}}
    sorted_segments::Vector{Tuple{Int,Int}}    # (contour_idx, node_idx)
    segment_midpoints::Vector{SVector{2,T}}    # midpoints parallel to sorted_segments
    leaf_indices::Vector{Int}                   # indices of leaf boxes
    interaction_lists::Vector{Vector{Int}}      # per-box M2L targets
    near_lists::Vector{Set{Int}}               # per-box direct neighbors (Set for O(1) lookup)
    level_boxes::Vector{Vector{Int}}           # per-level box indices for O(N) iteration
    max_level::Int
end

"""
    TreeEvalPlan{T}

Precomputed geometry and worklists shared by the treecode and proxy-FMM paths.

The plan separates the adaptive tree construction phase from the hot evaluation
loops so the runtime code can iterate over flat arrays and preclassified leaf
worklists instead of rebuilding topology-dependent metadata on every call.

- `flat_indices`: map each entry in `tree.sorted_segments` to its flat node index.
- `direct_lists`: per-target-leaf source boxes handled by direct summation.
- `approx_lists`: per-target-leaf source boxes handled by the treecode far field.
- `node_to_leaf`: lookup from `(contour_idx, node_idx)` to the owning leaf box.
- `segment_layers`: layer id for each sorted segment in multi-layer modal paths.
- `leaf_proxy_points`: per-leaf proxy-surface points for the proxy FMM.
- `leaf_check_points`: per-leaf dual check surfaces used by the proxy solve.
- `leaf_check_to_proxy`: dense `K(check, proxy)` matrices for each leaf.
- `leaf_augmented_check_to_proxy`: Euler-specific matrices with the zero-sum row.
- `leaf_check_to_proxy_qr`: cached least-squares operators for non-Euler kernels.
- `leaf_augmented_check_to_proxy_qr`: cached least-squares operators for Euler.
"""
struct TreeEvalPlan{T<:AbstractFloat}
    flat_indices::Vector{Int}
    direct_lists::Vector{Vector{Int}}
    approx_lists::Vector{Vector{Int}}
    node_to_leaf::Dict{Tuple{Int,Int}, Int}
    segment_layers::Vector{Int}
    leaf_proxy_points::Vector{Vector{SVector{2,T}}}
    leaf_check_points::Vector{Vector{SVector{2,T}}}
    leaf_check_to_proxy::Vector{Matrix{T}}
    leaf_augmented_check_to_proxy::Vector{Matrix{T}}
    leaf_check_to_proxy_qr::Vector{LinearAlgebra.QRCompactWY{T, Matrix{T}, Matrix{T}}}
    leaf_augmented_check_to_proxy_qr::Vector{LinearAlgebra.QRCompactWY{T, Matrix{T}, Matrix{T}}}
end

# ── Helper functions ────────────────────────────────────────

"""
    _compute_bounding_box(contours) -> (center, half_width)

Compute a square bounding box that encloses all segment midpoints with 1% padding.
"""
function _compute_bounding_box(contours::AbstractVector{PVContour{T}}) where {T}
    xmin = typemax(T)
    xmax = typemin(T)
    ymin = typemax(T)
    ymax = typemin(T)
    for c in contours
        n = nnodes(c)
        for j in 1:n
            nj = next_node(c, j)
            mx = (c.nodes[j][1] + nj[1]) / 2
            my = (c.nodes[j][2] + nj[2]) / 2
            xmin = min(xmin, mx)
            xmax = max(xmax, mx)
            ymin = min(ymin, my)
            ymax = max(ymax, my)
        end
    end
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    hw = max(xmax - xmin, ymax - ymin) / 2
    hw *= T(1.01)  # 1% padding
    # Ensure nonzero half_width
    if hw < eps(T)
        hw = one(T)
    end
    return SVector{2,T}(cx, cy), hw
end

"""
    _collect_segment_midpoints(contours) -> (midpoints, seg_ids)

Collect all segment midpoints and corresponding (contour_idx, node_idx) identifiers.
"""
function _collect_segment_midpoints(contours::AbstractVector{PVContour{T}}) where {T}
    total = sum(nnodes(c) for c in contours; init=0)
    midpoints = Vector{SVector{2,T}}(undef, total)
    seg_ids = Vector{Tuple{Int,Int}}(undef, total)
    k = 0
    for (ci, c) in enumerate(contours)
        n = nnodes(c)
        for j in 1:n
            k += 1
            nj = next_node(c, j)
            midpoints[k] = (c.nodes[j] + nj) / 2
            seg_ids[k] = (ci, j)
        end
    end
    return midpoints, seg_ids
end

"""
    _quadrant(pt, center) -> Int

Return the quadrant index: 1=SW, 2=SE, 3=NW, 4=NE.
"""
@inline function _quadrant(pt::SVector{2,T}, center::SVector{2,T}) where {T}
    east = pt[1] >= center[1]
    north = pt[2] >= center[2]
    if north
        return east ? 4 : 3
    else
        return east ? 2 : 1
    end
end

"""
    _child_center(parent_center, parent_hw, quad) -> SVector{2,T}

Compute the center of a child box for the given quadrant.
"""
@inline function _child_center(parent_center::SVector{2,T}, parent_hw::T, quad::Int) where {T}
    offset = parent_hw / 2
    if quad == 1      # SW
        return SVector{2,T}(parent_center[1] - offset, parent_center[2] - offset)
    elseif quad == 2  # SE
        return SVector{2,T}(parent_center[1] + offset, parent_center[2] - offset)
    elseif quad == 3  # NW
        return SVector{2,T}(parent_center[1] - offset, parent_center[2] + offset)
    else              # NE
        return SVector{2,T}(parent_center[1] + offset, parent_center[2] + offset)
    end
end

"""
    _partition_segments!(midpoints, seg_ids, lo, hi, center) -> NTuple{4,UnitRange{Int}}

In-place 4-way partition of segments in `lo:hi` into quadrants around `center`.
Returns four UnitRanges (SW, SE, NW, NE) into the arrays.
"""
function _partition_segments!(
    midpoints::Vector{SVector{2,T}},
    seg_ids::Vector{Tuple{Int,Int}},
    lo::Int, hi::Int,
    center::SVector{2,T}
) where {T}
    n = hi - lo + 1
    if n == 0
        empty = lo:(lo-1)
        return (empty, empty, empty, empty)
    end

    # Count elements per quadrant
    counts = MVector{4,Int}(0, 0, 0, 0)
    for i in lo:hi
        q = _quadrant(midpoints[i], center)
        counts[q] += 1
    end

    # Compute start positions for each quadrant
    starts = MVector{4,Int}(0, 0, 0, 0)
    starts[1] = lo
    starts[2] = starts[1] + counts[1]
    starts[3] = starts[2] + counts[2]
    starts[4] = starts[3] + counts[3]

    # Build ranges
    ranges = (
        starts[1]:(starts[2] - 1),
        starts[2]:(starts[3] - 1),
        starts[3]:(starts[4] - 1),
        starts[4]:(starts[4] + counts[4] - 1),
    )

    # Copy to temporary arrays, then write back in sorted order.
    # TODO: pre-allocate a single buffer of length total_segs and reuse across subdivisions
    # to reduce O(N log N) total allocation from per-subdivision copies.
    tmp_mid = midpoints[lo:hi]
    tmp_ids = seg_ids[lo:hi]

    pos = MVector{4,Int}(starts[1], starts[2], starts[3], starts[4])
    for i in 1:n
        q = _quadrant(tmp_mid[i], center)
        idx = pos[q]
        midpoints[idx] = tmp_mid[i]
        seg_ids[idx] = tmp_ids[i]
        pos[q] += 1
    end

    return ranges
end

"""
    _empty_tree(T) -> FMMTree{T}

Return an empty FMMTree for zero segments.
"""
function _empty_tree(::Type{T}) where {T<:AbstractFloat}
    FMMTree{T}(
        FMMBox{T}[],
        Tuple{Int,Int}[],
        SVector{2,T}[],
        Int[],
        Vector{Int}[],
        Set{Int}[],
        Vector{Int}[],
        0,
    )
end

"""
    _are_adjacent_or_self(a, b) -> Bool

Two boxes are adjacent (or identical) if their centers are within
`(a.half_width + b.half_width) * 1.01` in each dimension.
"""
@inline function _are_adjacent_or_self(a::FMMBox{T}, b::FMMBox{T}) where {T}
    threshold = (a.half_width + b.half_width) * T(1.01)
    return abs(a.center[1] - b.center[1]) <= threshold &&
           abs(a.center[2] - b.center[2]) <= threshold
end

# ── Tree building ───────────────────────────────────────────

"""
    build_fmm_tree(contours; max_per_leaf=_FMM_MAX_PER_LEAF, max_depth=_FMM_MAX_DEPTH) -> FMMTree

Build an adaptive quadtree over the segment midpoints of the given contours.
Uses an iterative stack to avoid deep recursion.
"""
function build_fmm_tree(
    contours::AbstractVector{PVContour{T}};
    max_per_leaf::Int = _FMM_MAX_PER_LEAF,
    max_depth::Int = _FMM_MAX_DEPTH,
) where {T}
    # Collect all segment midpoints
    total_segs = sum(nnodes(c) for c in contours; init=0)
    total_segs == 0 && return _empty_tree(T)

    midpoints, seg_ids = _collect_segment_midpoints(contours)
    center, hw = _compute_bounding_box(contours)

    # Preallocate boxes
    boxes = FMMBox{T}[]
    sizehint!(boxes, max(64, total_segs ÷ max_per_leaf * 2))
    leaf_indices = Int[]

    # Stack entries: (parent_idx, lo, hi, center, half_width, level)
    stack = Tuple{Int, Int, Int, SVector{2,T}, T, Int}[]

    # Create root box (placeholder; will be updated)
    root = FMMBox{T}(center, hw, 0, 0, SVector{4,Int}(0,0,0,0), 1:total_segs, true)
    push!(boxes, root)
    push!(stack, (1, 1, total_segs, center, hw, 0))

    while !isempty(stack)
        box_idx, lo, hi, box_center, box_hw, level = pop!(stack)
        n_seg = hi - lo + 1

        if n_seg <= max_per_leaf || level >= max_depth
            # This is a leaf
            boxes[box_idx] = FMMBox{T}(
                box_center, box_hw, level,
                boxes[box_idx].parent,
                SVector{4,Int}(0,0,0,0),
                lo:hi,
                true,
            )
            push!(leaf_indices, box_idx)
        else
            # Subdivide
            ranges = _partition_segments!(midpoints, seg_ids, lo, hi, box_center)
            child_hw = box_hw / 2
            children = MVector{4,Int}(0, 0, 0, 0)

            for q in 1:4
                r = ranges[q]
                if !isempty(r)
                    cc = _child_center(box_center, box_hw, q)
                    child_box = FMMBox{T}(
                        cc, child_hw, level + 1,
                        box_idx,
                        SVector{4,Int}(0,0,0,0),
                        r,
                        true,  # placeholder, may be updated
                    )
                    push!(boxes, child_box)
                    child_idx = length(boxes)
                    children[q] = child_idx
                    push!(stack, (child_idx, first(r), last(r), cc, child_hw, level + 1))
                end
            end

            # Update parent to be internal node
            boxes[box_idx] = FMMBox{T}(
                box_center, box_hw, level,
                boxes[box_idx].parent,
                SVector{4,Int}(children),
                lo:hi,
                false,
            )
        end
    end

    max_level = maximum(b.level for b in boxes)

    # Build interaction and near lists
    interaction_lists, near_lists = _build_lists(boxes, max_level)

    # Build per-level box index for efficient M2M/L2L iteration
    level_boxes = [Int[] for _ in 0:max_level]
    for i in eachindex(boxes)
        push!(level_boxes[boxes[i].level + 1], i)
    end

    return FMMTree{T}(
        boxes,
        seg_ids,
        midpoints,
        leaf_indices,
        interaction_lists,
        near_lists,
        level_boxes,
        max_level,
    )
end

# ── List building ───────────────────────────────────────────

"""
    _build_lists(boxes, max_level) -> (interaction_lists, near_lists)

Build interaction lists (M2L) and near lists for all boxes.

Interaction list rule: For box B with parent P (which itself has a parent),
look at all children of P's same-level neighbors. Those children that are
NOT adjacent to B form B's interaction list.

Near list rule (leaves only): self + all same-level leaf boxes that are adjacent.
"""
function _build_lists(boxes::Vector{FMMBox{T}}, max_level::Int) where {T}
    nboxes = length(boxes)
    interaction_lists = [Int[] for _ in 1:nboxes]
    near_lists = [Set{Int}() for _ in 1:nboxes]

    # Build level-wise index for efficient neighbor lookups
    level_boxes = [Int[] for _ in 0:max_level]
    for i in 1:nboxes
        push!(level_boxes[boxes[i].level + 1], i)  # +1 because Julia is 1-indexed
    end

    # Near lists: for each leaf box, find all leaf boxes (any level) that are adjacent.
    # In an adaptive tree, adjacent leaves can be at different levels.
    # Use spatial hashing to avoid O(L²) all-pairs comparison.
    all_leaves = Int[]
    for i in 1:nboxes
        boxes[i].is_leaf && push!(all_leaves, i)
    end

    if !isempty(all_leaves)
        # Grid cell size = 2 * max leaf half_width, so each leaf fits in one cell
        max_hw = maximum(boxes[i].half_width for i in all_leaves)
        cell_size = 2 * max_hw * T(1.02)  # slight padding for adjacency check

        # Hash leaves into grid cells
        leaf_grid = Dict{Tuple{Int,Int}, Vector{Int}}()
        for i in all_leaves
            cx = floor(Int, boxes[i].center[1] / cell_size)
            cy = floor(Int, boxes[i].center[2] / cell_size)
            key = (cx, cy)
            if haskey(leaf_grid, key)
                push!(leaf_grid[key], i)
            else
                leaf_grid[key] = [i]
            end
        end

        # For each leaf, only check leaves in the 3x3 neighborhood of grid cells
        for i in all_leaves
            cx = floor(Int, boxes[i].center[1] / cell_size)
            cy = floor(Int, boxes[i].center[2] / cell_size)
            for dx in -1:1, dy in -1:1
                nbr_key = (cx + dx, cy + dy)
                haskey(leaf_grid, nbr_key) || continue
                for j in leaf_grid[nbr_key]
                    if _are_adjacent_or_self(boxes[i], boxes[j])
                        push!(near_lists[i], j)
                    end
                end
            end
        end
    end

    # Interaction lists: for box B with parent P (P must also have a parent),
    # find P's neighbors at P's level, then check their children.
    # Children of P's neighbors that are NOT adjacent to B go into B's interaction list.
    for b_idx in 1:nboxes
        p_idx = boxes[b_idx].parent
        p_idx == 0 && continue  # root has no parent

        p_level = boxes[p_idx].level
        # Find P's same-level neighbors (including P itself)
        p_neighbors = Int[]
        for candidate in level_boxes[p_level + 1]
            if _are_adjacent_or_self(boxes[p_idx], boxes[candidate])
                push!(p_neighbors, candidate)
            end
        end

        # For each of P's neighbors, check their children
        for pn_idx in p_neighbors
            for q in 1:4
                child_idx = boxes[pn_idx].children[q]
                child_idx == 0 && continue
                if !_are_adjacent_or_self(boxes[b_idx], boxes[child_idx])
                    push!(interaction_lists[b_idx], child_idx)
                end
            end
        end
    end

    return interaction_lists, near_lists
end

"""
    _has_unhandled_coarse_leaf_interactions(tree) -> Bool

Return `true` when a leaf box has a well-separated colleague source that is a
coarser leaf rather than a same-level child box.

The current proxy-surface FMM translation operators only support same-level M2L
translations. On an unbalanced adaptive tree, a parent-level neighbor can be a
leaf itself; in that case there is no child box to put in the target leaf's
interaction list, and the contribution would be dropped unless the caller
falls back to a more conservative accelerator.
"""
function _has_unhandled_coarse_leaf_interactions(tree::FMMTree{T}) where {T}
    boxes = tree.boxes
    level_boxes = tree.level_boxes

    for leaf_idx in tree.leaf_indices
        leaf = boxes[leaf_idx]
        p_idx = leaf.parent
        p_idx == 0 && continue

        parent = boxes[p_idx]
        for candidate_idx in level_boxes[parent.level + 1]
            _are_adjacent_or_self(parent, boxes[candidate_idx]) || continue
            candidate_idx == p_idx && continue

            candidate = boxes[candidate_idx]
            if candidate.is_leaf && !_are_adjacent_or_self(leaf, candidate)
                return true
            end
        end
    end

    return false
end
