# surgery/rewrite.jl — CPU surgery stage.

"""
    reconnect!(contours, close_pairs[, domain])

Perform contour reconnection for identified close segment pairs.
Same contour pairs split. Different-contour pairs merge only after
`find_close_segments` has enforced Dritschel's compatible-interior-vorticity
condition.

Each sub-contour produced by a split contains both pinch-point nodes as
its first and last vertices, ensuring a well-formed closing segment.
Merged contours are stitched so that traversal orientation is preserved.

!!! warning
    Reconnection produces near-duplicate nodes at stitch points.
    Callers should [`remesh`](@ref) all contours after reconnection to
    clean up short/long segments.  The top-level [`surgery!`](@ref) does
    this automatically.
"""
function reconnect!(contours::Vector{PVContour{T}},
                close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                domain::AbstractDomain=UnboundedDomain()) where {T}

    isempty(close_pairs) && return
    selected_pairs = _select_reconnection_pairs(contours, close_pairs, domain)
    isempty(selected_pairs) && return

    # Process all independent pairs per iteration to reduce spatial index rebuilds.
    # "Independent" means no shared contour indices between processed pairs.
    # Splits first (they don't shift indices), then at most one merge (deleteat! shifts).
    used = Set{Int}()

    # Pass 1: batch all independent splits (ci == cj, no index shifting)
    for (ci, i, cj, j) in selected_pairs
        ci != cj && continue
        ci in used && continue
        _reconnect_split!(contours, ci, i, j, domain)
        push!(used, ci)
    end

    # Pass 2: batch independent merges, processing in decreasing cj order.
    # Since ci < cj (canonical ordering from find_close_segments), deleting cj
    # never invalidates indices of remaining pairs whose cj values are smaller.
    merge_pairs = Tuple{Int,Int,Int,Int}[]
    for (ci, i, cj, j) in selected_pairs
        ci == cj && continue
        (ci in used || cj in used) && continue
        push!(merge_pairs, (ci, i, cj, j))
        push!(used, ci)
        push!(used, cj)
    end
    sort!(merge_pairs, by=p -> p[3], rev=true)
    for (ci, i, cj, j) in merge_pairs
        _reconnect_merge!(contours, ci, i, cj, j, domain)
    end
end

# Signed area of a raw node vector (shoelace formula, no PVContour needed).
_shoelace_area(nodes::AbstractVector{SVector{2,T}}) where {T} =
    _raw_polygon_area(nodes)

function _reconnect_split!(contours::Vector{PVContour{T}}, ci::Int, i::Int, j::Int,
                        domain::AbstractDomain=UnboundedDomain()) where {T}
    c = contours[ci]
    if is_spanning(c)
        @warn "_reconnect_split!: called on spanning contour $ci — skipped" maxlog=5
        return
    end
    _, node_idx, seg_idx, stitch_point = _best_node_segment_contact(c, i, c, j, domain)
    # Add the chosen contact node before slicing the parent loop into two
    # daughters. The original endpoint and the inserted copy become the two
    # sides of the pinch point.
    nodes, inserted_idx = _insert_stitch_node(c.nodes, seg_idx, stitch_point)
    corners = _insert_corner_flag(c.corners, seg_idx, false)
    node_idx = seg_idx < node_idx ? node_idx + 1 : node_idx
    nc = length(nodes)
    lo, hi = minmax(node_idx, inserted_idx)

    # Dritschel's split introduces two labelled corner nodes at the break. Each
    # daughter keeps one label as its fixed corner; the closing segment completes
    # the local reconnection without retaining a zero-length duplicate edge.
    nodes1 = nodes[lo:(hi - 1)]
    nodes2 = vcat(nodes[hi:nc], nodes[1:(lo - 1)])
    corners1 = corners[lo:(hi - 1)]
    corners2 = vcat(corners[hi:nc], corners[1:(lo - 1)])

    if length(nodes1) >= 3 && length(nodes2) >= 3
        # Preserve the parent's orientation: if the parent was CCW (positive area),
        # each daughter should also be CCW.  Reverse nodes if the sign flips.
        orig_sign = sign(vortex_area(c))
        corner1_idx = 1
        corner2_idx = 1
        if orig_sign != 0
            if sign(_shoelace_area(nodes1)) != orig_sign
                reverse!(nodes1)
                reverse!(corners1)
                corner1_idx = length(nodes1)
            end
            if sign(_shoelace_area(nodes2)) != orig_sign
                reverse!(nodes2)
                reverse!(corners2)
                corner2_idx = length(nodes2)
            end
        end
        corners1[corner1_idx] = true
        corners2[corner2_idx] = true
        contours[ci] = PVContour(nodes1, c.pv, c.wrap, corners1)
        push!(contours, PVContour(nodes2, c.pv, c.wrap, corners2))
    else
        @warn "split aborted: daughter contours too small (n1=$(length(nodes1)), n2=$(length(nodes2))); contour $ci unchanged" maxlog=5
    end
end

# Squared local extent of a contour: the relative floor for sign decisions on
# the translation-stable shoelace area.
@inline _shoelace_noise_scale(c::PVContour) = _raw_polygon_scale2(c.nodes, c.wrap)

function _reconnect_merge!(contours::Vector{PVContour{T}}, ci::Int, i::Int, cj::Int, j::Int,
                           domain::AbstractDomain=UnboundedDomain()) where {T}
    c1 = contours[ci]
    c2 = contours[cj]

    # Check orientation consistency: both contours should have the same
    # sign of signed area. If they differ, reverse c2's node order so
    # the merged contour has consistent winding.
    # Use a robust threshold: only reverse if both signed areas are well above
    # the shoelace formula's rounding noise, which scales with each contour's
    # squared coordinate magnitude (an absolute eps floor would silently skip
    # the check for physically small contours).
    a1 = vortex_area(c1)
    a2 = vortex_area(c2)

    tol1 = eps(T) * T(1000) * _shoelace_noise_scale(c1)
    tol2 = eps(T) * T(1000) * _shoelace_noise_scale(c2)

    reversed = abs(a1) > tol1 && abs(a2) > tol2 && sign(a1) != sign(a2)
    c2_nodes = reversed ? reverse(c2.nodes) : c2.nodes
    c2_corners = reversed ? reverse(c2.corners) : copy(c2.corners)

    n2_orig = nnodes(c2)
    j_seg = reversed ? mod1(n2_orig - j, n2_orig) : j
    c2_eff = PVContour(c2_nodes, c2.pv, reversed ? -c2.wrap : c2.wrap, c2_corners)

    # For periodic domains, shift c2 into the image closest to the contact
    # point on c1 BEFORE choosing the stitch node. The stitch node is a copy of
    # an existing endpoint, so computing the shift after insertion would always
    # compare that point against its own copy and return zero, leaving the two
    # halves of the merged polygon in different periodic frames.
    shift = _periodic_merge_shift(c1.nodes[i], c2_nodes[j_seg], domain)
    if !iszero(shift)
        c2_nodes = [n + shift for n in c2_nodes]
        c2_eff = PVContour(c2_nodes, c2_eff.pv, c2_eff.wrap, c2_corners)
    end

    node_from_c1, node_idx, seg_idx, stitch_point = _best_node_segment_contact(c1, i, c2_eff, j_seg, domain)
    c1_nodes = c1.nodes
    c1_corners = copy(c1.corners)

    # Only one contour needs an inserted node: the other contour already owns
    # the endpoint that best represents the node-to-segment contact.
    if node_from_c1
        i = node_idx
        c2_nodes, j_eff = _insert_stitch_node(c2_nodes, seg_idx, stitch_point)
        c2_corners = _insert_corner_flag(c2_corners, seg_idx, false)
    else
        j_eff = node_idx
        c1_nodes, i = _insert_stitch_node(c1_nodes, seg_idx, stitch_point)
        c1_corners = _insert_corner_flag(c1_corners, seg_idx, false)
    end
    n1 = length(c1_nodes)
    n2 = length(c2_nodes)

    # Cross-join the two loops at the labelled surgery node. The two labels are
    # both fixed corners, but they are not adjacent in the node list, avoiding a
    # zero-length connector while preserving Dritschel's two-corner topology.
    c1_part_nodes = vcat(c1_nodes[i:n1], c1_nodes[1:(i - 1)])
    c1_part_corners = vcat(c1_corners[i:n1], c1_corners[1:(i - 1)])
    c2_part_nodes = vcat(c2_nodes[j_eff:n2], c2_nodes[1:(j_eff - 1)])
    c2_part_corners = vcat(c2_corners[j_eff:n2], c2_corners[1:(j_eff - 1)])

    c1_part_corners[1] = true
    c2_part_corners[1] = true

    new_nodes = vcat(c1_part_nodes, c2_part_nodes)
    new_corners = vcat(c1_part_corners, c2_part_corners)
    contours[ci] = PVContour(new_nodes, c1.pv, c1.wrap, new_corners)

    deleteat!(contours, cj)
end

@inline _periodic_merge_shift(::SVector{2,T}, ::SVector{2,T}, ::UnboundedDomain) where {T} = zero(SVector{2,T})
@inline function _periodic_merge_shift(ref::SVector{2,T}, pt::SVector{2,T},
                                        domain::PeriodicDomain{T}) where {T}
    raw = ref - pt
    return raw - _min_image(raw, domain)
end
