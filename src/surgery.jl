# ── Spatial Index ────────────────────────────────────────

struct SpatialIndex{T<:AbstractFloat}
    bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}  # (bin_x, bin_y) => [(contour_idx, node_idx)]
    bin_size::T
end

"""
Build a spatial index for all contour segments, binned by grid of size `delta`.
Each segment is binned at its endpoint *and* at its midpoint so that long segments
whose interiors cross a bin boundary are discoverable via neighbour-bin queries.
"""
function build_spatial_index(contours::Vector{PVContour{T}}, delta::T) where {T}
    bins = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (ci, c) in enumerate(contours)
        nc = nnodes(c)
        for ni in 1:nc
            a = c.nodes[ni]
            b = next_node(c, ni)
            mid = (a + b) / 2
            # Bin the node and the segment midpoint
            for pt in (a, mid)
                bx = floor(Int, pt[1] / delta)
                by = floor(Int, pt[2] / delta)
                key = (bx, by)
                if !haskey(bins, key)
                    bins[key] = Tuple{Int,Int}[]
                end
                push!(bins[key], (ci, ni))
            end
        end
    end

    return SpatialIndex(bins, delta)
end

# ── Contour Reconnection ────────────────────────────────

"""
    _segment_min_dist2(a1, b1, a2, b2)

Minimum squared distance between segments `a1→b1` and `a2→b2`.
Projects each segment's closest point onto the other and returns the smallest
squared distance found.
"""
function _segment_min_dist2(a1::SVector{2,T}, b1::SVector{2,T},
                            a2::SVector{2,T}, b2::SVector{2,T}) where {T}
    d1 = b1 - a1
    d2 = b2 - a2
    best = typemax(T)
    # Test four endpoint-to-segment projections + segment-segment
    for (p, a, d) in ((a1, a2, d2), (b1, a2, d2), (a2, a1, d1), (b2, a1, d1))
        len2 = d[1]^2 + d[2]^2
        if len2 < eps(T)
            r = p - a
        else
            t = clamp(((p[1]-a[1])*d[1] + (p[2]-a[2])*d[2]) / len2, zero(T), one(T))
            r = p - (a + t * d)
        end
        dist2 = r[1]^2 + r[2]^2
        dist2 < best && (best = dist2)
    end
    return best
end

"""
    find_close_segments(contours, spatial_index, delta)

Find pairs of contour segments whose closest approach is within `delta`,
using the spatial index for candidate filtering.
Returns vector of `(ci, i, cj, j)` tuples where `i`,`j` are segment indices
(each segment goes from node `i` to `next_node(c, i)`).
"""
function find_close_segments(contours::Vector{PVContour{T}}, idx::SpatialIndex{T}, delta::T) where {T}
    close_pairs = Tuple{Int,Int,Int,Int}[]
    delta2 = delta^2
    seen = Set{Tuple{Int,Int,Int,Int}}()

    for (ci, c) in enumerate(contours)
        is_spanning(c) && continue
        nc = nnodes(c)
        for i in 1:nc
            a_i = c.nodes[i]
            b_i = next_node(c, i)
            mid_i = (a_i + b_i) / 2
            bx = floor(Int, mid_i[1] / delta)
            by = floor(Int, mid_i[2] / delta)

            for dbx in -1:1, dby in -1:1
                key = (bx + dbx, by + dby)
                haskey(idx.bins, key) || continue
                for (cj, j) in idx.bins[key]
                    # Canonical ordering to avoid duplicates
                    pair = (ci, i) < (cj, j) ? (ci, i, cj, j) : (cj, j, ci, i)
                    pair in seen && continue
                    is_spanning(contours[cj]) && continue
                    if ci == cj
                        ncj = nnodes(contours[cj])
                        dist_along = min(abs(i - j), ncj - abs(i - j))
                        dist_along <= 2 && continue
                    else
                        contours[ci].pv != contours[cj].pv && continue
                    end

                    a_j = contours[cj].nodes[j]
                    b_j = next_node(contours[cj], j)
                    if _segment_min_dist2(a_i, b_i, a_j, b_j) < delta2
                        push!(seen, pair)
                        push!(close_pairs, pair)
                    end
                end
            end
        end
    end

    return close_pairs
end

"""
    reconnect!(contours, close_pairs)

Perform contour reconnection for identified close segment pairs.
Same contour → split; different contours with same PV → merge.

Each sub-contour produced by a split contains both pinch-point nodes as
its first and last vertices, ensuring a well-formed closing segment.
Merged contours are stitched so that traversal orientation is preserved.
"""
function reconnect!(contours::Vector{PVContour{T}}, close_pairs::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    isempty(close_pairs) && return

    # Process one reconnection at a time (simplest correct approach)
    # After each reconnection, rebuild spatial index (handled by caller)
    pair = close_pairs[1]
    ci, i, cj, j = pair

    if ci == cj
        # Split: pinch contour at nodes i and j to create two daughter contours.
        # Ensure i < j for consistent ordering.
        c = contours[ci]
        nc = nnodes(c)
        lo, hi = minmax(i, j)

        # Daughter 1: nodes lo → hi (both endpoints included)
        nodes1 = c.nodes[lo:hi]
        # Daughter 2: nodes hi → lo wrapping around (both endpoints included)
        nodes2 = vcat(c.nodes[hi:nc], c.nodes[1:lo])

        if length(nodes1) >= 3 && length(nodes2) >= 3
            contours[ci] = PVContour(nodes1, c.pv)
            push!(contours, PVContour(nodes2, c.pv))
        end
    else
        # Merge: stitch two same-PV contours at their closest segments.
        # Preserves traversal orientation of both contours.
        c1 = contours[ci]
        c2 = contours[cj]
        n1 = nnodes(c1)
        n2 = nnodes(c2)

        # Check orientation consistency: both contours should have the same
        # sign of signed area. If they differ, reverse c2's node order so
        # the merged contour has consistent winding.
        a1 = vortex_area(c1)
        a2 = vortex_area(c2)
        c2_nodes = (sign(a1) == sign(a2) || abs(a2) < eps(T)) ? c2.nodes : reverse(c2.nodes)
        # Recompute j for reversed contour
        j_eff = (c2_nodes === c2.nodes) ? j : (n2 - j + 1)

        new_nodes = vcat(
            c1.nodes[1:i],
            c2_nodes[j_eff:n2],
            c2_nodes[1:j_eff],
            c1.nodes[(i+1):n1]
        )
        contours[ci] = PVContour(new_nodes, c1.pv)
        deleteat!(contours, cj)
    end
end

# ── Filament Removal ─────────────────────────────────────

"""Remove contours with |area| < area_min. Spanning contours are always kept."""
function remove_filaments!(contours::Vector{PVContour{T}}, area_min::T) where {T}
    filter!(c -> is_spanning(c) || abs(vortex_area(c)) >= area_min, contours)
end

# ── Top-Level Surgery ────────────────────────────────────

"""
    surgery!(prob::ContourProblem, params::SurgeryParams)

Full Dritschel surgery suite: remesh → reconnect → remove filaments.
Mutates `prob.contours` in place.
"""
function surgery!(prob::ContourProblem, params::SurgeryParams)
    contours = prob.contours

    # 1. Remesh all contours
    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params)
    end

    # 2. Reconnection — iterate until no more close pairs remain
    for _ in 1:100  # safety limit
        idx = build_spatial_index(contours, params.delta)
        close_pairs = find_close_segments(contours, idx, params.delta)
        isempty(close_pairs) && break
        reconnect!(contours, close_pairs)
    end

    # 3. Remove filaments
    remove_filaments!(contours, params.area_min)

    return prob
end
