# ── Spatial Index ────────────────────────────────────────

struct SpatialIndex{T<:AbstractFloat}
    bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}  # (bin_x, bin_y) => [(contour_idx, node_idx)]
    bin_size::T
end

"""Build a spatial index for all contour segments, binned by grid of size `delta`."""
function build_spatial_index(contours::Vector{PVContour{T}}, delta::T) where {T}
    bins = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (ci, c) in enumerate(contours)
        for ni in 1:nnodes(c)
            bx = floor(Int, c.nodes[ni][1] / delta)
            by = floor(Int, c.nodes[ni][2] / delta)
            key = (bx, by)
            if !haskey(bins, key)
                bins[key] = Tuple{Int,Int}[]
            end
            push!(bins[key], (ci, ni))
        end
    end

    return SpatialIndex(bins, delta)
end

# ── Contour Reconnection ────────────────────────────────

"""
    find_close_segments(contours, spatial_index, delta)

Find pairs of contour segments within distance `delta` using the spatial index.
Returns vector of `(ci, i, cj, j)` tuples.
"""
function find_close_segments(contours::Vector{PVContour{T}}, idx::SpatialIndex{T}, delta::T) where {T}
    close_pairs = Tuple{Int,Int,Int,Int}[]
    delta2 = delta^2

    for (ci, c) in enumerate(contours)
        # Skip spanning contours — they should not be reconnected
        is_spanning(c) && continue
        nc = nnodes(c)
        for i in 1:nc
            bx = floor(Int, c.nodes[i][1] / delta)
            by = floor(Int, c.nodes[i][2] / delta)

            # Check neighboring bins
            for dbx in -1:1, dby in -1:1
                key = (bx + dbx, by + dby)
                haskey(idx.bins, key) || continue
                for (cj, j) in idx.bins[key]
                    # Avoid duplicate pairs and adjacent nodes on same contour
                    (ci, i) >= (cj, j) && continue
                    # Skip spanning contours
                    is_spanning(contours[cj]) && continue
                    if ci == cj
                        ncj = nnodes(contours[cj])
                        # Skip adjacent or near-adjacent nodes
                        dist_along = min(abs(i - j), ncj - abs(i - j))
                        dist_along <= 2 && continue
                    else
                        # Different PV → skip
                        contours[ci].pv != contours[cj].pv && continue
                    end

                    d = contours[ci].nodes[i] - contours[cj].nodes[j]
                    if d[1]^2 + d[2]^2 < delta2
                        push!(close_pairs, (ci, i, cj, j))
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
"""
function reconnect!(contours::Vector{PVContour{T}}, close_pairs::Vector{Tuple{Int,Int,Int,Int}}) where {T}
    isempty(close_pairs) && return

    # Process one reconnection at a time (simplest correct approach)
    # After each reconnection, rebuild spatial index (handled by caller)
    pair = close_pairs[1]  # process first pair only per surgery call
    ci, i, cj, j = pair

    if ci == cj
        # Split: pinch off to create two contours
        c = contours[ci]
        nc = nnodes(c)
        # Split into nodes[i..j] and nodes[j..i] (wrapping), avoiding duplicate
        # boundary nodes.  Each sub-contour includes both pinch-point nodes
        # exactly once (as first and last traversed node).
        if i < j
            nodes1 = c.nodes[i:j]
            nodes2 = vcat(c.nodes[(j+1):nc], c.nodes[1:i])
        else
            nodes1 = c.nodes[j:i]
            nodes2 = vcat(c.nodes[(i+1):nc], c.nodes[1:j])
        end
        if length(nodes1) >= 3 && length(nodes2) >= 3
            contours[ci] = PVContour(nodes1, c.pv)
            push!(contours, PVContour(nodes2, c.pv))
        end
    else
        # Merge: combine two contours into one.
        # Stitch at closest points, traversing each contour once without
        # duplicating the splice-point nodes.
        c1 = contours[ci]
        c2 = contours[cj]
        n1 = nnodes(c1)
        n2 = nnodes(c2)
        new_nodes = vcat(
            c1.nodes[1:i],
            c2.nodes[j:n2],
            c2.nodes[1:(j-1)],
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
