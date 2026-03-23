# ── Periodic Helpers ────────────────────────────────────

"""Wrap a scalar coordinate to the canonical interval [-L, L)."""
@inline function _wrap_coord(x::T, L::T) where {T}
    L2 = 2 * L
    return x - floor((x + L) / L2) * L2
end

"""Minimum-image displacement vector for a periodic domain."""
@inline function _min_image(r::SVector{2,T}, domain::PeriodicDomain{T}) where {T}
    Lx2 = 2 * domain.Lx
    Ly2 = 2 * domain.Ly
    SVector{2,T}(r[1] - round(r[1] / Lx2) * Lx2,
                 r[2] - round(r[2] / Ly2) * Ly2)
end
@inline _min_image(r::SVector{2}, ::UnboundedDomain) = r

# ── Spatial Index ────────────────────────────────────────

struct SpatialIndex{T<:AbstractFloat}
    bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}  # (bin_x, bin_y) => [(contour_idx, node_idx)]
    bin_size::T
end

"""
Build a spatial index for all contour segments, binned by grid of size `delta`.
Each segment is binned at its endpoint *and* at its midpoint so that long segments
whose interiors cross a bin boundary are discoverable via neighbour-bin queries.

For `PeriodicDomain`, coordinates are wrapped to the canonical domain and ghost
entries are inserted near boundaries so that segments physically close across
the periodic seam appear in adjacent bins.
"""
function build_spatial_index(contours::Vector{PVContour{T}}, delta,
                             domain::AbstractDomain=UnboundedDomain()) where {T}
    delta = T(delta)
    bins = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (ci, c) in enumerate(contours)
        nc = nnodes(c)
        for ni in 1:nc
            a = c.nodes[ni]
            b = next_node(c, ni)
            seg = b - a
            seg_len = sqrt(seg[1]^2 + seg[2]^2)
            # Bin at evenly spaced points along the segment, spaced at most delta
            # apart. This ensures every point on the segment is within delta/2 of
            # a binned point, so the 3×3 neighbourhood query in find_close_segments
            # can discover close pairs even when Delta_max >> delta.
            n_samples = max(2, ceil(Int, seg_len / delta) + 1)
            for k in 0:(n_samples - 1)
                t = T(k) / T(n_samples - 1)
                pt = a + t * seg
                _insert_bin!(bins, pt, ci, ni, delta, domain)
            end
        end
    end

    return SpatialIndex(bins, delta)
end

@inline function _push_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                            key::Tuple{Int,Int}, ci::Int, ni::Int)
    if !haskey(bins, key)
        bins[key] = Tuple{Int,Int}[]
    end
    push!(bins[key], (ci, ni))
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                      pt::SVector{2,T}, ci::Int, ni::Int, delta::T,
                      ::UnboundedDomain) where {T}
    bx = floor(Int, pt[1] / delta)
    by = floor(Int, pt[2] / delta)
    _push_bin!(bins, (bx, by), ci, ni)
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                      pt::SVector{2,T}, ci::Int, ni::Int, delta::T,
                      domain::PeriodicDomain{T}) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    x_w = _wrap_coord(pt[1], Lx)
    y_w = _wrap_coord(pt[2], Ly)

    bx = floor(Int, x_w / delta)
    by = floor(Int, y_w / delta)
    _push_bin!(bins, (bx, by), ci, ni)

    # Ghost entries near periodic boundaries so that the 3×3 neighbour query
    # in find_close_segments discovers segments close across the seam.
    near_xhi = x_w > Lx - delta
    near_xlo = x_w < -Lx + delta
    near_yhi = y_w > Ly - delta
    near_ylo = y_w < -Ly + delta

    # Edge ghosts
    near_xhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / delta), by), ci, ni)
    near_xlo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / delta), by), ci, ni)
    near_yhi && _push_bin!(bins, (bx, floor(Int, (y_w - 2Ly) / delta)), ci, ni)
    near_ylo && _push_bin!(bins, (bx, floor(Int, (y_w + 2Ly) / delta)), ci, ni)

    # Corner ghosts
    near_xhi && near_yhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / delta), floor(Int, (y_w - 2Ly) / delta)), ci, ni)
    near_xhi && near_ylo && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / delta), floor(Int, (y_w + 2Ly) / delta)), ci, ni)
    near_xlo && near_yhi && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / delta), floor(Int, (y_w - 2Ly) / delta)), ci, ni)
    near_xlo && near_ylo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / delta), floor(Int, (y_w + 2Ly) / delta)), ci, ni)
end

# ── Contour Reconnection ────────────────────────────────

"""
    _segment_min_dist2(a1, b1, a2, b2)

Minimum squared distance between segments `a1→b1` and `a2→b2`.

Uses the full parametric closest-point algorithm (Ericson, "Real-Time Collision
Detection") to handle all cases: endpoint-to-segment, endpoint-to-endpoint,
and interior-to-interior closest points.
"""
function _segment_min_dist2(a1::SVector{2,T}, b1::SVector{2,T},
                            a2::SVector{2,T}, b2::SVector{2,T}) where {T}
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2

    a = d1[1]^2 + d1[2]^2            # |d1|²
    e = d2[1]^2 + d2[2]^2            # |d2|²
    f = d2[1] * r[1] + d2[2] * r[2]  # d2 · r

    ε = eps(T)

    # Both segments degenerate to points
    if a <= ε && e <= ε
        return r[1]^2 + r[2]^2
    end

    local s::T, t::T

    if a <= ε
        # First segment degenerates to a point
        s = zero(T)
        t = clamp(f / e, zero(T), one(T))
    else
        c = d1[1] * r[1] + d1[2] * r[2]  # d1 · r
        if e <= ε
            # Second segment degenerates to a point
            t = zero(T)
            s = clamp(-c / a, zero(T), one(T))
        else
            # General non-degenerate case
            b_dot = d1[1] * d2[1] + d1[2] * d2[2]  # d1 · d2
            denom = a * e - b_dot * b_dot            # always >= 0

            # Closest point on the infinite lines
            if denom > ε * a
                s = clamp((b_dot * f - c * e) / denom, zero(T), one(T))
            else
                # Segments are nearly parallel
                s = zero(T)
            end

            # Optimal t from s
            t = (b_dot * s + f) / e

            # Clamp t and recompute s if needed
            if t < zero(T)
                t = zero(T)
                s = clamp(-c / a, zero(T), one(T))
            elseif t > one(T)
                t = one(T)
                s = clamp((b_dot - c) / a, zero(T), one(T))
            end
        end
    end

    diff = (a1 + s * d1) - (a2 + t * d2)
    return diff[1]^2 + diff[2]^2
end

# ── Periodic helpers for find_close_segments ─────────────

@inline _wrap_query_pt(pt::SVector{2,T}, ::UnboundedDomain) where {T} = pt
@inline function _wrap_query_pt(pt::SVector{2,T}, domain::PeriodicDomain{T}) where {T}
    SVector{2,T}(_wrap_coord(pt[1], domain.Lx), _wrap_coord(pt[2], domain.Ly))
end

@inline _shift_segment_to_image(a, b, ref, ::UnboundedDomain) = (a, b)
@inline function _shift_segment_to_image(a::SVector{2,T}, b::SVector{2,T},
                                          ref::SVector{2,T},
                                          domain::PeriodicDomain{T}) where {T}
    mid = (a + b) / 2
    raw = ref - mid
    mi = _min_image(raw, domain)
    shift = raw - mi
    iszero(shift) && return (a, b)
    return (a + shift, b + shift)
end

"""
    find_close_segments(contours, spatial_index, delta[, domain])

Find pairs of contour segments whose closest approach is within `delta`,
using the spatial index for candidate filtering.
Returns vector of `(ci, i, cj, j)` tuples where `i`,`j` are segment indices
(each segment goes from node `i` to `next_node(c, i)`).

For `PeriodicDomain`, minimum-image distances are used so that segments
close across the periodic boundary are correctly detected.
"""
function find_close_segments(contours::Vector{PVContour{T}}, idx::SpatialIndex{T}, delta,
                             domain::AbstractDomain=UnboundedDomain()) where {T}
    delta = T(delta)
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

            # Wrap query midpoint for consistent bin lookup in periodic domains
            mid_q = _wrap_query_pt(mid_i, domain)
            bx = floor(Int, mid_q[1] / delta)
            by = floor(Int, mid_q[2] / delta)

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

                    # Shift segment j to closest periodic image of segment i
                    # Use wrapped mid_q (not raw mid_i) so the minimum-image
                    # computation selects the correct periodic replica.
                    a_j_img, b_j_img = _shift_segment_to_image(a_j, b_j, mid_q, domain)

                    if _segment_min_dist2(a_i, b_i, a_j_img, b_j_img) < delta2
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
    _best_stitch_nodes(c1, i1, c2, i2[, domain])

Given that segment `i1` of contour `c1` is close to segment `i2` of contour `c2`,
find the pair of node indices (one from each contour) that are closest.
Returns `(best_i1, best_i2)` — node indices into `c1.nodes` and `c2.nodes`.

For `PeriodicDomain`, minimum-image distances are used.
"""
function _best_stitch_nodes(c1::PVContour{T}, i1::Int, c2::PVContour{T}, i2::Int,
                            domain::AbstractDomain=UnboundedDomain()) where {T}
    nc1 = nnodes(c1)
    nc2 = nnodes(c2)
    # Candidate node indices: start and end of each close segment
    i1_end = mod1(i1 + 1, nc1)
    i2_end = mod1(i2 + 1, nc2)
    best_d2 = typemax(T)
    best = (i1, i2)
    for ni in (i1, i1_end)
        p1 = c1.nodes[ni]
        for nj in (i2, i2_end)
            p2 = c2.nodes[nj]
            r = _min_image(p1 - p2, domain)
            d2 = r[1]^2 + r[2]^2
            if d2 < best_d2
                best_d2 = d2
                best = (ni, nj)
            end
        end
    end
    return best
end

"""
    reconnect!(contours, close_pairs[, domain])

Perform contour reconnection for identified close segment pairs.
Same contour → split; different contours with same PV → merge.

Each sub-contour produced by a split contains both pinch-point nodes as
its first and last vertices, ensuring a well-formed closing segment.
Merged contours are stitched so that traversal orientation is preserved.

!!! warning
    Reconnection produces near-duplicate nodes at stitch points.
    Callers should [`remesh`](@ref) all contours after reconnection to
    clean up short/long segments.  The top-level [`surgery!`](@ref) does
    this automatically.
"""
function reconnect!(contours::Vector{PVContour{T}}, close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                    domain::AbstractDomain=UnboundedDomain()) where {T}
    isempty(close_pairs) && return

    # Process all independent pairs per iteration to reduce spatial index rebuilds.
    # "Independent" means no shared contour indices between processed pairs.
    # Splits first (they don't shift indices), then at most one merge (deleteat! shifts).
    used = Set{Int}()

    # Pass 1: batch all independent splits (ci == cj, no index shifting)
    for (ci, i, cj, j) in close_pairs
        ci != cj && continue
        ci in used && continue
        _reconnect_split!(contours, ci, i, j)
        push!(used, ci)
    end

    # Pass 2: batch independent merges, processing in decreasing cj order.
    # Since ci < cj (canonical ordering from find_close_segments), deleting cj
    # never invalidates indices of remaining pairs whose cj values are smaller.
    merge_pairs = Tuple{Int,Int,Int,Int}[]
    for (ci, i, cj, j) in close_pairs
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

function _reconnect_split!(contours::Vector{PVContour{T}}, ci::Int, i::Int, j::Int) where {T}
    c = contours[ci]
    i, j = _best_stitch_nodes(c, i, c, j)
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
end

function _reconnect_merge!(contours::Vector{PVContour{T}}, ci::Int, i::Int, cj::Int, j::Int,
                           domain::AbstractDomain=UnboundedDomain()) where {T}
    c1 = contours[ci]
    c2 = contours[cj]
    n1 = nnodes(c1)
    n2 = nnodes(c2)

    i, j = _best_stitch_nodes(c1, i, c2, j, domain)

    # Check orientation consistency: both contours should have the same
    # sign of signed area. If they differ, reverse c2's node order so
    # the merged contour has consistent winding.
    # Use a robust threshold: only reverse if c2 has a meaningfully signed area
    # (well above numerical noise from the shoelace formula).
    a1 = vortex_area(c1)
    a2 = vortex_area(c2)
    reversed = sign(a1) != sign(a2) && abs(a2) > eps(T) * T(1000)
    c2_nodes = reversed ? reverse(c2.nodes) : c2.nodes
    j_eff = reversed ? (n2 - j + 1) : j

    # For periodic domains, shift c2 to closest image of the merge point
    shift = _periodic_merge_shift(c1.nodes[i], c2_nodes[j_eff], domain)
    if !iszero(shift)
        c2_nodes = [n + shift for n in c2_nodes]
    end

    new_nodes = vcat(
        c1.nodes[1:i],
        c2_nodes[j_eff:n2],
        c2_nodes[1:j_eff-1],  # exclude j_eff to avoid duplicate at stitch point
        c1.nodes[(i+1):n1]
    )
    contours[ci] = PVContour(new_nodes, c1.pv)
    deleteat!(contours, cj)
end

@inline _periodic_merge_shift(::SVector{2,T}, ::SVector{2,T}, ::UnboundedDomain) where {T} = zero(SVector{2,T})
@inline function _periodic_merge_shift(ref::SVector{2,T}, pt::SVector{2,T},
                                        domain::PeriodicDomain{T}) where {T}
    raw = ref - pt
    return raw - _min_image(raw, domain)
end

# ── Filament Removal ─────────────────────────────────────

"""Remove contours with |area| < area_min. Spanning contours are always kept."""
function remove_filaments!(contours::Vector{PVContour{T}}, area_min) where {T}
    amin = T(area_min)
    filter!(c -> is_spanning(c) || abs(vortex_area(c)) >= amin, contours)
end

# ── Top-Level Surgery ────────────────────────────────────

"""
    _check_spanning_proximity(contours, delta[, domain])

Warn if any closed contour node is within `delta` of a spanning contour node.
This situation cannot be resolved by surgery (spanning contours are exempt from
reconnection) and may indicate insufficient resolution or an overly large delta.

For `PeriodicDomain`, minimum-image distances are used.
"""
function _check_spanning_proximity(contours::Vector{PVContour{T}}, delta,
                                   domain::AbstractDomain=UnboundedDomain()) where {T}
    delta = T(delta)
    delta2 = delta^2
    # Bin spanning nodes for O(1) proximity lookup instead of O(N_spanning) per query.
    # Wrap coordinates before binning so that spanning nodes (which may have drifted
    # outside [-Lx,Lx) since wrap_nodes! skips them) land in the same bin space as
    # the wrapped closed-contour nodes.
    spanning_bins = Dict{Tuple{Int,Int}, Vector{SVector{2,T}}}()
    has_spanning = false
    for c in contours
        is_spanning(c) || continue
        has_spanning = true
        for sn in c.nodes
            sn_w = _wrap_query_pt(sn, domain)
            bx = floor(Int, sn_w[1] / delta)
            by = floor(Int, sn_w[2] / delta)
            key = (bx, by)
            if !haskey(spanning_bins, key)
                spanning_bins[key] = SVector{2,T}[]
            end
            push!(spanning_bins[key], sn)  # store original (unwrapped) for _min_image
        end
    end
    has_spanning || return
    for c in contours
        is_spanning(c) && continue
        for node in c.nodes
            node_w = _wrap_query_pt(node, domain)
            bx = floor(Int, node_w[1] / delta)
            by = floor(Int, node_w[2] / delta)
            for dbx in -1:1, dby in -1:1
                key = (bx + dbx, by + dby)
                haskey(spanning_bins, key) || continue
                for sn in spanning_bins[key]
                    r = _min_image(node - sn, domain)
                    d2 = r[1]^2 + r[2]^2
                    if d2 < delta2
                        @warn "surgery!: closed contour node within delta of spanning contour — this cannot be resolved by reconnection" distance=sqrt(d2) delta maxlog=1
                        return
                    end
                end
            end
        end
    end
end

"""
    surgery!(prob::ContourProblem, params::SurgeryParams)

Full Dritschel surgery suite: remesh → reconnect → remove filaments.
Mutates `prob.contours` in place.
"""
function surgery!(prob::ContourProblem, params::SurgeryParams)
    contours = prob.contours
    domain = prob.domain

    # 1. Remesh all contours
    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params)
    end

    # 2. Reconnection — iterate until no more close pairs remain.
    #    Stall detection: if close-pair count fails to decrease for 3
    #    consecutive iterations, stop early to avoid infinite cycling.
    reconnected = false
    max_reconnect_iter = 100
    prev_n_pairs = typemax(Int)
    stall_count = 0
    for iter in 1:max_reconnect_iter
        idx = build_spatial_index(contours, params.delta, domain)
        close_pairs = find_close_segments(contours, idx, params.delta, domain)
        isempty(close_pairs) && break
        n_pairs = length(close_pairs)
        if n_pairs >= prev_n_pairs
            stall_count += 1
            if stall_count >= 3
                @warn "surgery!: reconnection stalled (close pairs not decreasing: $n_pairs) — stopping early"
                break
            end
        else
            stall_count = 0
        end
        prev_n_pairs = n_pairs
        reconnect!(contours, close_pairs, domain)
        reconnected = true
        if iter == max_reconnect_iter
            @warn "surgery!: reconnection iteration limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
        end
    end

    # 3. Re-remesh after reconnection to clean up short/long segments
    #    created at stitch junctions during merge or split.
    if reconnected
        for i in eachindex(contours)
            contours[i] = remesh(contours[i], params)
        end
    end

    # 4. Remove filaments
    remove_filaments!(contours, params.area_min)

    # 5. Warn if closed contours are too close to spanning contours
    _check_spanning_proximity(contours, params.delta, domain)

    return prob
end
