# Contour surgery is organized as:
#   1. build a spatial index over non-spanning contour segments,
#   2. find admissible node-to-segment contacts within the cutoff scale,
#   3. split or merge independent contact pairs,
#   4. remesh with surgery corners fixed, then remove unresolved debris.
# Lower-level helpers in this file keep those phases separate so CPU, multilayer,
# and device paths can share the same topology rules.

# ── Periodic Helpers ────────────────────────────────────

"""Wrap a scalar coordinate to the canonical interval [-L, L) using floor."""
@inline function _wrap_coord(x::T, L::T) where {T}
    L2 = 2 * L
    return x - floor((x + L) / L2) * L2
end

"""
Minimum-image displacement vector for a periodic domain using round.

Note: uses round (nearest-integer) rather than floor, giving the interval (-L, L]
instead of [-L, L). The difference matters only at exact boundary values, which
are astronomically unlikely with floating-point arithmetic.
"""
@inline function _min_image(r::SVector{2,Tr}, domain::PeriodicDomain{Td}) where {Tr, Td}
    T = promote_type(Tr, Td)
    Lx2 = 2 * T(domain.Lx)
    Ly2 = 2 * T(domain.Ly)
    SVector{2,T}(T(r[1]) - round(T(r[1]) / Lx2) * Lx2,
                 T(r[2]) - round(T(r[2]) / Ly2) * Ly2)
end
@inline _min_image(r::SVector{2}, ::UnboundedDomain) = r

# ── Spatial Index ────────────────────────────────────────

struct SpatialIndex{T<:AbstractFloat}
    bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}  # (bin_x, bin_y) => [(contour_idx, node_idx)]
    bin_size::T
end

"""
Build a spatial index for all contour segments, binned by grid of size `bin_size`.
Each segment is binned at evenly spaced points no farther than `bin_size` apart
so that long segments whose interiors cross bin boundaries remain discoverable
via neighbour-bin queries.

For `PeriodicDomain`, coordinates are wrapped to the canonical domain and ghost
entries are inserted near boundaries so that segments physically close across
the periodic seam appear in adjacent bins.
"""
function build_spatial_index(contours::Vector{PVContour{T}}, bin_size,
                             domain::AbstractDomain=UnboundedDomain()) where {T}
    bin_size = T(bin_size)
    bins = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (ci, c) in enumerate(contours)
        is_spanning(c) && continue  # spanning contours never participate in reconnection
        nc = nnodes(c)
        nc < 3 && continue  # degenerate contours cannot participate in surgery
        for ni in 1:nc
            a = c.nodes[ni]
            b = next_node(c, ni)
            seg = b - a
            seg_len = sqrt(seg[1]^2 + seg[2]^2)
            # Bin at evenly spaced points along the segment, spaced at most bin_size
            # apart. This ensures every point on the segment is within bin_size/2 of
            # a binned point, so the 3×3 neighbourhood query in find_close_segments
            # can discover close pairs even when Δ_max >> δ.
            n_samples = max(2, ceil(Int, seg_len / bin_size) + 1)
            for k in 0:(n_samples - 1)
                t = T(k) / T(n_samples - 1)
                pt = a + t * seg
                _insert_bin!(bins, pt, ci, ni, bin_size, domain)
            end
        end
    end

    return SpatialIndex(bins, bin_size)
end

@inline function _push_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                            key::Tuple{Int,Int}, ci::Int, ni::Int)
    entry = (ci, ni)
    if !haskey(bins, key)
        bins[key] = [entry]
    else
        # Deduplicate: same segment can be sampled at multiple points in the
        # same bin.  Only need one entry per segment per bin; duplicates cause
        # redundant distance computations in find_close_segments.
        vec = bins[key]
        entry ∈ vec || push!(vec, entry)
    end
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                    pt::SVector{2,T}, ci::Int, ni::Int, δ::T,
                    ::UnboundedDomain) where {T}

    bx = floor(Int, pt[1] / δ)
    by = floor(Int, pt[2] / δ)
    _push_bin!(bins, (bx, by), ci, ni)
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                      pt::SVector{2,T}, ci::Int, ni::Int, δ::T,
                      domain::PeriodicDomain{T}) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    x_w = _wrap_coord(pt[1], Lx)
    y_w = _wrap_coord(pt[2], Ly)

    bx = floor(Int, x_w / δ)
    by = floor(Int, y_w / δ)
    _push_bin!(bins, (bx, by), ci, ni)

    # Ghost entries near periodic boundaries so that the 3×3 neighbour query
    # in find_close_segments discovers segments close across the seam.
    # Use 2*δ threshold: the query extends ±1 bin, so a segment up to
    # 2*δ from the seam can have its periodic image within δ of a
    # query on the opposite side.
    near_xhi = x_w > Lx  - 2δ
    near_xlo = x_w < -Lx + 2δ
    near_yhi = y_w > Ly  - 2δ
    near_ylo = y_w < -Ly + 2δ

    # Edge ghosts
    near_xhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), by), ci, ni)
    near_xlo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), by), ci, ni)
    near_yhi && _push_bin!(bins, (bx, floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_ylo && _push_bin!(bins, (bx, floor(Int, (y_w + 2Ly) / δ)), ci, ni)

    # Corner ghosts
    near_xhi && near_yhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_xhi && near_ylo && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), floor(Int, (y_w + 2Ly) / δ)), ci, ni)
    near_xlo && near_yhi && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_xlo && near_ylo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), floor(Int, (y_w + 2Ly) / δ)), ci, ni)
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

function _point_segment_closest(p::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                domain::AbstractDomain=UnboundedDomain()) where {T}
    # Return the squared distance from p to segment a-b and the closest point in
    # the segment's original coordinate image. Periodic domains are handled by
    # shifting the segment near the wrapped query point, then shifting back.
    p_img = _wrap_query_pt(p, domain)
    a_img, b_img = _shift_segment_to_image(a, b, p_img, domain)
    shift = a_img - a
    seg = b_img - a_img
    seg_len2 = seg[1]^2 + seg[2]^2
    if seg_len2 <= eps(T)
        d = p_img - a_img
        return d[1]^2 + d[2]^2, a
    end
    t = clamp(((p_img[1] - a_img[1]) * seg[1] + (p_img[2] - a_img[2]) * seg[2]) / seg_len2,
              zero(T), one(T))
    closest_img = a_img + t * seg
    d = p_img - closest_img
    return d[1]^2 + d[2]^2, closest_img - shift
end

function _point_segment_dist2(p::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    seg = b - a
    seg_len2 = seg[1]^2 + seg[2]^2
    if seg_len2 <= eps(T)
        d = p - a
        return d[1]^2 + d[2]^2
    end
    t = clamp(((p[1] - a[1]) * seg[1] + (p[2] - a[2]) * seg[2]) / seg_len2,
              zero(T), one(T))
    closest = a + t * seg
    d = p - closest
    return d[1]^2 + d[2]^2
end

function _surgery_contact_distance2(a1::SVector{2,T}, b1::SVector{2,T},
                                    a2::SVector{2,T}, b2::SVector{2,T}) where {T}
    # Dritschel surgery uses node-to-segment proximity, not full line-segment
    # intersection. Check both endpoints of each segment against the other
    # segment and keep the nearest endpoint contact.
    return min(_point_segment_dist2(a1, a2, b2),
               _point_segment_dist2(b1, a2, b2),
               _point_segment_dist2(a2, a1, b1),
               _point_segment_dist2(b2, a1, b1))
end

function _best_node_segment_contact(c1::PVContour{T}, i1::Int,
                                    c2::PVContour{T}, i2::Int,
                                    domain::AbstractDomain=UnboundedDomain()) where {T}
    # Choose which endpoint becomes the stitch node. The boolean in the return
    # value says whether the endpoint came from c1; seg_idx is the segment on the
    # other contour where the stitch node must be inserted.
    n1 = nnodes(c1)
    n2 = nnodes(c2)
    i1_end = mod1(i1 + 1, n1)
    i2_end = mod1(i2 + 1, n2)

    best_d2 = typemax(T)
    best = (true, i1, i2, c1.nodes[i1])

    a2 = c2.nodes[i2]
    b2 = next_node(c2, i2)
    for node_idx in (i1, i1_end)
        d2, _ = _point_segment_closest(c1.nodes[node_idx], a2, b2, domain)
        if d2 < best_d2
            best_d2 = d2
            best = (true, node_idx, i2, c1.nodes[node_idx])
        end
    end

    a1 = c1.nodes[i1]
    b1 = next_node(c1, i1)
    for node_idx in (i2, i2_end)
        d2, _ = _point_segment_closest(c2.nodes[node_idx], a1, b1, domain)
        if d2 < best_d2
            best_d2 = d2
            best = (false, node_idx, i1, c2.nodes[node_idx])
        end
    end

    return best
end

function _insert_stitch_node(nodes::Vector{SVector{2,T}}, seg_idx::Int,
                             point::SVector{2,T}) where {T}
    # Insert immediately after seg_idx, which is the local representation of a
    # point lying on segment seg_idx -> seg_idx+1. If seg_idx is the closing
    # segment, append so cyclic ordering is preserved.
    n = length(nodes)
    new_nodes = Vector{SVector{2,T}}(undef, n + 1)
    if seg_idx == n
        copyto!(new_nodes, 1, nodes, 1, n)
        new_nodes[n + 1] = point
        return new_nodes, n + 1
    end

    copyto!(new_nodes, 1, nodes, 1, seg_idx)
    new_nodes[seg_idx + 1] = point
    copyto!(new_nodes, seg_idx + 2, nodes, seg_idx + 1, n - seg_idx)
    return new_nodes, seg_idx + 1
end

function _insert_corner_flag(corners::BitVector, seg_idx::Int, flag::Bool=false)
    # Mirror _insert_stitch_node for the corner BitVector. New stitch nodes are
    # usually inserted unlabelled and promoted to fixed corners only when the
    # reconnect operation has decided which topology it is creating.
    n = length(corners)
    new_corners = falses(n + 1)
    if seg_idx == n
        copyto!(new_corners, 1, corners, 1, n)
        new_corners[n + 1] = flag
        return new_corners
    end

    copyto!(new_corners, 1, corners, 1, seg_idx)
    new_corners[seg_idx + 1] = flag
    copyto!(new_corners, seg_idx + 2, corners, seg_idx + 1, n - seg_idx)
    return new_corners
end

@inline function _segment_has_corner(c::PVContour, i::Int)
    # A segment touching a fixed corner is temporarily protected from additional
    # surgery so a fresh stitch is remeshed before being considered again.
    n = nnodes(c)
    return n > 0 && (c.corners[i] || c.corners[mod1(i + 1, n)])
end

function _demote_obtuse_corners(c::PVContour{T}) where {T}
    # Dritschel corners are temporary labels. Once local geometry has opened into
    # an obtuse angle, the node can return to the ordinary remeshing population.
    any(c.corners) || return c
    corners = copy(c.corners)
    changed = false
    n = nnodes(c)
    n < 3 && return c

    @inbounds for i in 1:n
        corners[i] || continue
        prev = c.nodes[mod1(i - 1, n)]
        curr = c.nodes[i]
        nxt = next_node(c, i)
        v_prev = prev - curr
        v_next = nxt - curr
        l_prev = sqrt(v_prev[1]^2 + v_prev[2]^2)
        l_next = sqrt(v_next[1]^2 + v_next[2]^2)
        (l_prev <= eps(T) || l_next <= eps(T)) && continue
        if v_prev[1] * v_next[1] + v_prev[2] * v_next[2] < zero(T)
            corners[i] = false
            changed = true
        end
    end

    return changed ? PVContour(c.nodes, c.pv, c.wrap, corners) : c
end

function _promote_high_curvature_corners(c::PVContour{T}, δ) where {T}
    # If a bend is sharper than the surgical cutoff can represent and is acute,
    # label it as a corner so subsequent remeshing keeps it fixed instead of
    # smearing the cusp over neighbouring nodes.
    is_spanning(c) && return c
    n = nnodes(c)
    n < 3 && return c

    corners = copy(c.corners)
    κ = _signed_node_curvatures(c)
    threshold = inv(T(δ))
    changed = false

    @inbounds for i in 1:n
        corners[i] && continue
        prev_i = mod1(i - 1, n)
        next_i = mod1(i + 1, n)
        (corners[prev_i] || corners[next_i]) && continue
        abs(κ[i]) < threshold && continue

        prev = i == 1 ? c.nodes[n] - c.wrap : c.nodes[prev_i]
        curr = c.nodes[i]
        nxt = next_node(c, i)
        v1 = prev - curr
        v2 = nxt - curr
        acute = v1[1] * v2[1] + v1[2] * v2[2] > zero(T)
        if acute
            corners[i] = true
            changed = true
        end
    end

    return changed ? PVContour(c.nodes, c.pv, c.wrap, corners) : c
end

function _promote_high_curvature_corners!(contours::Vector{PVContour{T}}, δ) where {T}
    for i in eachindex(contours)
        contours[i] = _promote_high_curvature_corners(contours[i], δ)
    end
    return contours
end

@inline function _ray_crosses_segment(pt::SVector{2,T},
                                      a::SVector{2,T},
                                      b::SVector{2,T}) where {T}
    y = pt[2]
    (a[2] > y) == (b[2] > y) && return false
    x_cross = a[1] + (y - a[2]) * (b[1] - a[1]) / (b[2] - a[2])
    return pt[1] < x_cross
end

function _point_in_closed_contour(pt::SVector{2,T}, c::PVContour{T},
                                  ::UnboundedDomain) where {T}
    # Even-odd ray casting. Spanning contours are intentionally excluded because
    # they do not define a closed interior in the local surgery sense.
    is_spanning(c) && return false
    inside = false
    @inbounds for i in 1:nnodes(c)
        inside = xor(inside, _ray_crosses_segment(pt, c.nodes[i], next_node(c, i)))
    end
    return inside
end

function _point_in_closed_contour(pt::SVector{2,T}, c::PVContour{T},
                                  domain::PeriodicDomain{T}) where {T}
    is_spanning(c) && return false
    pt_q = _wrap_query_pt(pt, domain)
    inside = false
    @inbounds for i in 1:nnodes(c)
        a_img, b_img = _shift_segment_to_image(c.nodes[i], next_node(c, i), pt_q, domain)
        inside = xor(inside, _ray_crosses_segment(pt_q, a_img, b_img))
    end
    return inside
end

function _segment_interior_probe(c::PVContour{T}, i::Int, δ,
                                 domain::AbstractDomain=UnboundedDomain()) where {T}
    # Probe just inside segment i. The sign of contour area tells us which side
    # is inward, so nested contours can be compared by the vorticity they enclose
    # locally rather than by PV jump alone.
    a = c.nodes[i]
    b = next_node(c, i)
    seg = b - a
    seg_len = sqrt(seg[1]^2 + seg[2]^2)
    seg_len <= eps(T) && return _wrap_query_pt((a + b) / 2, domain)

    area_sign = sign(vortex_area(c))
    area_sign == 0 && (area_sign = one(T))
    left_normal = SVector{2,T}(-seg[2] / seg_len, seg[1] / seg_len)
    inward = area_sign > 0 ? left_normal : -left_normal
    probe_distance = max(T(δ) / T(10), eps(T) * (one(T) + abs(a[1]) + abs(a[2]) + seg_len))
    return _wrap_query_pt((a + b) / 2 + probe_distance * inward, domain)
end

function _local_interior_vorticity(contours::Vector{PVContour{T}},
                                   ci::Int, i::Int, δ,
                                   domain::AbstractDomain=UnboundedDomain()) where {T}
    # Sum PV jumps of all closed contours containing the interior probe point.
    # Equal values mean two nearby contour parts bound the same fluid level and
    # are eligible to merge.
    pt = _segment_interior_probe(contours[ci], i, δ, domain)
    q = zero(T)
    for c in contours
        _point_in_closed_contour(pt, c, domain) && (q += c.pv)
    end
    return q
end

function _same_local_interior_vorticity(contours::Vector{PVContour{T}},
                                        ci::Int, i::Int, cj::Int, j::Int, δ,
                                        domain::AbstractDomain=UnboundedDomain()) where {T}
    # Keep this scalar predicate separate from find_close_segments so tests and
    # device kernels can validate the same merge admissibility rule.
    qi = _local_interior_vorticity(contours, ci, i, δ, domain)
    qj = _local_interior_vorticity(contours, cj, j, δ, domain)
    tol = sqrt(eps(T)) * max(one(T), abs(qi), abs(qj))
    return isapprox(qi, qj; atol=tol, rtol=sqrt(eps(T)))
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
    find_close_segments(contours, spatial_index, δ[, domain])

Find pairs of contour segments whose closest approach is within `δ`,
using the spatial index for candidate filtering.
Returns vector of `(ci, i, cj, j)` tuples where `i`,`j` are segment indices
(each segment goes from node `i` to `next_node(c, i)`).

For `PeriodicDomain`, minimum-image distances are used so that segments
close across the periodic boundary are correctly detected.
"""
function find_close_segments(contours::Vector{PVContour{T}}, idx::SpatialIndex{T}, δ,
                            domain::AbstractDomain=UnboundedDomain()) where {T}
    δ = T(δ)
    bin_size = idx.bin_size
    close_pairs = Tuple{Int,Int,Int,Int}[]
    δ2 = δ^2
    # Compact deduplication: encode each canonical pair as a UInt64 when indices
    # fit in 16 bits (covers up to 65535 contours × 65535 nodes each).  This
    # halves the per-entry memory vs Set{Tuple{Int,Int,Int,Int}} and speeds
    # hashing.  Fall back to the tuple Set for (impractically) large problems.
    max_idx = 0
    for c in contours
        max_idx = max(max_idx, nnodes(c))
    end
    use_compact = length(contours) <= typemax(UInt16) && max_idx <= typemax(UInt16)
    seen_compact = use_compact ? Set{UInt64}() : nothing
    seen_tuple = use_compact ? nothing : Set{Tuple{Int,Int,Int,Int}}()
    interior_q_cache = Dict{Tuple{Int,Int}, T}()

    @inline function _pair_seen(ci, i, cj, j)
        a, b, c_idx, d = (ci, i) < (cj, j) ? (ci, i, cj, j) : (cj, j, ci, i)
        if use_compact
            key = (UInt64(a) << 48) | (UInt64(b) << 32) | (UInt64(c_idx) << 16) | UInt64(d)
            return key in seen_compact::Set{UInt64}
        else
            return (a, b, c_idx, d) in seen_tuple::Set{Tuple{Int,Int,Int,Int}}
        end
    end

    @inline function _pair_insert!(ci, i, cj, j)
        a, b, c_idx, d = (ci, i) < (cj, j) ? (ci, i, cj, j) : (cj, j, ci, i)
        if use_compact
            key = (UInt64(a) << 48) | (UInt64(b) << 32) | (UInt64(c_idx) << 16) | UInt64(d)
            push!(seen_compact::Set{UInt64}, key)
        else
            push!(seen_tuple::Set{Tuple{Int,Int,Int,Int}}, (a, b, c_idx, d))
        end
    end

    function _cached_interior_q(ci, i)
        key = (ci, i)
        return get!(interior_q_cache, key) do
            _local_interior_vorticity(contours, ci, i, δ, domain)
        end
    end

    @inline function _cached_same_interior_q(ci, i, cj, j)
        qi = _cached_interior_q(ci, i)
        qj = _cached_interior_q(cj, j)
        tol = sqrt(eps(T)) * max(one(T), abs(qi), abs(qj))
        return isapprox(qi, qj; atol=tol, rtol=sqrt(eps(T)))
    end

    for (ci, c) in enumerate(contours)
        is_spanning(c) && continue
        nc = nnodes(c)
        for i in 1:nc
            _segment_has_corner(c, i) && continue
            a_i = c.nodes[i]
            b_i = next_node(c, i)
            mid_q = _wrap_query_pt((a_i + b_i) / 2, domain)

            seg_i = b_i - a_i
            seg_i_len = sqrt(seg_i[1]^2 + seg_i[2]^2)
            n_query = max(2, ceil(Int, seg_i_len / bin_size) + 1)
            query_bins = Tuple{Int,Int}[]
            for k in 0:(n_query - 1)
                t = T(k) / T(n_query - 1)
                pt = _wrap_query_pt(a_i + t * seg_i, domain)
                bx = floor(Int, pt[1] / bin_size)
                by = floor(Int, pt[2] / bin_size)
                for dbx in -1:1, dby in -1:1
                    key = (bx + dbx, by + dby)
                    key in query_bins || push!(query_bins, key)
                end
            end

            for key in query_bins
                haskey(idx.bins, key) || continue
                for (cj, j) in idx.bins[key]
                    _segment_has_corner(contours[cj], j) && continue
                    # Canonical ordering to avoid duplicates
                    _pair_seen(ci, i, cj, j) && continue
                    pair = (ci, i) < (cj, j) ? (ci, i, cj, j) : (cj, j, ci, i)
                    # Invariant: build_spatial_index (line 40) skips spanning contours, so cj is always non-spanning
                    if ci == cj
                        ncj = nnodes(contours[cj])
                        dist_along = min(abs(i - j), ncj - abs(i - j))
                        dist_along <= 2 && continue
                    else
                        # Dritschel's merge condition is stricter than equal PV
                        # jump: the contour parts must enclose identical interior
                        # vorticity.  This prevents cross-level reconnections in
                        # nested vortices where several contours carry the same
                        # jump but bound different vorticity levels.
                        pv_i, pv_j = contours[ci].pv, contours[cj].pv
                        !isapprox(pv_i, pv_j; atol=sqrt(eps(T)), rtol=sqrt(eps(T))) && continue
                        _cached_same_interior_q(ci, i, cj, j) || continue
                    end

                    a_j = contours[cj].nodes[j]
                    b_j = next_node(contours[cj], j)

                    # Shift both segments to be near the wrapped midpoint of
                    # segment i, so the distance test uses consistent images.
                    # Without this, a_i drifted outside [-Lx,Lx) would be
                    # compared against a_j_img near the wrapped midpoint.
                    a_i_img, b_i_img = _shift_segment_to_image(a_i, b_i, mid_q, domain)
                    a_j_img, b_j_img = _shift_segment_to_image(a_j, b_j, mid_q, domain)

                    if _surgery_contact_distance2(a_i_img, b_i_img, a_j_img, b_j_img) < δ2
                        _pair_insert!(ci, i, cj, j)
                        push!(close_pairs, pair)
                    end
                end
            end
        end
    end

    return close_pairs
end

function _close_pair_distance2(contours::Vector{PVContour{T}},
                               pair::Tuple{Int,Int,Int,Int},
                               domain::AbstractDomain=UnboundedDomain()) where {T}
    # Recompute the same endpoint-to-segment surgery distance used during
    # candidate discovery. This is used only for ranking selected pairs.
    ci, i, cj, j = pair
    c_i = contours[ci]
    c_j = contours[cj]
    a_i = c_i.nodes[i]
    b_i = next_node(c_i, i)
    a_j = c_j.nodes[j]
    b_j = next_node(c_j, j)
    mid_q = _wrap_query_pt((a_i + b_i) / 2, domain)
    a_i_img, b_i_img = _shift_segment_to_image(a_i, b_i, mid_q, domain)
    a_j_img, b_j_img = _shift_segment_to_image(a_j, b_j, mid_q, domain)
    return _surgery_contact_distance2(a_i_img, b_i_img, a_j_img, b_j_img)
end

function _select_reconnection_pairs(contours::Vector{PVContour{T}},
                                    close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                    domain::AbstractDomain=UnboundedDomain()) where {T}
    isempty(close_pairs) && return close_pairs

    # Sort by physical proximity so the most urgent contacts are handled before
    # weaker candidates that may disappear after remeshing.
    ranked = Vector{Tuple{T,Tuple{Int,Int,Int,Int}}}(undef, length(close_pairs))
    for (k, pair) in pairs(close_pairs)
        ranked[k] = (_close_pair_distance2(contours, pair, domain), pair)
    end
    sort!(ranked, by=first)

    # Process the closest independent reconnections first.  Limiting each
    # contour to one reconnect per pass avoids repeated local surgery on the
    # same bridge before remeshing has removed near-duplicate stitch nodes.
    used_contours = Set{Int}()
    selected = Tuple{Int,Int,Int,Int}[]
    sizehint!(selected, min(length(close_pairs), length(contours)))
    for (_, pair) in ranked
        ci, _, cj, _ = pair
        (ci in used_contours || cj in used_contours) && continue
        push!(selected, pair)
        push!(used_contours, ci)
        push!(used_contours, cj)
    end
    return selected
end

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
function _shoelace_area(nodes::AbstractVector{SVector{2,T}}) where {T}
    n = length(nodes)
    n < 3 && return zero(T)
    s = zero(T)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        s += nodes[i][1] * nodes[j][2] - nodes[j][1] * nodes[i][2]
    end
    return s / 2
end

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

function _reconnect_merge!(contours::Vector{PVContour{T}}, ci::Int, i::Int, cj::Int, j::Int,
                           domain::AbstractDomain=UnboundedDomain()) where {T}
    c1 = contours[ci]
    c2 = contours[cj]

    # Check orientation consistency: both contours should have the same
    # sign of signed area. If they differ, reverse c2's node order so
    # the merged contour has consistent winding.
    # Use a robust threshold: only reverse if c2 has a meaningfully signed area
    # (well above numerical noise from the shoelace formula).
    a1 = vortex_area(c1)
    a2 = vortex_area(c2)
    reversed = abs(a1) > eps(T) * T(1000) && abs(a2) > eps(T) * T(1000) && sign(a1) != sign(a2)
    c2_nodes = reversed ? reverse(c2.nodes) : c2.nodes
    c2_corners = reversed ? reverse(c2.corners) : copy(c2.corners)
    n2_orig = nnodes(c2)
    j_seg = reversed ? mod1(n2_orig - j, n2_orig) : j
    c2_eff = PVContour(c2_nodes, c2.pv, reversed ? -c2.wrap : c2.wrap, c2_corners)

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

    # For periodic domains, shift c2 to closest image of the merge point.
    shift = _periodic_merge_shift(c1_nodes[i], c2_nodes[j_eff], domain)
    if !iszero(shift)
        c2_nodes = [n + shift for n in c2_nodes]
    end

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

# ── Filament Removal ─────────────────────────────────────

function _contour_perimeter(c::PVContour{T}) where {T}
    # Perimeter is used as a cheap thickness proxy during filament cleanup.
    n = nnodes(c)
    n < 2 && return zero(T)
    perimeter = zero(T)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        perimeter += sqrt(d[1]^2 + d[2]^2)
    end
    return perimeter
end

"""Remove closed contours below the retained area or perimeter scale."""
function _is_corner_filament(c::PVContour{T}, area::T, perimeter::T, μ::T) where {T}
    any(c.corners) || return false
    # Dritschel (1988) removes contours with too few nodes (four or fewer) as
    # unresolved surgical debris.  Restricting this rule to labelled-corner
    # contours preserves coarse user-provided input contours until remeshing has
    # a chance to resolve them.
    nnodes(c) <= 4 && return true
    μ > zero(T) || return false
    perimeter <= eps(T) && return true
    width = T(2) * area / perimeter
    return area <= μ^2 || width < μ
end

"""
    remove_filaments!(contours, area_min[, μ])

Remove unresolved closed contours from `contours` in place.

Closed contours with area below `area_min` are discarded. When `μ` is supplied,
very short or thin labelled-corner fragments produced by reconnection are also
removed. Spanning contours are retained.
"""
function remove_filaments!(contours::Vector{PVContour{T}}, area_min, μ=nothing) where {T}
    # Remove unresolved closed debris. The optional μ cutoff also rejects very
    # short or thin labelled-corner fragments produced by reconnection.
    amin = T(area_min)
    μ_cut = μ === nothing ? zero(T) : T(μ)
    min_perimeter = μ === nothing ? zero(T) : T(4) * μ_cut
    filter!(contours) do c
        is_spanning(c) && return true
        nnodes(c) >= 3 || return false
        area = abs(vortex_area(c))
        perimeter = _contour_perimeter(c)
        area >= amin &&
            perimeter >= min_perimeter &&
            !_is_corner_filament(c, area, perimeter, μ_cut)
    end
end

function _demote_obtuse_corners!(contours::Vector{PVContour{T}}) where {T}
    # Batch wrapper used after every remesh stage, when fixed corners may have
    # relaxed into ordinary smooth contour points.
    for i in eachindex(contours)
        contours[i] = _demote_obtuse_corners(contours[i])
    end
    return contours
end

# ── Top-Level Surgery ────────────────────────────────────

"""
    _check_spanning_proximity(contours, δ[, domain])

Warn if any closed contour node is within `δ` of a spanning contour node.
This situation cannot be resolved by surgery (spanning contours are exempt from
reconnection) and may indicate insufficient resolution or an overly large δ.

For `PeriodicDomain`, minimum-image distances are used.
"""
function _check_spanning_proximity(contours::Vector{PVContour{T}}, δ,
                                   domain::AbstractDomain=UnboundedDomain()) where {T}
    δ = T(δ)
    δ2 = δ^2
    # Bin spanning nodes for O(1) proximity lookup instead of O(N_spanning) per query.
    # Wrap coordinates before binning so that spanning nodes (which may have drifted
    # outside [-Lx,Lx) since wrap_nodes! skips them) land in the same bin space as
    # the wrapped closed-contour nodes.
    spanning_bins = Dict{Tuple{Int,Int}, Vector{SVector{2,T}}}()
    has_spanning = false
    _push_spanning_bin!(bins, key, sn) = (v = get!(bins, key, SVector{2,T}[]); push!(v, sn))
    for c in contours
        is_spanning(c) || continue
        has_spanning = true
        for sn in c.nodes
            sn_w = _wrap_query_pt(sn, domain)
            bx = floor(Int, sn_w[1] / δ)
            by = floor(Int, sn_w[2] / δ)
            _push_spanning_bin!(spanning_bins, (bx, by), sn)
            # Ghost entries near periodic boundaries (same 2*δ threshold
            # as _insert_bin!) so 3×3 queries find spanning nodes across seams.
            if domain isa PeriodicDomain
                Lx, Ly = domain.Lx, domain.Ly
                two_δ = 2 * δ
                near_xhi = sn_w[1] > Lx - two_δ
                near_xlo = sn_w[1] < -Lx + two_δ
                near_yhi = sn_w[2] > Ly - two_δ
                near_ylo = sn_w[2] < -Ly + two_δ
                near_xhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), by), sn)
                near_xlo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), by), sn)
                near_yhi && _push_spanning_bin!(spanning_bins, (bx, floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_ylo && _push_spanning_bin!(spanning_bins, (bx, floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
                near_xhi && near_yhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_xhi && near_ylo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
                near_xlo && near_yhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_xlo && near_ylo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
            end
        end
    end
    has_spanning || return
    for c in contours
        is_spanning(c) && continue
        for node in c.nodes
            node_w = _wrap_query_pt(node, domain)
            bx = floor(Int, node_w[1] / δ)
            by = floor(Int, node_w[2] / δ)
            for dbx in -1:1, dby in -1:1
                key = (bx + dbx, by + dby)
                haskey(spanning_bins, key) || continue
                for sn in spanning_bins[key]
                    r = _min_image(node - sn, domain)
                    d2 = r[1]^2 + r[2]^2
                    if d2 < δ2
                        @warn "surgery!: closed contour node within δ of spanning contour — this cannot be resolved by reconnection" distance=sqrt(d2) δ maxlog=1
                        return
                    end
                end
            end
        end
    end
end

"""
    surgery!(prob::ContourProblem, params::SurgeryParams)

Dritschel-style contour-surgery pass:
pre-clean unresolved debris → update corner labels → remesh → reconnect close
compatible contour parts until exhausted → post-remesh → remove unresolved
filaments.

This implements the standard topological surgery loop from Dritschel's contour
surgery algorithm (Dritschel 1988, Table III): identify admissible close contour
parts enclosing the same interior vorticity, split or merge them, repeat until
exhausted, and redistribute nodes afterward. Reconnection creates labelled
corner nodes that remain fixed during remeshing until they become obtuse.
Remeshing uses the Dritschel nonlocal node-density rule with cubic interpolation
arcs; the velocity paths use the same signed-curvature geometry when evaluating
curved segments.
Mutates `prob.contours` in place.
"""
function surgery!(prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T}, params::SurgeryParams) where {T}
    contours = prob.contours
    domain = prob.domain

    # Drop already-subgrid closed contours before remeshing so tiny filaments
    # do not trigger degenerate redistribution warnings.
    remove_filaments!(contours, params.area_min, params.μ)
    _demote_obtuse_corners!(contours)
    _promote_high_curvature_corners!(contours, params.δ)

    # 1. Remesh all contours. This corresponds to Dritschel's node
    # redistribution step: fixed surgery corners divide the contour into spans,
    # while non-corner contours keep one reference node fixed.
    # Pre-allocate workspace buffers to avoid per-contour allocations in remesh.
    _remesh_buf = SVector{2, T}[]
    _arc_buf = T[]
    _vnodes_buf = SVector{2, T}[]
    density_sources = copy(contours)
    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params;
                             _buf=_remesh_buf,
                             _arc_buf=_arc_buf,
                             _vnodes_buf=_vnodes_buf,
                             _density_sources=density_sources)
    end
    _demote_obtuse_corners!(contours)

    _cleanup_reconnect_artifacts!() = begin
        density_sources = copy(contours)
        for i in eachindex(contours)
            contours[i] = remesh(contours[i], params;
                                 _buf=_remesh_buf,
                                 _arc_buf=_arc_buf,
                                 _vnodes_buf=_vnodes_buf,
                                 _density_sources=density_sources)
        end
        _demote_obtuse_corners!(contours)
    end
    reconnection_bin_size = max(T(params.δ), T(params.μ))

    # 2. Reconnection — iterate until no more close pairs remain.
    #    Stall detection: stop early if close-pair count is not making progress.
    #    We track both consecutive increases and failure to improve the minimum,
    #    catching both monotone growth and oscillation (e.g. 10→12→10→12).
    reconnected = false
    max_reconnect_iter = 100
    stall_warning_pairs = 100
    prev_n_pairs = typemax(Int)
    min_n_pairs = typemax(Int)
    stall_count = 0
    no_improve_count = 0
    for iter in 1:max_reconnect_iter
        idx = build_spatial_index(contours, reconnection_bin_size, domain)
        close_pairs = find_close_segments(contours, idx, params.δ, domain)
        isempty(close_pairs) && break
        n_pairs = length(close_pairs)
        if n_pairs > prev_n_pairs
            stall_count += 1
        else
            stall_count = 0
        end
        if n_pairs < min_n_pairs
            min_n_pairs = n_pairs
            no_improve_count = 0
        else
            no_improve_count += 1
        end
        if stall_count >= 3 || no_improve_count >= 6
            # Reconnection creates near-duplicate stitch nodes by construction.
            # Before warning about a stall, run the cleanup remesh that surgery!
            # would perform after a successful reconnect and re-check proximity.
            if reconnected
                _cleanup_reconnect_artifacts!()
                remove_filaments!(contours, params.area_min, params.μ)
                idx = build_spatial_index(contours, reconnection_bin_size, domain)
                close_pairs = find_close_segments(contours, idx, params.δ, domain)
                isempty(close_pairs) && break
                remeshed_n_pairs = length(close_pairs)
                if remeshed_n_pairs < n_pairs
                    prev_n_pairs = remeshed_n_pairs
                    min_n_pairs = min(min_n_pairs, remeshed_n_pairs)
                    stall_count = 0
                    no_improve_count = 0
                    continue
                end
                n_pairs = remeshed_n_pairs
            end
            if n_pairs >= stall_warning_pairs
                @warn "surgery!: reconnection stalled ($n_pairs close pairs, min seen: $min_n_pairs) — stopping early"
            end
            break
        end
        prev_n_pairs = n_pairs
        reconnect!(contours, close_pairs, domain)
        reconnected = true
        remove_filaments!(contours, params.area_min, params.μ)
        _cleanup_reconnect_artifacts!()
        remove_filaments!(contours, params.area_min, params.μ)
        if iter == max_reconnect_iter
            @warn "surgery!: reconnection iteration limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
        end
    end

    # 3. Re-remesh after reconnection to clean up short/long segments
    #    created at stitch junctions during merge or split.
    if reconnected
        _cleanup_reconnect_artifacts!()
    end

    # 4. Remove filaments
    remove_filaments!(contours, params.area_min, params.μ)

    # 5. Warn if closed contours are too close to spanning contours
    _check_spanning_proximity(contours, params.δ, domain)

    return prob
end
