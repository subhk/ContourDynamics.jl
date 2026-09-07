# surgery/pairs.jl — CPU surgery stage.

@inline function _ray_crosses_segment(pt::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return _flat_ray_crosses_segment(pt[1], pt[2], a[1], a[2], b[1], b[2])
end

function _point_in_closed_contour(pt::SVector{2,T}, c::PVContour{T}, ::UnboundedDomain) where {T}
    # Even-odd ray casting. Spanning contours are intentionally excluded because
    # they do not define a closed interior in the local surgery sense.
    is_spanning(c) && return false
    inside = false
    @inbounds for i in 1:nnodes(c)
        inside = xor(inside, _ray_crosses_segment(pt, c.nodes[i], next_node(c, i)))
    end
    return inside
end

function _point_in_closed_contour(pt::SVector{2,T}, c::PVContour{T}, domain::PeriodicDomain{T}) where {T}
    is_spanning(c) && return false
    pt_q = _wrap_query_pt(pt, domain)
    inside = false
    @inbounds for i in 1:nnodes(c)
        a_img, b_img = _shift_segment_to_image(c.nodes[i], next_node(c, i), pt_q, domain)
        inside = xor(inside, _ray_crosses_segment(pt_q, a_img, b_img))
    end
    return inside
end

function _segment_interior_probe(c::PVContour{T}, i::Int, δ, domain::AbstractDomain=UnboundedDomain()) where {T}
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
    # Include the canonical pair in the ordering so exactly tied distances are
    # deterministic across CPU spatial-index and device-compaction traversal.
    sort!(ranked)

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
