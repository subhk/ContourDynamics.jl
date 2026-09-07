# geometry/contacts.jl — CPU surgery stage.

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
    distance2, cx, cy = _flat_point_segment_closest(
        p_img[1], p_img[2], a_img[1], a_img[2], b_img[1], b_img[2])
    return distance2, SVector{2,T}(cx, cy) - shift
end

function _point_segment_dist2(p::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return _flat_point_segment_dist2(p[1], p[2], a[1], a[2], b[1], b[2])
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
