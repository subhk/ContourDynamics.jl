# Scalar contact and surgery predicates used by both backends.

# PV jumps and interior levels carry units. An absolute epsilon floor would
# identify distinct weak levels (even opposite signs) as the same fluid.
@inline function _same_surgery_pv(a::T, b::T) where {T}
    return a == b || abs(a - b) <= sqrt(eps(T)) * max(abs(a), abs(b))
end

@inline _flat_point_segment_dist2(px, py, ax, ay, bx, by) =
    first(_flat_point_segment_closest(px, py, ax, ay, bx, by))

@inline function _flat_point_segment_closest(px, py, ax, ay, bx, by)
    sx = bx - ax
    sy = by - ay
    len2 = sx * sx + sy * sy
    if len2 <= eps(typeof(len2))
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy, ax, ay
    end
    t = ((px - ax) * sx + (py - ay) * sy) / len2
    t = min(max(t, zero(t)), one(t))
    cx = ax + t * sx
    cy = ay + t * sy
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy, cx, cy
end

@inline function _flat_surgery_contact_distance2(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2)
    d1 = _flat_point_segment_dist2(ax1, ay1, ax2, ay2, bx2, by2)
    d2 = _flat_point_segment_dist2(bx1, by1, ax2, ay2, bx2, by2)
    d3 = _flat_point_segment_dist2(ax2, ay2, ax1, ay1, bx1, by1)
    d4 = _flat_point_segment_dist2(bx2, by2, ax1, ay1, bx1, by1)
    return min(min(d1, d2), min(d3, d4))
end

@inline function _flat_ray_crosses_segment(px, py, ax, ay, bx, by)
    (ay > py) == (by > py) && return false
    x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
    return px < x_cross
end

@inline function _point_in_polygon(px, py, n::Int, getnode::F) where {F}
    n < 3 && return false
    inside = false
    @inbounds for i in 1:n
        ax, ay = getnode(i)
        bx, by = getnode(i < n ? i + 1 : 1)
        inside = xor(inside, _flat_ray_crosses_segment(px, py, ax, ay, bx, by))
    end
    return inside
end

@inline function _periodic_point_in_polygon(px::T, py::T, n::Int, getnode::F,
                                            Lx::T, Ly::T) where {T,F}
    n < 3 && return false
    xmin, ymin = getnode(1)
    xmax, ymax = xmin, ymin
    @inbounds for i in 2:n
        x, y = getnode(i)
        xmin, xmax = min(xmin, x), max(xmax, x)
        ymin, ymax = min(ymin, y), max(ymax, y)
    end

    # Translate the query by whole periods, keeping the polygon closed in its
    # original coordinate frame. Only images inside its bounding box can be
    # contained. Enumerating them also supports contours wider than a half
    # period, for which choosing one nearest image is insufficient.
    period_x, period_y = T(2) * Lx, T(2) * Ly
    ixlo = ceil(Int, (xmin - px) / period_x)
    ixhi = floor(Int, (xmax - px) / period_x)
    iylo = ceil(Int, (ymin - py) / period_y)
    iyhi = floor(Int, (ymax - py) / period_y)
    for ix in ixlo:ixhi, iy in iylo:iyhi
        qx, qy = px + T(ix) * period_x, py + T(iy) * period_y
        _point_in_polygon(qx, qy, n, getnode) && return true
    end
    return false
end
