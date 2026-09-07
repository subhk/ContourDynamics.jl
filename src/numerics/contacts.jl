# Scalar geometric predicates used by both surgery backends.

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
