# geometry/polygons.jl — geometry and remeshing stage helpers.

function _raw_polygon_area(nodes::AbstractVector{SVector{2,T}},
                           wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    n = length(nodes)
    n < 3 && return zero(T)
    origin = nodes[1]
    area = zero(T)
    @inbounds for i in 1:n
        point = nodes[i] - origin
        nxt = (i < n ? nodes[i + 1] : nodes[1] + wrap) - origin
        area += point[1] * nxt[2] - nxt[1] * point[2]
    end
    return area / 2
end

function _raw_polygon_scale2(nodes::AbstractVector{SVector{2,T}},
                             wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    isempty(nodes) && return zero(T)
    origin = nodes[1]
    scale = max(abs(wrap[1]), abs(wrap[2]))
    @inbounds for point in nodes
        relative = point - origin
        scale = max(scale, abs(relative[1]), abs(relative[2]))
    end
    return scale * scale
end

@inline function _raw_polygon_area_tolerance(nodes::AbstractVector{SVector{2,T}},
                                             wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    return eps(T) * T(max(length(nodes), 1)) * _raw_polygon_scale2(nodes, wrap)
end

function _raw_polygon_mean(nodes::AbstractVector{SVector{2,T}}) where {T}
    isempty(nodes) && return zero(SVector{2,T})
    origin = nodes[1]
    relative_sum = zero(SVector{2,T})
    @inbounds for point in nodes
        relative_sum += point - origin
    end
    return origin + relative_sum / length(nodes)
end

function _raw_polygon_centroid(nodes::AbstractVector{SVector{2,T}},
                               wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    n = length(nodes)
    n == 0 && return zero(SVector{2,T})
    area = _raw_polygon_area(nodes, wrap)
    if abs(area) <= _raw_polygon_area_tolerance(nodes, wrap)
        return _raw_polygon_mean(nodes)
    end

    origin = nodes[1]
    relative_moment = zero(SVector{2,T})
    @inbounds for i in 1:n
        point = nodes[i] - origin
        nxt = (i < n ? nodes[i + 1] : nodes[1] + wrap) - origin
        cross = point[1] * nxt[2] - nxt[1] * point[2]
        relative_moment += (point + nxt) * cross
    end
    inv6A = one(T) / (6 * area)
    return origin + relative_moment * inv6A
end

function _preserve_closed_area!(nodes::Vector{SVector{2,T}}, target_area::T) where {T}
    # Remeshing changes node locations slightly. For closed contours, apply a
    # uniform centroid-centered rescale so the signed polygon area is preserved.
    new_area = _raw_polygon_area(nodes)
    area_tolerance = _raw_polygon_area_tolerance(nodes)
    (abs(target_area) <= area_tolerance || abs(new_area) <= area_tolerance) && return nodes
    sign(target_area) == sign(new_area) || return nodes

    scale = sqrt(abs(target_area / new_area))
    abs(scale - one(T)) <= sqrt(eps(T)) && return nodes
    ctr = _raw_polygon_centroid(nodes)
    @inbounds for i in eachindex(nodes)
        nodes[i] = ctr + scale * (nodes[i] - ctr)
    end
    return nodes
end

@inline _cross2(a::SVector{2,T}, b::SVector{2,T}) where {T} = a[1] * b[2] - a[2] * b[1]
@inline _norm2(a::SVector{2,T}) where {T} = sqrt(a[1]^2 + a[2]^2)

# Smallest-magnitude real root of a t² + b t + c = 0, or `nothing` when there is
# no usable root. Uses the numerically stable form (compute the large root via
# the standard formula, the small one via c/q) so the near-zero root we want is
# not lost to cancellation when c is small.
@inline function _smallest_quadratic_root(a::T, b::T, c::T) where {T}
    coefficient_scale = max(abs(a), abs(b), abs(c))
    iszero(coefficient_scale) && return nothing
    a /= coefficient_scale
    b /= coefficient_scale
    c /= coefficient_scale
    if abs(a) <= eps(T)
        abs(b) <= eps(T) && return nothing
        return -c / b
    end
    disc = b * b - 4 * a * c
    disc < 0 && return nothing
    sd = sqrt(disc)
    q = -(b + (b >= 0 ? sd : -sd)) / 2
    r1 = q / a
    abs(q) <= eps(T) && return r1
    r2 = c / q
    return abs(r1) <= abs(r2) ? r1 : r2
end

function _preserve_closed_area_fixed_corners!(nodes::Vector{SVector{2,T}},
                                              corners::AbstractVector{Bool},
                                              target_area::T) where {T}
    # Area-preserving rescale for contours with fixed surgery corners. The
    # corner-free path scales every node about the centroid, but that moves the
    # corners; here the corners must stay exactly put (they are reconnection
    # break points). So only the free (non-corner) nodes move, along
    # d_i = (p_i - centroid). The signed area is a quadratic A0 + B·t + C·t² in
    # the scalar step `t`; solve for the root nearest zero and apply it. Corners
    # keep d_i = 0 and therefore do not move at all.
    n = length(nodes)
    n < 3 && return nodes
    A0 = _raw_polygon_area(nodes)
    area_tolerance = _raw_polygon_area_tolerance(nodes)
    abs(target_area) <= area_tolerance && return nodes
    abs(A0) <= area_tolerance && return nodes
    sign(target_area) == sign(A0) || return nodes

    rhs = target_area - A0
    abs(rhs) <= sqrt(eps(T)) * abs(target_area) && return nodes  # already conserved

    ctr = _raw_polygon_centroid(nodes)
    B = zero(T)
    C = zero(T)
    @inbounds for i in 1:n
        j = i < n ? i + 1 : 1
        pi = nodes[i]
        pj = nodes[j]
        di = corners[i] ? zero(SVector{2,T}) : pi - ctr
        dj = corners[j] ? zero(SVector{2,T}) : pj - ctr
        B += _cross2(pi - ctr, dj) + _cross2(di, pj - ctr)
        C += _cross2(di, dj)
    end
    B /= 2
    C /= 2

    t = _smallest_quadratic_root(C, B, -rhs)
    # Skip when the correction is ill-conditioned (no real root, or so large it
    # would distort the geometry); a small unmeasured drift is safer than that.
    (t === nothing || !isfinite(t) || abs(t) > T(1) / 2) && return nodes

    @inbounds for i in 1:n
        corners[i] && continue
        nodes[i] = nodes[i] + t * (nodes[i] - ctr)
    end
    return nodes
end

"""
    arc_lengths(c::PVContour)

Return a vector of segment lengths for each consecutive node pair in contour `c`.
"""
function arc_lengths(c::PVContour{T}) where {T}
    n = nnodes(c)
    n < 1 && return T[]
    lengths = Vector{T}(undef, n)
    return _arc_lengths!(lengths, c)
end

function _arc_lengths!(lengths::Vector{T}, c::PVContour{T}) where {T}
    n = nnodes(c)
    resize!(lengths, n)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        lengths[i] = sqrt(d[1]^2 + d[2]^2)
    end
    return lengths
end
