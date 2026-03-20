"""
    vortex_area(c::PVContour)

Signed area enclosed by contour `c` using the shoelace formula.
"""
function vortex_area(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    A = zero(T)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        A += nodes[i][1] * nodes[j][2] - nodes[j][1] * nodes[i][2]
    end
    return A / 2
end

"""
    centroid(c::PVContour)

Centroid of the region enclosed by contour `c`, via Green's theorem.
"""
function centroid(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(SVector{2, T})
    A = vortex_area(c)
    abs(A) < eps(T) && return zero(SVector{2, T})

    cx = zero(T)
    cy = zero(T)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        cross = nodes[i][1] * nodes[j][2] - nodes[j][1] * nodes[i][2]
        cx += (nodes[i][1] + nodes[j][1]) * cross
        cy += (nodes[i][2] + nodes[j][2]) * cross
    end
    inv6A = one(T) / (6 * A)
    return SVector{2, T}(cx * inv6A, cy * inv6A)
end

"""
    ellipse_moments(c::PVContour)

Second moments → (aspect_ratio, orientation_angle).
"""
function ellipse_moments(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    A = vortex_area(c)
    ctr = centroid(c)

    Jxx = zero(T)
    Jyy = zero(T)
    Jxy = zero(T)

    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        xi, yi = nodes[i][1] - ctr[1], nodes[i][2] - ctr[2]
        xj, yj = nodes[j][1] - ctr[1], nodes[j][2] - ctr[2]
        cross = xi * yj - xj * yi
        Jxx += (xi^2 + xi * xj + xj^2) * cross
        Jyy += (yi^2 + yi * yj + yj^2) * cross
        Jxy += (xi * yj + 2 * xi * yi + 2 * xj * yj + xj * yi) * cross
    end

    Jxx /= 12
    Jyy /= 12
    Jxy /= 24

    Jxx /= abs(A)
    Jyy /= abs(A)
    Jxy /= abs(A)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy^2
    disc = sqrt(max(zero(T), trace^2 / 4 - det))
    lambda1 = trace / 2 + disc
    lambda2 = trace / 2 - disc

    aspect_ratio = sqrt(max(one(T), lambda1 / max(lambda2, eps(T))))
    angle = T(0.5) * atan(2 * Jxy, Jxx - Jyy)

    return (aspect_ratio, angle)
end
