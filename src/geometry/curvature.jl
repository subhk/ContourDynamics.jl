# geometry/curvature.jl — geometry and remeshing stage helpers.

# Normalize local differences before forming the cubic-length denominator.
# The degeneracy threshold is dimensionless, so changing coordinate units does
# not suppress curvature. Shared by contour, fixed-corner path, and KA code.
@inline function _local_signed_curvature(ax::T, ay::T, bx::T, by::T,
                                         cx::T, cy::T) where {T}
    scale = max(abs(ax), abs(ay), abs(bx), abs(by), abs(cx), abs(cy))
    iszero(scale) && return zero(T)
    ax /= scale; ay /= scale
    bx /= scale; by /= scale
    cx /= scale; cy /= scale
    denom = sqrt(ax * ax + ay * ay) * sqrt(bx * bx + by * by) *
            sqrt(cx * cx + cy * cy)
    denom <= eps(T) && return zero(T)
    return (T(2) * (ax * by - ay * bx) / denom) / scale
end

@inline function _signed_node_curvature(c::PVContour{T}, i::Int) where {T}
    n = nnodes(c)
    n < 3 && return zero(T)
    @boundscheck (1 <= i <= n || throw(BoundsError(c.nodes, i)))

    @inbounds begin
        prev_i = mod1(i - 1, n)
        next_i = mod1(i + 1, n)
        if c.corners[prev_i] || c.corners[i] || c.corners[next_i]
            return zero(T)
        end

        prev = i == 1 ? c.nodes[n] - c.wrap : c.nodes[i - 1]
        curr = c.nodes[i]
        nxt = next_node(c, i)
        a = curr - prev
        b = nxt - curr
        chord = nxt - prev
        return _local_signed_curvature(a[1], a[2], b[1], b[2], chord[1], chord[2])
    end
end

function _signed_node_curvatures!(κ::AbstractVector{T}, c::PVContour{T}) where {T}
    # Curvature is signed so cubic interpolation can bend to the correct side of
    # each segment. Surgery corners deliberately suppress curvature through the
    # adjacent nodes to keep fixed break points sharp.
    n = nnodes(c)
    length(κ) >= n || throw(DimensionMismatch("curvature buffer length ($(length(κ))) must be >= node count ($n)"))
    @inbounds for i in 1:n
        κ[i] = _signed_node_curvature(c, i)
    end
    return κ
end

function _signed_node_curvatures(c::PVContour{T}) where {T}
    κ = zeros(T, nnodes(c))
    return _signed_node_curvatures!(κ, c)
end

function _node_curvatures(c::PVContour{T}) where {T}
    κ = _signed_node_curvatures(c)
    @inbounds for i in eachindex(κ)
        κ[i] = abs(κ[i])
    end
    return κ
end
