# geometry/interpolation.jl — geometry and remeshing stage helpers.

function _signed_path_curvatures(path::Vector{SVector{2,T}}, corners::AbstractVector{Bool}) where {T}
    n = length(path)
    κ = zeros(T, n)
    n < 3 && return κ

    @inbounds for i in 2:(n - 1)
        if corners[i - 1] || corners[i] || corners[i + 1]
            continue
        end

        a = path[i] - path[i - 1]
        b = path[i + 1] - path[i]
        chord = path[i + 1] - path[i - 1]
        κ[i] = _local_signed_curvature(a[1], a[2], b[1], b[2], chord[1], chord[2])
    end
    return κ
end

function _cubic_segment_point(a::SVector{2,T}, b::SVector{2,T},
                              κa::T, κb::T, p::T) where {T}
    _norm2(b - a) <= eps(T) && return a
    sx, sy, tx, ty = _cubic_point_tangent_scalar(a[1], a[2], b[1], b[2], κa, κb, p)
    return SVector{2,T}(sx, sy)
end

function _cubic_segment_tangent(a::SVector{2,T}, b::SVector{2,T},
                                κa::T, κb::T, p::T) where {T}
    _norm2(b - a) <= eps(T) && return zero(SVector{2,T})
    sx, sy, tx, ty = _cubic_point_tangent_scalar(a[1], a[2], b[1], b[2], κa, κb, p)
    return SVector{2,T}(tx, ty)
end

function _cubic_segment_point(c::PVContour{T}, i::Int, p::T, κ::Vector{T}) where {T}
    return _cubic_segment_point(c.nodes[i], next_node(c, i), κ[i], κ[mod1(i + 1, nnodes(c))], p)
end
