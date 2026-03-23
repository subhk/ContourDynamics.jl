using ContourDynamics
using StaticArrays

"""Create a circular PVContour with N nodes, radius R, centered at origin."""
function circular_patch(R::T, N::Int, pv::T) where {T<:AbstractFloat}
    nodes = [SVector{2,T}(R * cos(2π * i / N), R * sin(2π * i / N)) for i in 0:(N-1)]
    return PVContour(nodes, pv)
end

"""Create an elliptical PVContour with N nodes, semi-axes a and b, centered at origin."""
function elliptical_patch(a::T, b::T, N::Int, pv::T) where {T<:AbstractFloat}
    nodes = [SVector{2,T}(a * cos(2π * i / N), b * sin(2π * i / N)) for i in 0:(N-1)]
    return PVContour(nodes, pv)
end

"""Create a rotated elliptical PVContour with N nodes, semi-axes a and b, rotated by angle θ."""
function rotated_elliptical_patch(a::T, b::T, N::Int, pv::T, θ::T) where {T<:AbstractFloat}
    cosθ, sinθ = cos(θ), sin(θ)
    nodes = [SVector{2,T}(a * cos(2π * i / N) * cosθ - b * sin(2π * i / N) * sinθ,
                           a * cos(2π * i / N) * sinθ + b * sin(2π * i / N) * cosθ) for i in 0:(N-1)]
    return PVContour(nodes, pv)
end
