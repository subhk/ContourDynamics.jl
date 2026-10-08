using ContourDynamics
using StaticArrays

# circular_patch and elliptical_patch are now exported from ContourDynamics.
# Only rotated_elliptical_patch remains here (not exported, test-only helper).

"""Create a rotated elliptical PVContour with N nodes, semi-axes a and b, rotated by angle θ."""
function rotated_elliptical_patch(a::T, b::T, N::Int, pv::T, θ::T) where {T<:AbstractFloat}
    cosθ, sinθ = cos(θ), sin(θ)
    nodes = [SVector{2,T}(a * cos(2π * i / N) * cosθ - b * sin(2π * i / N) * sinθ,
                           a * cos(2π * i / N) * sinθ + b * sin(2π * i / N) * cosθ) for i in 0:(N-1)]
    return PVContour(nodes, pv)
end

"""Axis-aligned rectangle PVContour with `nside` nodes per side, counter-clockwise from (xmin, ymin)."""
function rectangle_patch(xmin, xmax, ymin, ymax, nside, pv)
    nodes = SVector{2,Float64}[]
    for k in 0:(nside - 1)
        push!(nodes, SVector(xmin + (xmax - xmin) * k / nside, ymin))
    end
    for k in 0:(nside - 1)
        push!(nodes, SVector(xmax, ymin + (ymax - ymin) * k / nside))
    end
    for k in 0:(nside - 1)
        push!(nodes, SVector(xmax - (xmax - xmin) * k / nside, ymax))
    end
    for k in 0:(nside - 1)
        push!(nodes, SVector(xmin, ymax - (ymax - ymin) * k / nside))
    end
    return PVContour(nodes, pv)
end
