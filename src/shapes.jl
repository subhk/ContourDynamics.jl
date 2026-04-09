# Convenience constructors for common contour shapes.

"""
    circular_patch(R, N, pv; cx=0.0, cy=0.0, T=Float64)

Create a circular [`PVContour`](@ref) with radius `R`, `N` nodes, and PV jump `pv`,
centered at `(cx, cy)`.
"""
function circular_patch(R, N::Int, pv; cx=0.0, cy=0.0, T::Type{<:AbstractFloat}=Float64)
    R, pv, cx, cy = T(R), T(pv), T(cx), T(cy)
    N >= 3 || throw(ArgumentError("N must be >= 3, got $N"))
    R > zero(T) || throw(ArgumentError("R must be positive, got $R"))
    nodes = [SVector{2,T}(cx + R * cos(2T(π) * T(i) / T(N)),
                          cy + R * sin(2T(π) * T(i) / T(N))) for i in 0:(N-1)]
    PVContour(nodes, pv)
end

"""
    elliptical_patch(a, b, N, pv; cx=0.0, cy=0.0, θ=0.0, T=Float64)

Create an elliptical [`PVContour`](@ref) with semi-axes `a` and `b`, `N` nodes,
PV jump `pv`, centered at `(cx, cy)`, rotated by angle `θ` (radians).
"""
function elliptical_patch(a, b, N::Int, pv; cx=0.0, cy=0.0, θ=0.0, T::Type{<:AbstractFloat}=Float64)
    a, b, pv, cx, cy, θ = T(a), T(b), T(pv), T(cx), T(cy), T(θ)
    N >= 3 || throw(ArgumentError("N must be >= 3, got $N"))
    a > zero(T) || throw(ArgumentError("semi-axis a must be positive, got $a"))
    b > zero(T) || throw(ArgumentError("semi-axis b must be positive, got $b"))
    cosθ, sinθ = cos(θ), sin(θ)
    nodes = [SVector{2,T}(cx + a * cos(2T(π) * T(i) / T(N)) * cosθ - b * sin(2T(π) * T(i) / T(N)) * sinθ,
                          cy + a * cos(2T(π) * T(i) / T(N)) * sinθ + b * sin(2T(π) * T(i) / T(N)) * cosθ) for i in 0:(N-1)]
    PVContour(nodes, pv)
end

"""
    rankine_vortex(R, N, Γ; cx=0.0, cy=0.0, T=Float64)

Create a Rankine vortex (uniform vorticity patch) with radius `R`, `N` nodes, and
total circulation `Γ`, centered at `(cx, cy)`. Returns a single-element
`Vector{PVContour{T}}` with `pv = Γ / (π R²)`.
"""
function rankine_vortex(R, N::Int, Γ; cx=0.0, cy=0.0, T::Type{<:AbstractFloat}=Float64)
    R, Γ = T(R), T(Γ)
    N >= 3 || throw(ArgumentError("N must be >= 3, got $N"))
    R > zero(T) || throw(ArgumentError("R must be positive, got $R"))
    pv = Γ / (T(π) * R^2)
    [circular_patch(R, N, pv; cx=cx, cy=cy, T=T)]
end
