# ── Surgery Parameters ───────────────────────────────────

"""
    SurgeryParams{T}(δ, μ, Δ_max, area_min, n_surgery)

Parameters controlling contour surgery.

- `δ`: proximity threshold for detecting close contour segments.
- `μ`: minimum target segment length for remeshing.
- `Δ_max`: maximum target segment length for remeshing.
- `area_min`: minimum enclosed area; contours smaller than this are removed.
- `n_surgery`: number of time-steps between surgery passes.

`δ`, `μ`, and `Δ_max` use the same length units as the contour coordinates.
The adapted Dritschel density uses the dimensionless ratio `μ/L`, where `L`
is the estimated large-scale contour length; `μ` itself is not the
dimensionless node-density parameter of the literature. Spacing bounds apply
to the redistribution targets; cubic interpolation and area correction can
change the final chord lengths.
"""
struct SurgeryParams{T<:AbstractFloat}
    δ::T
    μ::T
    Δ_max::T
    area_min::T
    n_surgery::Int
    function SurgeryParams(δ::T, μ::T, Δ_max::T, area_min::T, n_surgery::Int) where {T<:AbstractFloat}
        isfinite(δ) && δ > zero(T) || throw(ArgumentError("δ must be finite and positive"))
        isfinite(μ) && μ > zero(T) || throw(ArgumentError("μ must be finite and positive"))
        isfinite(Δ_max) && Δ_max > μ || throw(ArgumentError("Δ_max must be finite and greater than μ"))

        isfinite(area_min) && area_min > zero(T) || throw(ArgumentError("area_min must be finite and positive"))
        n_surgery > 0 || throw(ArgumentError("n_surgery must be positive"))
        
        if δ > μ / 4
            @warn "SurgeryParams: δ ($δ) > μ/4 = $(μ/4); δ should usually be <= μ/4 for Dritschel surgery" maxlog=1
        end
        
        new{T}(δ, μ, Δ_max, area_min, n_surgery)
    end
end

function Base.getproperty(p::SurgeryParams, name::Symbol)
    name === :delta && return getfield(p, :δ)
    name === :mu && return getfield(p, :μ)
    name === :Delta_max && return getfield(p, :Δ_max)
    return getfield(p, name)
end

function Base.propertynames(::SurgeryParams; private::Bool=false)
    return (:δ, :μ, :Δ_max, :area_min, :n_surgery, :delta, :mu, :Delta_max)
end
