# ── Surgery Parameters ───────────────────────────────────

"""
    SurgeryParams{T}(δ, μ, Δ_max, area_min, n_surgery)

Parameters controlling contour surgery.

- `δ`: proximity threshold for detecting close contour segments.
- `μ`: minimum segment length after remeshing.
- `Δ_max`: maximum segment length after remeshing.
- `area_min`: minimum enclosed area; contours smaller than this are removed.
- `n_surgery`: number of time-steps between surgery passes.

!!! note "Leapfrog stepper interaction"
    Surgery may change the number of nodes (via reconnection, filament removal,
    and remeshing), which invalidates the previous-step history required by
    [`LeapfrogStepper`](@ref).  After each surgery pass the leapfrog method
    re-bootstraps with an RK2 midpoint half-step, maintaining second-order
    accuracy.  If high-order accuracy is important, prefer [`RK4Stepper`](@ref)
    or increase `n_surgery` to reduce the frequency of re-bootstrapping.
"""
struct SurgeryParams{T<:AbstractFloat}
    δ::T
    μ::T
    Δ_max::T
    area_min::T
    n_surgery::Int
    function SurgeryParams(δ::T, μ::T, Δ_max::T, area_min::T, n_surgery::Int) where {T<:AbstractFloat}
        δ > zero(T) || throw(ArgumentError("δ must be positive"))
        μ > zero(T) || throw(ArgumentError("μ must be positive"))
        Δ_max > μ || throw(ArgumentError("Δ_max must be greater than μ"))
        area_min > zero(T) || throw(ArgumentError("area_min must be positive"))
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
