# ── Time Steppers ────────────────────────────────────────

"""
    AbstractTimeStepper

Supertype for time steppers. Steppers own reusable work
buffers sized by [`total_nodes`](@ref) and are advanced with [`timestep!`](@ref)
or [`evolve!`](@ref).
"""
abstract type AbstractTimeStepper end

"""
    RK4Stepper{T,A}(dt, n; dev=CPU())

Classical fourth-order Runge–Kutta time stepper with step size `dt`.
Allocates internal buffers for `n` nodes on the given `dev`ice.
"""
struct RK4Stepper{T<:AbstractFloat, A<:AbstractVector{SVector{2,T}}} <: AbstractTimeStepper
    dt::T
    k1::A
    k2::A
    k3::A
    k4::A
    nodes_buf::A
    vel_bufs::Vector{A}
    node_ranges::Vector{Vector{UnitRange{Int}}}
end

"""
    LeapfrogStepper(args...; kwargs...)

Removed in ContourDynamics v1.0.21. Calling this stub raises an
`ArgumentError` with migration guidance; use [`RK4Stepper`](@ref) instead.
"""
function LeapfrogStepper(args...; kwargs...)
    throw(ArgumentError(
        "LeapfrogStepper (the Robert-Asselin filtered leapfrog scheme and its " *
        "ra_coeff parameter) was removed in ContourDynamics v1.0.21. Use " *
        "RK4Stepper(dt, n; dev) — or stepper=:RK4 with the Problem keyword " *
        "factory — instead."))
end

function RK4Stepper(dt::T, n::Int; dev::AbstractDevice=CPU()) where {T<:AbstractFloat}
    isfinite(dt) && dt > zero(T) || throw(ArgumentError(
        "RK4Stepper requires finite dt > 0, got dt = $dt"))
    n >= 0 || throw(ArgumentError("RK4Stepper requires n >= 0, got n = $n"))
    k1 = device_zeros(dev, SVector{2,T}, n)
    A = typeof(k1)
    RK4Stepper(dt, k1,
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               A[], Vector{Vector{UnitRange{Int}}}())
end
