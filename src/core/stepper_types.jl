# ── Time Steppers ────────────────────────────────────────

"""
    AbstractTimeStepper

Supertype for built-in time integration schemes. Steppers own reusable work
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

function RK4Stepper(dt::T, n::Int; dev::AbstractDevice=CPU()) where {T<:AbstractFloat}
    k1 = device_zeros(dev, SVector{2,T}, n)
    A = typeof(k1)
    RK4Stepper(dt, k1,
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               device_zeros(dev, SVector{2,T}, n),
               A[], Vector{Vector{UnitRange{Int}}}())
end

"""
    LeapfrogStepper{T,A}(dt, n; dev=CPU(), ra_coeff=0.05)

Leapfrog (second-order centred) time stepper with step size `dt`.
The first step is bootstrapped with an RK2 midpoint half-step.
Buffers are allocated on the given `dev`ice.
"""
mutable struct LeapfrogStepper{T<:AbstractFloat, A<:AbstractVector{SVector{2,T}}} <: AbstractTimeStepper
    dt::T
    nodes_prev::A
    vel_buf::A
    nodes_buf::A
    vel_mid::A
    initialized::Bool
    ra_coeff::T
    vel_bufs::Vector{A}
    node_ranges::Vector{Vector{UnitRange{Int}}}
end

function LeapfrogStepper(dt::T, n::Int; dev::AbstractDevice=CPU(), ra_coeff::Real=0.05) where {T<:AbstractFloat}
    # `nodes_prev` stores the filtered previous level; `initialized` records
    # whether the midpoint bootstrap has already supplied that history.
    nodes_prev = device_zeros(dev, SVector{2,T}, n)
    A = typeof(nodes_prev)
    LeapfrogStepper(dt, nodes_prev,
                    device_zeros(dev, SVector{2,T}, n),
                    device_zeros(dev, SVector{2,T}, n),
                    device_zeros(dev, SVector{2,T}, n),
                    false, T(ra_coeff),
                    A[], Vector{Vector{UnitRange{Int}}}())
end
