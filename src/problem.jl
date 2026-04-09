# High-level Problem wrapper — bundles ContourProblem + stepper + surgery params
# into a single object for GeophysicalFlows-style convenience.

"""
    Problem{P,S,SP}

Convenience wrapper bundling a [`ContourProblem`](@ref) (or
[`MultiLayerContourProblem`](@ref)), a time stepper, and optional
[`SurgeryParams`](@ref) into a single object.

Construct via the keyword factory [`Problem(; kwargs...)`](@ref) or directly:

    Problem(contour_problem, stepper, surgery_params)
"""
struct Problem{P<:Union{ContourProblem, MultiLayerContourProblem},
               S<:AbstractTimeStepper,
               SP<:Union{SurgeryParams, Nothing}}
    contour_problem::P
    stepper::S
    surgery_params::SP
end

# ── Forwarded accessors ─────────────────────────────────

"""Return the contours of the underlying problem."""
contours(prob::Problem) = prob.contour_problem.contours

"""Return the kernel of the underlying problem."""
kernel(prob::Problem) = prob.contour_problem.kernel

"""Return the domain of the underlying problem."""
domain(prob::Problem) = prob.contour_problem.domain

total_nodes(prob::Problem) = total_nodes(prob.contour_problem)
energy(prob::Problem) = energy(prob.contour_problem)
circulation(prob::Problem) = circulation(prob.contour_problem)
enstrophy(prob::Problem) = enstrophy(prob.contour_problem)
angular_momentum(prob::Problem) = angular_momentum(prob.contour_problem)
velocity!(vel, prob::Problem) = velocity!(vel, prob.contour_problem)
velocity(prob::Problem, x) = velocity(prob.contour_problem, x)
vortex_area(prob::Problem) = vortex_area.(contours(prob))
nlayers(prob::Problem) = nlayers(prob.contour_problem)

# ── evolve! overload ────────────────────────────────────

"""
    evolve!(prob::Problem; nsteps, callbacks=nothing)

Run the simulation for `nsteps` time steps. Surgery is applied according to
`prob.surgery_params` (or skipped if `nothing`).
"""
function evolve!(prob::Problem; nsteps::Int, callbacks=nothing)
    evolve!(prob.contour_problem, prob.stepper, prob.surgery_params;
            nsteps, callbacks)
    return prob
end

# ── Surgery presets ─────────────────────────────────────

const _SURGERY_PRESETS = Dict{Symbol, NamedTuple}(
    :standard     => (delta=0.005, mu=0.02,  Delta_max=0.2,  area_min=1e-6,  n_surgery=5),
    :conservative => (delta=0.002, mu=0.01,  Delta_max=0.15, area_min=1e-8,  n_surgery=10),
    :aggressive   => (delta=0.01,  mu=0.04,  Delta_max=0.3,  area_min=1e-4,  n_surgery=3),
)

function _build_surgery(surgery, ::Type{T}) where {T}
    surgery isa SurgeryParams && return surgery
    surgery === :none && return nothing
    if surgery isa Symbol
        haskey(_SURGERY_PRESETS, surgery) || throw(ArgumentError(
            "Unknown surgery preset :$surgery. Use :standard, :conservative, :aggressive, :none, or a SurgeryParams."))
        p = _SURGERY_PRESETS[surgery]
        return SurgeryParams(T(p.delta), T(p.mu), T(p.Delta_max), T(p.area_min), p.n_surgery)
    end
    throw(ArgumentError("surgery must be a Symbol preset or SurgeryParams, got $(typeof(surgery))"))
end

# ── Factory constructor ─────────────────────────────────

"""
    Problem(; contours, dt, kernel=:euler, domain=:unbounded, stepper=:RK4, surgery=:standard, dev=:cpu, T=Float64, ...)

Convenience constructor that builds a [`ContourProblem`](@ref),
time stepper, and [`SurgeryParams`](@ref) from keyword arguments.

# Required
- `contours`: `Vector{PVContour}` — the initial vortex boundaries (single-layer)
- `dt`: time step size

# Optional — Kernel
- `kernel=:euler`: `:euler`, `:qg`, `:sqg`, or `:multilayer_qg`
- `Ld`: deformation radius (required for `:qg`; `SVector` for `:multilayer_qg`)
- `delta_sqg`: regularization length (required for `:sqg`)
- `coupling`: layer coupling matrix (required for `:multilayer_qg`)

# Optional — Domain
- `domain=:unbounded`: `:unbounded` or `:periodic`
- `Lx`, `Ly`: domain half-widths (required for `:periodic`)

# Optional — Stepper
- `stepper=:RK4`: `:RK4` or `:leapfrog`
- `ra_coeff=0.05`: Robert-Asselin filter coefficient (`:leapfrog` only)

# Optional — Surgery
- `surgery=:standard`: `:standard`, `:conservative`, `:aggressive`, `:none`, or a `SurgeryParams`

# Optional — Device and type
- `dev=:cpu`: `:cpu` or `:gpu`
- `T=Float64`: floating-point type

# Multi-layer
Use `layers` (tuple of contour vectors) instead of `contours`, with
`kernel=:multilayer_qg`, `Ld` (SVector), and `coupling` (SMatrix).
"""
function Problem(;
    contours=nothing,
    dt::Real,
    layers=nothing,
    coupling=nothing,
    kernel::Symbol=:euler,
    Ld=nothing,
    delta_sqg::Real=NaN,
    domain::Symbol=:unbounded,
    Lx::Real=NaN,
    Ly::Real=NaN,
    stepper::Symbol=:RK4,
    ra_coeff::Real=0.05,
    surgery=:standard,
    dev::Symbol=:cpu,
    T::Type{<:AbstractFloat}=Float64,
)
    # ── Validate contours / layers ──
    _is_multilayer = kernel === :multilayer_qg
    if _is_multilayer
        layers !== nothing || throw(ArgumentError(
            "kernel=:multilayer_qg requires `layers` (tuple of contour vectors), not `contours`."))
        contours === nothing || throw(ArgumentError(
            "`contours` and `layers` are mutually exclusive. Use `layers` for multi-layer problems."))
    else
        contours !== nothing || throw(ArgumentError(
            "Required keyword `contours` not provided. Pass a Vector{PVContour}."))
        layers === nothing || throw(ArgumentError(
            "`layers` requires `kernel=:multilayer_qg`."))
    end

    # ── Build kernel ──
    k = if kernel === :euler
        EulerKernel()
    elseif kernel === :qg
        Ld === nothing && throw(ArgumentError("kernel=:qg requires `Ld` (deformation radius)."))
        QGKernel(T(Ld))
    elseif kernel === :sqg
        isnan(delta_sqg) && throw(ArgumentError("kernel=:sqg requires `delta_sqg` (regularization length)."))
        SQGKernel(T(delta_sqg))
    elseif kernel === :multilayer_qg
        Ld === nothing && throw(ArgumentError("kernel=:multilayer_qg requires `Ld` (SVector of deformation radii)."))
        coupling === nothing && throw(ArgumentError("kernel=:multilayer_qg requires `coupling` (SMatrix)."))
        MultiLayerQGKernel(T.(Ld), T.(coupling))
    else
        throw(ArgumentError("Unknown kernel :$kernel. Use :euler, :qg, :sqg, or :multilayer_qg."))
    end

    # ── Build domain ──
    d = if domain === :unbounded
        UnboundedDomain()
    elseif domain === :periodic
        (isnan(Lx) || isnan(Ly)) && throw(ArgumentError(
            "domain=:periodic requires `Lx` and `Ly` (domain half-widths)."))
        PeriodicDomain(T(Lx), T(Ly))
    else
        throw(ArgumentError("Unknown domain :$domain. Use :unbounded or :periodic."))
    end

    # ── Build device ──
    device = if dev === :cpu
        CPU()
    elseif dev === :gpu
        GPU()
    else
        throw(ArgumentError("Unknown device :$dev. Use :cpu or :gpu."))
    end

    # ── Build ContourProblem or MultiLayerContourProblem ──
    cp = if _is_multilayer
        MultiLayerContourProblem(k, d, layers; dev=device)
    else
        ContourProblem(k, d, contours; dev=device)
    end

    # ── Build stepper ──
    N = total_nodes(cp)
    s = if stepper === :RK4
        RK4Stepper(T(dt), N; dev=device)
    elseif stepper === :leapfrog
        LeapfrogStepper(T(dt), N; dev=device, ra_coeff=T(ra_coeff))
    else
        throw(ArgumentError("Unknown stepper :$stepper. Use :RK4 or :leapfrog."))
    end

    # ── Build surgery params ──
    sp = _build_surgery(surgery, T)

    return Problem(cp, s, sp)
end
