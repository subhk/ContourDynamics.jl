# Public package module.
#
# File organization follows the simulation pipeline:
#   1. core types and problem construction
#   2. velocity kernels and acceleration paths
#   3. diagnostics
#   4. time integration and user helpers
#
# Package extensions add optional I/O, plotting, ODE, GPU, and recorder methods
# by implementing the empty function stubs declared near the exports below.
module ContourDynamics

using StaticArrays
using LinearAlgebra
using SpecialFunctions
using PrecompileTools: @setup_workload, @compile_workload

# Core object model and user-facing problem construction.
include("core/device.jl")
include("core/types.jl")
include("core/contours.jl")
include("core/domains.jl")
include("core/problem.jl")
include("core/surgery.jl")

# Velocity evaluation and accelerators.
include("velocity/common.jl")
include("velocity/unbounded/single_layer.jl")
include("velocity/periodic/cache.jl")
include("velocity/periodic/single_layer.jl")
include("accel/ka/packing.jl")
include("accel/ka/kernels.jl")
include("accel/ka/velocity.jl")
include("accel/ka/surgery/types.jl")
include("accel/ka/surgery/filaments.jl")
include("accel/ka/surgery/pairs.jl")
include("accel/ka/surgery/rewrite.jl")
include("accel/ka/surgery/remesh.jl")
include("accel/ka/surgery/driver.jl")

# Diagnostics.
include("diagnostics/geometry.jl")
include("diagnostics/ka_energy.jl")
include("diagnostics/unbounded/single_layer.jl")
include("diagnostics/unbounded/multilayer_qg.jl")
include("diagnostics/periodic/common.jl")
include("diagnostics/periodic/single_layer.jl")
include("diagnostics/periodic/multilayer_qg.jl")

# Time integration and presentation helpers.
include("core/evolution.jl")
include("beta_plane.jl")
include("core/show.jl")
include("core/shapes.jl")

# Public exports are grouped by concept so new API additions have an obvious
# home and users can scan the surface area without reading implementation files.
export AbstractDevice, CPU, GPU, device_array, device_zeros, to_cpu, to_device
export AbstractKernel, EulerKernel, QGKernel, BetaPlaneQGKernel, SQGKernel, MultiLayerQGKernel
export PVContour, nnodes, is_corner, corner_indices, is_spanning, next_node, beta_staircase
export DeviceContourState, materialize_contours
export AbstractDomain, UnboundedDomain, PeriodicDomain, wrap_nodes!
export ContourProblem, MultiLayerContourProblem
export SurgeryParams
export AbstractTimeStepper, RK4Stepper, LeapfrogStepper
export nlayers, total_nodes
export velocity!, velocity, segment_velocity
export vortex_area, centroid, ellipse_moments
export energy, enstrophy, circulation, angular_momentum
export remesh, arc_lengths, surgery!
export circular_patch, elliptical_patch, rankine_vortex
export Problem, contours, kernel, domain
export EwaldCache, build_ewald_cache, setup_ewald_cache!, clear_ewald_cache!
export clear_state_workspace_cache!
export timestep!, resize_buffers!, evolve!

# Extension stubs — implemented by package extensions when optional
# dependencies are loaded. Keeping these names here lets users call, e.g.,
# `save_snapshot` after `using JLD2` without ContourDynamics depending on JLD2.
"""
    flatten_nodes(prob::ContourProblem)

Return the node coordinates of `prob` as a flat vector `[x1, y1, x2, y2, ...]`
in contour order. This generic is implemented by the OrdinaryDiffEq extension
when `OrdinaryDiffEq` is loaded.
"""
function flatten_nodes end

"""
    unflatten_nodes!(prob::ContourProblem, u)

Write a flat coordinate vector back into `prob.contours`, using the ordering
created by [`flatten_nodes`](@ref). Implemented by the OrdinaryDiffEq extension.
"""
function unflatten_nodes! end

"""
    to_ode_problem(prob::ContourProblem, tspan; kwargs...)

Wrap a contour problem as an `ODEProblem` for OrdinaryDiffEq. The extension
returns either an `ODEProblem` or a `(ode_prob, callback)` tuple when contour
surgery is requested.
"""
function to_ode_problem end

"""
    record_evolution(prob, stepper, params; nsteps, frameskip=10, filename, callbacks=nothing)

Record an in-place Makie animation of a contour simulation. Implemented by the
Makie extension when `Makie` is loaded.
"""
function record_evolution end

"""
    recorded_diagnostics(prob; dt, nsteps, record_every=1)

Build RecordedArrays diagnostic recorders and a callback suitable for
[`evolve!`](@ref). Implemented by the RecordedArrays extension.
"""
function recorded_diagnostics end

"""
    save_snapshot(filename, prob, step; kwargs...)

Save a JLD2 snapshot of the current contour state. Implemented by the JLD2
extension when `JLD2` is loaded.
"""
function save_snapshot end

"""
    load_snapshot(filename, step)

Load one JLD2 snapshot written by [`save_snapshot`](@ref). Implemented by the
JLD2 extension.
"""
function load_snapshot end

"""
    jld2_recorder(filename; save_every=nothing, save_dt=nothing, dt=nothing, diagnostics=true)

Create an `evolve!` callback that periodically writes JLD2 snapshots.
Implemented by the JLD2 extension.
"""
function jld2_recorder end

"""
    load_simulation(filename)

Load all JLD2 snapshots from `filename`, sorted by simulation step. Implemented
by the JLD2 extension.
"""
function load_simulation end

"""
    load_problem(filename, step; dev=CPU())

Reconstruct a runnable [`ContourProblem`](@ref) from a JLD2 snapshot, using the
kernel/domain metadata saved by [`save_snapshot`](@ref). Supported for
single-layer `EulerKernel`, `QGKernel`, and `SQGKernel`. Stepper and surgery
state are not persisted, so the returned object is a `ContourProblem` (not a
[`Problem`](@ref)); recreate the stepper/`SurgeryParams` to continue a run.
Implemented by the JLD2 extension.
"""
function load_problem end

export flatten_nodes, unflatten_nodes!, to_ode_problem, record_evolution
export recorded_diagnostics
export save_snapshot, load_snapshot, jld2_recorder, load_simulation, load_problem

# ── Precompile workload ─────────────────────────────────────────────────
# Drives the hot paths — velocity, timestep, surgery, diagnostics — for
# the kernel/domain/stepper combinations used in tutorials, examples, and
# tests. Running once here caches the JIT for first-call sites elsewhere.
@setup_workload begin
    N = 16
    @compile_workload begin
        # Single-layer: Euler / Unbounded, RK4 + Leapfrog
        cE = circular_patch(0.5, N, 2π)
        probE = Problem(; contours=[cE], dt=0.01)
        evolve!(probE; nsteps=2)
        energy(probE); circulation(probE); enstrophy(probE)
        centroid(cE); ellipse_moments(cE)

        cE2 = circular_patch(0.5, N, 2π)
        probE_lf = Problem(; contours=[cE2], dt=0.01, stepper=:leapfrog)
        evolve!(probE_lf; nsteps=2)

        # QG / Unbounded
        cQ = circular_patch(0.5, N, 2π)
        probQ = Problem(; contours=[cQ], dt=0.01, kernel=:qg, Ld=1.0)
        evolve!(probQ; nsteps=2)
        energy(probQ); circulation(probQ)

        # QG / Periodic — warm Ewald cache for (Lx, Ly) combos used in docs.
        cQP = circular_patch(0.3, N, 2π)
        probQP = Problem(; contours=[cQP], dt=0.005, kernel=:qg, Ld=1.0,
                         domain=:periodic, Lx=3.0, Ly=3.0)
        evolve!(probQP; nsteps=2)
        circulation(probQP)

        cQP2 = circular_patch(0.3, N, 1.0)
        probQP2 = Problem(; contours=[cQP2], dt=0.05, kernel=:qg, Ld=2.0,
                          domain=:periodic, Lx=Float64(π), Ly=Float64(π))
        evolve!(probQP2; nsteps=2)

        # beta_staircase + closed vortex on a contour beta-plane QG domain.
        # Keep small: this only has to specialize the methods; runtime cost
        # during precompilation scales with contour count × nsteps.
        sc = beta_staircase(1.0, PeriodicDomain(3.0), 2; nodes_per_contour=N)
        cV = circular_patch(0.3, N, 2π)
        probBP = Problem(; contours=vcat(sc, [cV]), dt=0.005,
                         kernel=:beta_plane_qg, beta=1.0, Ld=1.0,
                         domain=:periodic, Lx=3.0, Ly=3.0)
        evolve!(probBP; nsteps=2)

        # SQG / Unbounded
        cS = circular_patch(0.5, N, 1.0)
        probS = Problem(; contours=[cS], dt=0.005, kernel=:sqg, δ_sqg=0.02)
        evolve!(probS; nsteps=2)

        # Multi-layer QG / Unbounded
        cM = circular_patch(0.5, N, 2π)
        F = 0.5
        probM = Problem(; layers=([cM], PVContour{Float64}[]),
                        dt=0.01, kernel=:multilayer_qg,
                        Ld=[1.0], coupling=[-F F; F -F])
        evolve!(probM; nsteps=2)
        energy(probM); circulation(probM)
    end
end

end # module
