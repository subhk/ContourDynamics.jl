# API Reference: Extensions

These entry points are exported by `ContourDynamics.jl`, but become usable only
after loading the corresponding optional package extension.

## Available Extensions

- `OrdinaryDiffEq` extension: `to_ode_problem`, plus the low-level
  `flatten_nodes` and `unflatten_nodes!` helpers used by the ODE bridge
- `RecordedArrays` extension: `recorded_diagnostics`
- `JLD2` extension: `save_snapshot`, `load_snapshot`, `jld2_recorder`,
  `load_simulation`, `load_problem`
- `Makie` extension: `record_evolution`

```@docs
flatten_nodes
unflatten_nodes!
to_ode_problem
record_evolution
recorded_diagnostics
save_snapshot
load_snapshot
jld2_recorder
load_simulation
load_problem
```

## OrdinaryDiffEq Bridge

`to_ode_problem(prob::ContourProblem, tspan; surgery_params=nothing, surgery_dt=nothing)`
wraps a single-layer contour problem as an OrdinaryDiffEq `ODEProblem`. When
`surgery_params` is provided, it returns a named tuple containing both the ODE
problem and the surgery callback.

The ODE bridge is CPU-only: it flattens contours to a CPU vector and mutates
`prob.contours` inside the RHS closure. Passing a `GPU()` problem throws an
`ArgumentError` instead of materializing device state implicitly.

`flatten_nodes` and `unflatten_nodes!` are also exported for advanced users who
need the contour-state packing used by the ODE bridge.

## Recorded Diagnostics

`recorded_diagnostics(prob; dt, nsteps, record_every=1)` creates RecordedArrays
recorders for energy, enstrophy, circulation, angular momentum, a shared clock,
and an `evolve!` callback.

## Snapshots and Checkpointing

`save_snapshot(filename, prob, step; dt=nothing, diagnostics=true)` writes a
single simulation snapshot to a JLD2 file. For `GPU()` problems, this is an
explicit output boundary: contour geometry is copied back with
`materialize_contours(prob)` immediately before writing.

`load_snapshot(filename, step)` loads one stored snapshot as a named tuple.

`jld2_recorder(filename; save_every=nothing, save_dt=nothing, dt=nothing, diagnostics=true)`
creates an `evolve!` callback that periodically writes JLD2 snapshots.

`load_simulation(filename)` loads all snapshots from a JLD2 file, sorted by step.

## Makie Recording

`record_evolution` writes a simple animation by advancing the built-in time
steppers and capturing contour geometry with Makie. Load `Makie` before calling
it. Makie frames are explicit output boundaries and materialize GPU contours
only for drawing.
