# Architecture Guide

This page is the shortest path through the codebase for a new contributor.

If you only want to understand how a simulation runs, read these files in order:

1. [`src/core/problem_factory.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/core/problem_factory.jl)
2. [`src/core/evolution.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/core/evolution.jl)
3. [`src/velocity/common.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/velocity/common.jl)

## Mental Model

The package simulates piecewise-constant PV patches by evolving their **contour nodes**.

The main object flow is:

```text
Problem(...) -> ContourProblem / MultiLayerContourProblem
             -> timestep!(...)
             -> velocity!(...)
             -> contour node updates
             -> optional surgery!
```

Most code falls into one of five layers:

- `src/core/`: types, problem construction, time integration, surgery, geometry helpers
- `src/velocity/`: public velocity API and direct CPU policies
- `src/velocity/periodic/`: Ewald cache and periodic single-layer corrections
- `src/accel/gpu/`: KernelAbstractions-based direct kernels for CPU/GPU backends

## Core Types

The object model is split by responsibility under `src/core/`:

- `kernel_types.jl`: `EulerKernel`, `QGKernel`, `SQGKernel`, `MultiLayerQGKernel`
- `contour_types.jl`: `PVContour`
- `domain_types.jl`: `UnboundedDomain`, `PeriodicDomain`
- `problem_types.jl`: `ContourProblem`, `MultiLayerContourProblem`
- `surgery_types.jl`: `SurgeryParams`
- `stepper_types.jl`: `RK4Stepper`, `LeapfrogStepper`

The high-level wrapper type is in `src/core/problem.jl`:

- `Problem` bundles a contour problem, a stepper, and optional surgery parameters
- `Problem(...)` keyword construction is implemented in `src/core/problem_factory.jl`

## Execution Flow

### 1. Build a problem

Most users start with:

```julia
prob = Problem(; contours=[circular_patch(1.0, 128, 2pi)], dt=0.01)
```

The constructor in `src/core/problem_factory.jl` does four things:

1. validates whether this is single-layer or multilayer
2. builds the kernel and domain
3. builds `ContourProblem` or `MultiLayerContourProblem`
4. builds the time stepper and surgery settings

### 2. Evolve in time

`evolve!` in `src/core/evolution.jl` is the main simulation loop:

1. run callbacks for step 0
2. call `timestep!`
3. wrap nodes for periodic domains
4. optionally run `surgery!`
5. resize buffers if surgery changed node counts
6. run callbacks for the new step

The low-level flat-buffer packing/scattering helpers live in
`src/core/evolution_buffers.jl` so the public stepping logic stays readable.

### 3. Compute velocities

`velocity!` in `src/velocity/common.jl` is the top-level dispatcher.

For single-layer CPU problems the current policy is:

- direct evaluation

For multilayer CPU problems:

- direct modal decomposition

For supported GPU-tagged single-layer problems:

- KernelAbstractions direct path

## Velocity Backends

The package has several velocity implementations.

### Direct CPU

Implemented mostly in `src/velocity/common.jl`.

- easiest to read
- useful as the reference implementation
- used for small problems and many tests

### Periodic / Ewald

Implemented in `src/velocity/periodic/`.

- `cache.jl`: Ewald cache construction and locking
- `single_layer.jl`: periodic single-layer corrections

For periodic CPU direct velocity, the code prefetches the Ewald cache once per call.

### KernelAbstractions (KA)

Implemented in `src/accel/gpu/common.jl`, with experimental topology-surgery
building blocks in `src/accel/gpu/surgery.jl`.

Despite the filename, this file handles both:

- CPU execution through `KernelAbstractions.CPU()`
- GPU execution through the CUDA extension

The KA layer contains:

- flat segment kernels
- periodic KA variants
- workspace reuse
- flat contour topology buffers for future GPU surgery phases
- device-side contour cleanup flags, close-pair candidate detection, and compact
  close-pair candidate buffers
- device-side admissibility filtering for unbounded close-pair buffers,
  including the same-local-interior-vorticity merge predicate
- device-side reconnect distance planning with compact selected-pair buffers
  built by a KA greedy selection kernel over device distances
- device-side topology rewrite sizing for selected split/merge operations
- device-side materialization of selected split/merge operation outputs
- device-side full contour-list layout and prefix-offset construction, followed
  by materialization that copies unchanged contours, replaces split/merge
  sources, skips deleted merge targets, and appends split daughters in the same
  order as the CPU reconnect path
- device-side Dritschel weighted remeshing for closed contours, including
  fixed-corner span remeshing after topology surgery
- an unbounded single-layer `GPU()` surgery dispatch that uses device-side
  cleanup flags, close-pair scans, remeshing, reconnection planning, and contour
  rewrites, then updates the active `DeviceContourState`

## Threading and Parallelism

There are three kinds of parallelism in the codebase:

### Explicit Julia threading

Used in several CPU paths with `Threads.@threads`, especially:

- direct velocity loops in `src/velocity/common.jl`
- diagnostics pair loops in `src/diagnostics/`

### KA-managed CPU parallelism

When KA kernels run on `CPU()`, they use `KernelAbstractions.CPU()`.
That is a separate execution path from the explicit threaded loops above.

### GPU parallelism

When the CUDA extension is loaded and `dev=GPU()`, the active contour state is a
device-resident `DeviceContourState`. Supported velocity, timestepping,
single-layer surgery, and scalar diagnostics read that state directly. CPU
contour reconstruction is reserved for explicit output boundaries such as
`materialize_contours`, JLD2 snapshots, Makie animation frames, and interactive
inspection.

## Read This File If...

- “I want to understand how a user call becomes a simulation”:
  `src/core/problem_factory.jl`, then `src/core/evolution.jl`
- “I want to understand velocity dispatch”:
  `src/velocity/common.jl`
- “I want to understand periodic domains”:
  `src/velocity/periodic/cache.jl`, then `src/velocity/periodic/single_layer.jl`
- “I want to understand acceleration”:
  `src/accel/gpu/common.jl`
- “I want to understand GPU / KA code”:
  `src/accel/gpu/common.jl`, then `src/accel/gpu/surgery.jl`
- “I want to understand surgery”:
  `src/core/surgery.jl`

## Typical Change Map

- Add a new kernel:
  type in `src/core/kernel_types.jl`, direct segment logic, then diagnostics/tests
- Change default user construction:
  `src/core/problem_factory.jl`
- Change time stepping:
  `src/core/evolution.jl`
- Change contour packing / flat buffer logic:
  `src/core/evolution_buffers.jl`
- Change public velocity selection policy:
  `src/velocity/common.jl`
- Change GPU / KA direct kernels:
  `src/accel/gpu/common.jl`

## Beginner Advice

Do not start by reading the whole package top-to-bottom.

A better sequence is:

1. create a tiny `Problem`
2. follow `Problem(...)`
3. follow `evolve!`
4. follow `velocity!`
5. only then read acceleration internals if needed

That gives the right mental model before the performance layers add complexity.
