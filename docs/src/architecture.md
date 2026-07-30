# Architecture Guide

This page is the shortest path through the codebase for a new contributor.

If you only want to understand how a simulation runs, read these files in order:

1. [`src/core/problem_factory.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/core/problem_factory.jl)
2. [`src/core/evolution.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/core/evolution.jl)
3. [`src/velocity/common.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/velocity/common.jl)
4. [`src/beta_plane.jl`](https://github.com/subhk/ContourDynamics.jl/blob/main/src/beta_plane.jl), for beta-plane QG problems

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

Most code falls into one of six layers:

- `src/core/`: types, problem construction, time integration, surgery, geometry helpers
- `src/beta_plane.jl`: beta-plane QG velocity composition
- `src/velocity/`: public velocity API and direct CPU policies
- `src/velocity/periodic/`: Ewald cache and periodic single-layer corrections
- `src/accel/ka/`: KernelAbstractions-based direct kernels for CPU/GPU backends
- `src/diagnostics/`: geometry and integral diagnostics

## Codebase Structure

```text
src/
├── ContourDynamics.jl            module definition, includes, exports
├── beta_plane.jl                 beta-plane QG velocity composition
├── core/
│   ├── types.jl                  includes the type-definition files below
│   ├── kernel_types.jl           EulerKernel, QGKernel, SQGKernel, MultiLayerQGKernel
│   ├── beta_plane_types.jl       BetaPlaneQGKernel + the shared sawtooth jet formula
│   ├── contour_types.jl          PVContour
│   ├── domain_types.jl           UnboundedDomain, PeriodicDomain
│   ├── problem_types.jl          ContourProblem, MultiLayerContourProblem
│   ├── stepper_types.jl          RK4Stepper, LeapfrogStepper
│   ├── surgery_types.jl          SurgeryParams
│   ├── problem.jl                the high-level Problem wrapper
│   ├── problem_factory.jl        keyword construction of Problem
│   ├── contours.jl               contour geometry, remeshing, beta_staircase
│   ├── domains.jl                periodic wrapping, minimum-image helpers
│   ├── surgery.jl                CPU surgery: filaments, reconnection, remesh
│   ├── evolution.jl              evolve!, timestep!, periodic wrap dispatch
│   ├── evolution_buffers.jl      flat packing/scatter, stepper update kernels
│   ├── device.jl                 CPU/GPU device tags and allocation shims
│   ├── device_state.jl           DeviceContourState, the device-resident layout
│   ├── shapes.jl                 circular_patch, elliptical_patch, …
│   └── show.jl                   pretty printing for the public types
├── velocity/
│   ├── common.jl                 public velocity! API and dispatch policy
│   ├── unbounded/single_layer.jl unbounded Euler/QG/SQG segment velocity
│   └── periodic/
│       ├── cache.jl              Ewald cache construction and locking
│       └── single_layer.jl       periodic single-layer corrections
├── accel/ka/
│   ├── packing.jl                flat SegmentData layout, reusable workspaces
│   ├── kernels.jl                @kernel velocity kernels and scalar helpers
│   ├── velocity.jl               launch wrappers, dispatch, entry points
│   └── surgery/                  device-resident surgery pipeline
│       ├── types.jl              FlatContourTopology and the flat packing helpers
│       ├── filaments.jl          filament flagging and stream compaction
│       ├── pairs.jl              close-pair detection, admissibility, pair selection
│       ├── rewrite.jl            split/merge topology rewrite and output layout
│       ├── remesh.jl             Dritschel weighted remeshing, area preservation
│       └── driver.jl             pipeline drivers and the public surgery! methods
└── diagnostics/
    ├── geometry.jl               area, circulation, enstrophy, angular momentum
    ├── ka_energy.jl              device-resident energy, single- and multi-layer
    ├── unbounded/                unbounded energy, single- and multi-layer
    └── periodic/                 periodic energy, single- and multi-layer

ext/                              package extensions, loaded on demand
├── ContourDynamicsCUDAExt.jl     wires GPU() to CuArray and the CUDA KA backend
├── ContourDynamicsDiffEqExt.jl   OrdinaryDiffEq bridge (CPU state)
├── ContourDynamicsJLD2Ext.jl     checkpointing and recorders
├── ContourDynamicsMakieExt.jl    plotting and animation
└── ContourDynamicsRecordedArraysExt.jl   time-series recording
```

The densest reading is `core/surgery.jl` (the CPU reference surgery pass) and
the `accel/ka/surgery/` stages that mirror it on the device; neither is a good
starting point for reading the package. `core/evolution_buffers.jl` is likewise
mostly flat-buffer bookkeeping rather than model logic.

## Core Types

The object model is split by responsibility under `src/core/`:

- `kernel_types.jl`: `EulerKernel`, `QGKernel`, `SQGKernel`, `MultiLayerQGKernel`
- `beta_plane_types.jl`: `BetaPlaneQGKernel`
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

The constructor in `src/core/problem_factory.jl` does six things:

1. validates whether this is single-layer or multilayer
2. normalizes contour or layer precision to the requested `T`
3. builds the kernel and domain
4. attaches beta-plane reference contours when `kernel=:beta_plane_qg`
5. builds `ContourProblem` or `MultiLayerContourProblem`
6. builds the time stepper and surgery settings

When input contours already have element type `T`, their containing vectors are
reused. Mismatched inputs are copied into `PVContour{T}` values, including node,
wrap, and corner data, so the contour problem and time-stepper buffers always
agree on precision.

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

- direct evaluation, including beta-plane QG

For multilayer CPU problems:

- direct modal decomposition

For GPU-tagged problems — single-layer Euler, QG, SQG, and beta-plane QG, and
multi-layer QG:

- KernelAbstractions direct path, reading the device-resident state

All GPU-tagged problems are device-resident: velocity, RK4/leapfrog
timestepping, periodic wrapping, surgery, and diagnostics operate on
`DeviceContourState` without a per-step host round-trip. Surgery runs entirely
on the device in both unbounded and periodic domains, including periodic
minimum-image pair detection and cross-seam topology rewrites.

## Velocity Backends

The package has several velocity implementations.

### Direct CPU

Implemented mostly in `src/velocity/common.jl`.

- easiest to read
- useful as the reference implementation
- used for small problems and many tests
- beta-plane QG adds its direct correction in `src/beta_plane.jl`

### Periodic / Ewald

Implemented in `src/velocity/periodic/`.

- `cache.jl`: Ewald cache construction and locking
- `single_layer.jl`: periodic single-layer corrections

For periodic CPU direct velocity, the code prefetches the Ewald cache once per call.

### KernelAbstractions (KA)

Implemented under `src/accel/ka/`, split by concern:

- `packing.jl`: flat `SegmentData` layout and per-size workspace buffers
- `kernels.jl`: scalar contribution helpers and the `@kernel` velocity kernels
- `velocity.jl`: launch wrappers, dispatch, and the `_ka_velocity!` entry points
- `surgery/`: the staged device-resident surgery pipeline

The KA layer runs on both backends:

- CPU execution through `KernelAbstractions.CPU()` (used to validate the kernels
  against the scalar reference in tests)
- GPU execution through the CUDA extension

The KA layer contains:

- flat segment kernels
- periodic KA variants
- flat contour topology buffers backing the device surgery pipeline
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
- an unbounded `GPU()` surgery dispatch (single-layer, and per-layer for
  multi-layer problems) that uses device-side cleanup flags, close-pair scans,
  remeshing, reconnection planning, and contour rewrites, then updates the
  active `DeviceContourState`
- a periodic `GPU()` surgery dispatch using device-side minimum-image
  admissibility, deterministic pair selection, and cross-seam frame shifts in
  the topology rewrite kernels
- multi-layer velocity evaluation through the state-based modal evaluator,
  which packs per-layer segments with modal PV weights and reuses the
  single-layer KA kernels once per vertical mode
- multi-layer energy through the same modal trick: the segment geometry is
  packed once and only the per-segment PV weight is rewritten per mode
- the beta-plane device path, which caches the frozen reference staircase with
  negated PV in the tail of its segment buffers and adds the analytic sawtooth
  jet with a separate kernel

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
device-resident `DeviceContourState`. Velocity, timestepping, surgery, and
diagnostics read that state directly, for single-layer Euler, QG, SQG, and
beta-plane QG as well as multi-layer QG. CPU contour reconstruction is reserved
for explicit output boundaries such as `materialize_contours`, JLD2 snapshots,
Makie animation frames, and interactive inspection.

Single-point velocity probes upload one target and copy back the two result
scalars; scalar surgery counts and diagnostic reductions may likewise cross for
control flow or return values. The OrdinaryDiffEq bridge uses a CPU vector state
and therefore rejects GPU problems rather than falling back.

## Read This File If...

- “I want to understand how a user call becomes a simulation”:
  `src/core/problem_factory.jl`, then `src/core/evolution.jl`
- “I want to understand velocity dispatch”:
  `src/velocity/common.jl`
- “I want to understand beta-plane QG”:
  `src/core/beta_plane_types.jl`, then `src/beta_plane.jl`
- “I want to understand periodic domains”:
  `src/velocity/periodic/cache.jl`, then `src/velocity/periodic/single_layer.jl`
- “I want to understand acceleration”:
  `src/accel/ka/packing.jl`, then `src/accel/ka/kernels.jl`
- “I want to understand GPU / KA code”:
  `src/accel/ka/` (`packing.jl` → `kernels.jl` → `velocity.jl`), then
  `src/accel/ka/surgery/` (`types.jl` → `driver.jl`, then the stage you need)
- “I want to understand surgery”:
  `src/core/surgery.jl`, then `src/accel/ka/surgery/driver.jl` for the device pass

## Typical Change Map

- Add a new kernel:
  type in `src/core/kernel_types.jl`, direct segment logic, then diagnostics/tests
- Change beta-plane QG:
  `src/core/beta_plane_types.jl` (kernel type and the shared sawtooth formula),
  `src/beta_plane.jl` (CPU composition), `src/core/contours.jl`
  (`beta_staircase`), and `src/accel/ka/velocity.jl` for the device path.
  The sawtooth jet lives in `_beta_sawtooth_u` so the CPU evaluator and the KA
  kernel cannot drift apart — change it there, not in either caller.
- Change default user construction:
  `src/core/problem_factory.jl`
- Change time stepping:
  `src/core/evolution.jl`
- Change contour packing / flat buffer logic:
  `src/core/evolution_buffers.jl`
- Change public velocity selection policy:
  `src/velocity/common.jl`
- Change GPU / KA direct kernels:
  `src/accel/ka/kernels.jl` (math), `src/accel/ka/velocity.jl` (dispatch)

## Beginner Advice

Do not start by reading the whole package top-to-bottom.

A better sequence is:

1. create a tiny `Problem`
2. follow `Problem(...)`
3. follow `evolve!`
4. follow `velocity!`
5. only then read acceleration internals if needed

That gives the right mental model before the performance layers add complexity.
