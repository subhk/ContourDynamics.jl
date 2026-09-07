# Architecture Guide

A simulation builds a `Problem`, evaluates contour velocity, advances material
nodes with RK4, and optionally changes topology through surgery. Public kernel,
domain, device, and stepping APIs are unchanged by the internal layering below.

## Dependency direction

```text
core types + storage + execution workspace
    ↓
scalar numerics → geometry → remeshing
    ↓
CPU / KernelAbstractions evaluation and surgery stages
    ↓
evolution, diagnostics, and optional output extensions
```

The package uses one Julia module and multiple source files. These layers are
code ownership boundaries, not dynamically dispatched service objects. Inner
loops retain concrete types and allocation-free scalar operations.

## Numerical formulas and backend loops

`src/numerics/` owns formulas used by both backends:

- `green_functions.jl`: stable panel integrals and special-function approximations
- `velocity_segments.jl`: Euler/QG/SQG segment contributions and periodic corrections
- `interpolation.jl`: cubic point/tangent evaluation
- `contacts.jl`: node-to-segment projections and ray-crossing predicates

`src/velocity/` supplies typed CPU adapters and contour traversal.
`src/accel/ka/kernels.jl` supplies parallel loops over flat arrays. Both call the
same scalar formulas. Periodic QG accepts precomputed correction coefficients
on CPU while device loops can compute them from wavenumbers in the shared helper.

Backend agreement tests check packing and execution. Independent analytical,
Fourier, and image-sum oracles remain necessary: two backends using the same
formula cannot detect a mistake in that formula by comparing with each other.

## State ownership

`ContourProblem` and `MultiLayerContourProblem` each own one storage object:

- `_HostContourStorage` contains the live contour vector or layer tuple.
- `_DeviceContourStorage` contains device-resident flat state or a tuple of states.

There is no retained host geometry mirror on GPU problems. Internal CPU
algorithms call `_host_contours(prob)`, which refuses device storage. Device
algorithms call `_device_state(prob)`. Storage-aware orchestration can dispatch
on `_active_storage(prob)` without repeating device checks.

The output APIs have explicit ownership contracts:

```julia
live = contours(prob)             # CPU only; borrowed, mutations affect prob
saved = snapshot_contours(prob)   # either backend; independent CPU copy
```

The legacy `materialize_contours` keeps its CPU-borrow/GPU-copy behavior for
compatibility. Legacy CPU `.contours`/`.layers` properties still expose live
vectors. GPU reads of these properties materialize current state; mutating the
returned copy does not update the device. `.device_state` remains available for
inspection, and `.velocity_scratch` forwards to the execution workspace.

JLD2 snapshots and Makie output capture use owned snapshots so later evolution
cannot mutate captured host data.

## Workspace ownership and concurrency

Each problem constructs an `ExecutionWorkspace(T)` by default. It owns:

- typed CPU curvature, modal-transform, velocity, and energy scratch;
- reusable CPU surgery node, arc-length, and virtual-node buffers;
- backend-specific velocity and energy buffers and uploaded Ewald tables.

The problem constructor accepts an explicit owner:

```julia
ws = ExecutionWorkspace(Float64)
prob = Problem(contours=[circular_patch(1.0, 64, 1.0)], dt=0.01, workspace=ws)
clear_state_workspace_cache!(prob) # frees computational buffers, keeps geometry
```

Public problem paths pass this workspace through RK stages and device
velocity/energy entry points. Separate problems no longer replace one another's
buffers when their node counts differ. Sequential reuse across models refreshes
modal transforms when the kernel changes. Sharing one mutable workspace across
concurrent computations is unsupported; use independent problems/workspaces.
The same live problem must not be evolved concurrently.

RK4 stage arrays and node ranges remain on the stepper: they belong to that
integration session. Resizing after topology changes is handled by `evolve!`.
Workspace cleanup does not resize or invalidate stepper arrays.

Standalone internal state evaluators accept `workspace=ws`; omitted workspaces
use a task-local compatibility owner. The no-argument
`clear_state_workspace_cache!()` clears those compatibility owners only.

## Geometry, remeshing, and surgery

`src/geometry/` contains polygon moments, signed curvature, interpolation
adapters, and contact geometry. `src/remeshing/` separates density construction,
weighted resampling, and public remeshing with area/corner preservation.

`src/surgery/` contains CPU stages:

1. `spatial_index.jl`: periodic indexing and neighborhood construction
2. `corners.jl`: corner labels and stitch-node insertion
3. `pairs.jl`: containment, admissibility, and independent contact selection
4. `rewrite.jl`: split/merge topology changes
5. `cleanup.jl`: filament removal and spanning-contour checks
6. `driver.jl`: remesh/reconnect/cleanup orchestration and stall handling

The device stages under `src/accel/ka/surgery/` retain their flat-array execution
model. They share scalar contact and interpolation predicates with CPU surgery,
and the reconnection loop shares `_reconnect_until_exhausted!`. Backend-specific
packing, compaction, and topology writes remain local to their execution layer.

`core/contours.jl` and `core/surgery.jl` are small include manifests, preserving
familiar entry points for contributors navigating the source tree.

## Test groups

The group registry is `test/test_groups.jl`. Use:

```bash
julia --project=. --threads=2 test/runtests.jl core
julia --project=. --threads=2 test/runtests.jl numerical device performance
julia --project=. test/runtests.jl jld2
julia --project=. test/runtests.jl hardware
```

`core` includes state/workspace contracts, geometry, stepping, and surgery.
`numerical` contains independent scientific oracles. `device` exercises KA on
CPU, `performance` checks allocations, and `hardware` requires working CUDA.
`jld2`, `diffeq`, and `recorded` request individual extensions; `extensions`
requires all three. No arguments or `all` runs all CPU groups and installed
extensions. Only missing optional dependencies are skipped; load failures and
test failures propagate. CI provisions extension dependencies explicitly.
