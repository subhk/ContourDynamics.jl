# API Reference: Devices

`CPU()` keeps contour state in ordinary Julia arrays and all simulation work
uses CPU implementations. `GPU()` keeps the active contour state in
device-resident buffers for supported velocity, timestepping, surgery, and
diagnostic paths. GPU problems retain only the active device representation;
use `snapshot_contours(prob)` when you need an owned CPU copy for output,
plotting, file writing, or interactive inspection.

Single-layer Euler, QG, and SQG (unbounded or periodic), beta-plane QG
(periodic), and multi-layer QG all support device-resident velocity,
RK4 timestepping, periodic wrapping, surgery, and geometry diagnostics.
Energy is available for single-layer Euler, QG, and SQG and for multi-layer QG;
beta-plane QG has no energy diagnostic on either CPU or GPU.

The whole surgery pass — cleanup flags, close-pair scans, reconnection planning,
contour rewrites, and Dritschel remeshing — runs on the device in both unbounded
and periodic domains. Periodic close-pair and interior-vorticity tests use
minimum-image geometry, and cross-seam merge translations are applied by the
device topology-rewrite kernels.

Single-point `velocity(prob, x)` probes also evaluate from the authoritative
device state with the same KA segment kernels as node velocity. Only the final
two velocity scalars are copied back. Small scalar counts and diagnostic results
may cross to the host for allocation, control flow, or return values. Bulk host
copies occur only at explicit output boundaries such as `materialize_contours`,
snapshots, plotting, and animation. The CPU-vector OrdinaryDiffEq bridge rejects
GPU problems instead of falling back.

The device velocity and energy paths reuse buffers owned by the problem's
`ExecutionWorkspace`, sized to the current topology. Call
`clear_state_workspace_cache!(prob)` to release segment, copy-back, scan, and
reduction buffers without changing the physical state.

```@docs
AbstractDevice
CPU
GPU
DeviceContourState
materialize_contours
device_array
device_zeros
to_cpu
to_device
clear_state_workspace_cache!
```

## State and workspace ownership

`contours(prob)` borrows live CPU contours and rejects GPU storage.
`snapshot_contours(prob)` returns an independent CPU copy on either backend.
The legacy `materialize_contours(prob)` retains its CPU-borrow/GPU-copy behavior.
GPU field-style reads (`prob.contours` or `prob.layers`) now materialize current
state rather than exposing an obsolete host mirror; modifying that copy does
not update the device.

```@docs
snapshot_contours
ExecutionWorkspace
execution_workspace
```

A workspace belongs to one computation at a time. Independent problems receive
independent workspaces by default. Call `clear_state_workspace_cache!(prob)` to
release a problem's computational buffers; this preserves physical state and
stepper stage arrays. The zero-argument form clears only task-local compatibility
workspaces used by standalone internal state operations.
