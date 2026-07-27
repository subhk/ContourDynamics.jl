# API Reference: Devices

`CPU()` keeps contour state in ordinary Julia arrays and all simulation work
uses CPU implementations. `GPU()` keeps the active contour state in
device-resident buffers for supported velocity, timestepping, surgery, and
diagnostic paths. Host contour containers on a GPU problem are initialization
shadows; use `materialize_contours(prob)` only when you intentionally need a CPU
copy for output, plotting, file writing, or interactive inspection.

Single-layer Euler, QG, and SQG (unbounded or periodic), beta-plane QG
(periodic), and multi-layer QG all support device-resident velocity,
RK4/leapfrog timestepping, periodic wrapping, surgery, and diagnostics —
including energy.

Surgery differs by domain. On unbounded domains the whole pass — cleanup flags,
close-pair scans, reconnection planning, contour rewrites, and Dritschel
remeshing — runs on the device. On periodic domains the pass materializes at the
host boundary, runs the CPU surgery pass (whose cross-seam merge logic handles
minimum-image proximity and periodic frame shifts), and reloads the device
state in place; that cost is amortized over the `n_surgery` steps between
passes.

Two paths remain CPU-only by design: the `velocity(prob, x)` single-point probe
for beta-plane problems, and the OrdinaryDiffEq bridge, which uses a CPU vector
state.

The device velocity paths cache their scratch workspaces in task-local storage
and size them to the current node count, so repeated steps reuse the same
buffers. Those buffers live as long as the task; call
`clear_state_workspace_cache!` to release them — the workspace counterpart to
[`clear_ewald_cache!`](@ref).

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
