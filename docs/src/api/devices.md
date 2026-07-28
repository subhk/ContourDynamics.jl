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

Two paths deliberately cross to CPU representations. A `velocity(prob, x)`
single-point probe materializes the authoritative device state and evaluates
the scalar direct method on that current geometry. The OrdinaryDiffEq bridge
uses a CPU vector state.

The device velocity and energy paths cache scratch workspaces in task-local
storage and size them to the current topology, so repeated calls reuse segment,
copy-back, scan, and reduction buffers. Those buffers live as long as the task; call
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
