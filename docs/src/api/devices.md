# API Reference: Devices

`CPU()` keeps contour state in ordinary Julia arrays and all simulation work
uses CPU implementations. `GPU()` keeps the active contour state in
device-resident buffers for supported velocity, timestepping, surgery, and
diagnostic paths. Host contour containers on a GPU problem are initialization
shadows; use `materialize_contours(prob)` only when you intentionally need a CPU
copy for output, plotting, file writing, or interactive inspection.

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
