# GPU Completeness Design

## Goal

Make the advertised NVIDIA CUDA path internally consistent, numerically covered,
and allocation-stable. The supported contract is `GPU()` through CUDA.jl for
single-layer Euler, QG, and SQG on unbounded and periodic domains, periodic
beta-plane QG, and unbounded or periodic multi-layer QG. Cross-vendor backends
and the deliberately CPU-only OrdinaryDiffEq bridge remain out of scope.

## Contract and boundaries

For supported problems, authoritative geometry lives in `DeviceContourState`.
Bulk velocity, RK4/leapfrog evolution, periodic wrapping, unbounded surgery,
and diagnostics operate against that state. Periodic surgery remains an
explicit host boundary because its mature cross-seam reconnection logic is the
CPU implementation; the result is reloaded into the same device-state object.
Scalar point probes, snapshots, plotting, and other inspection paths may also
materialize intentionally.

Public methods must never validate against construction-time host shadows after
device-side topology changes. In particular, multi-layer `velocity!` will size
each output from the current per-layer device states. Unsupported operations
will fail early with messages matching the actual support matrix.

## Multi-layer energy workspace

Multi-layer energy currently packs all layer geometry into fresh arrays on each
call. Replace that path with a task-local workspace keyed by scalar type and
device, rebuilt only when the layer topology changes. One reusable layer
packing workspace compacts valid contours. A device copy kernel appends each
layer into reusable concatenated arrays while offsetting contour identifiers.
The modal loop changes only PV weights in place and reuses one partial-reduction
buffer. Periodic Ewald arrays are cached in the same workspace.

This keeps allocation independent of node count after warm-up while preserving
the existing modal formulas and normalization. Workspace cleanup remains under
`clear_state_workspace_cache!`.

## Testing

Backend-neutral KernelAbstractions tests will run on CPU storage and compare
against the direct CPU reference for beta-plane velocity and two-/three-layer
energy on unbounded and periodic domains. Allocation tests will compare warm
small and large topologies, which catches size-proportional replacement buffers.
CUDA-guarded tests will enforce multi-layer energy success, periodic
multi-layer surgery success, beta-plane velocity, device-authoritative state,
and supported timestepping when CUDA hardware is available. This host cannot
execute CUDA kernels, so the backend-neutral tests and full package suite are
the local verification boundary.
