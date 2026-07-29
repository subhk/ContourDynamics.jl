# GPU Execution Without CPU Fallback Design

## Goal and contract

Every operation advertised for `GPU()` problems must perform its numerical and
topological work through device arrays and KernelAbstractions kernels.  A GPU
path may copy small scalar counts or final scalar results to the host for
allocation, control flow, and returned diagnostics.  Bulk materialization is
allowed only when the caller explicitly requests a host representation, such
as `materialize_contours`, a snapshot, plotting data, or animation output.
Unsupported kernel/domain combinations must throw instead of silently running
a CPU implementation.

## Point velocity

Single-point velocity queries will pack segments directly from the
authoritative `DeviceContourState`, upload the query point, run the same
unbounded or periodic KA kernels used by node velocity, and copy back only the
two result scalars.  Beta-plane queries will use the cached live-plus-reference
segment workspace and apply the sawtooth correction on-device.  Multi-layer
queries will evaluate every vertical mode from concatenated device state and
project the modal point velocities back to physical layers without constructing
a host `MultiLayerContourProblem`.

## Periodic surgery

Periodic surgery will reuse the existing device surgery pipeline.  Close-pair
detection, interior-vorticity admissibility, pair ranking, and topology rewrite
planning will accept the domain geometry and use minimum-image segment frames.
For cross-seam merges, the rewrite plan will record the periodic translation
that places the second contour in the first contour's contact frame; the output
kernel will apply that translation while preserving wrap orientation.  Remesh,
corner handling, filament removal, compaction, and state replacement remain the
existing device-native stages.  Single-layer and multi-layer periodic dispatch
will call this pipeline directly.

## Verification

Backend-neutral tests will compare device-state point probes and periodic
surgery against CPU reference behavior, including cross-seam reconnection and
multi-layer cases.  CUDA-guarded tests will run with `CUDA.allowscalar(false)`
and verify authoritative state, numerical equivalence, and periodic topology.
Dispatch/source guards will ensure GPU methods do not call host materialization
or construct CPU problems.  Documentation will describe scalar/output
boundaries and remove the obsolete periodic-surgery and point-probe fallback
claims.
