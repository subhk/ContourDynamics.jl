# Correctness and Allocation Fixes Design

## Goal

Fix five confirmed defects: stale GPU point-velocity queries, inconsistent
`Problem` precision, per-call multilayer GPU velocity allocations, per-call GPU
energy allocations, and acceptance of negative Ewald truncation counts.

## Design

The high-level `Problem` constructor will make `T` authoritative. Contours or
layers whose coordinate type differs from `T` will be copied into
`PVContour{T}` values, converting nodes, PV jumps, wrap vectors, and preserving
corner flags. Inputs already using `T` will retain their current identity and
in-place mutation behavior.

GPU point queries will explicitly dispatch on `GPU` and evaluate materialized
authoritative device state instead of the stale host shadow. This correctness
boundary may allocate because a scalar query crosses from device state to host
geometry; bulk GPU velocity remains the performance path.

The public multilayer GPU `velocity!` path will reuse task-local flat output and
copy-back buffers instead of allocating a new flat device array and host copy on
every call. GPU energy will similarly reuse task-local segment-packing and
partial-reduction storage, rebuilding only when the required layout or size
changes. Task-local ownership matches the existing device velocity workspaces
and prevents concurrent tasks from racing on shared buffers.

All public Ewald cache builders and setup methods will reject negative
`n_fourier` and `n_images` values with `ArgumentError`. Zero remains valid as an
explicit lowest-order truncation.

## Testing

Each behavior will be developed test-first. Regression tests will cover
precision conversion and identity preservation, authoritative device-state
point queries, negative Ewald parameters, and warm allocation scaling for the
multilayer velocity and device energy paths. Focused tests will be followed by
the complete package suite and clean-worktree verification.
