# GPU Execution Without CPU Fallback Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make every advertised `GPU()` computation device-native while retaining only scalar control/result transfers and explicit output materialization.

**Architecture:** Point probes will reuse state-packed KA segment kernels with one device target. Periodic surgery will reuse the flat device pipeline after threading periodic geometry through candidate detection, admissibility, pair selection, and topology rewrite; cross-seam merge translations will be stored in the rewrite plan and applied by the output kernel.

**Tech Stack:** Julia 1.10+, CUDA.jl extension, KernelAbstractions, StaticArrays, Test.

### Task 1: Device-native single-layer point velocity

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_cuda_surgery.jl`
- Modify: `src/accel/ka/velocity.jl`
- Modify: `src/velocity/common.jl`

**Step 1: Write failing backend-neutral point tests**

For Euler, QG, and SQG on unbounded and periodic domains, construct a
`DeviceContourState(..., CPU())`, call the wished-for API

```julia
ContourDynamics._ka_velocity_at_state(state, kernel, domain, point, CPU())
```

and compare it with `velocity(cpu_prob, point)`. Add the beta-plane case using
the same API. Use a state whose geometry differs from the host problem to prove
the state is authoritative.

**Step 2: Run the focused test and verify RED**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL with `_ka_velocity_at_state` undefined.

**Step 3: Implement the minimal device point path**

Add one-point device target/output buffers and launch `_ka_apply_velocity!`
against `_state_segment_data!`. Return only `to_cpu(vx)[1]` and
`to_cpu(vy)[1]`. Add a beta-plane overload that uses `_BetaPlaneWorkspace`,
subtracts the cached reference segments, and launches `_beta_sawtooth_add_ka!`
for the one target. Replace the GPU `velocity(prob, x)` method in
`src/velocity/common.jl` with this helper and remove the host-problem helper.

**Step 4: Add CUDA enforcement and verify GREEN**

Extend `test/test_cuda_surgery.jl` to compare every supported single-layer
point probe with CPU while `CUDA.allowscalar(false)` is active.

Run: `julia --project=. test/test_device.jl`

Expected: PASS.

**Step 5: Commit**

```sh
git add test/test_device.jl test/test_cuda_surgery.jl src/accel/ka/velocity.jl src/velocity/common.jl
git commit -m "fix: evaluate gpu point velocity on device"
```

### Task 2: Device-native multi-layer point velocity

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_cuda_surgery.jl`
- Modify: `src/accel/ka/velocity.jl`
- Modify: `src/velocity/common.jl`

**Step 1: Write a failing modal point test**

Create two- and three-layer state tuples for unbounded and periodic domains.
Compare the wished-for API

```julia
ContourDynamics._ka_multilayer_velocity_at_states(
    states, kernel, domain, point, CPU())
```

with the CPU public point evaluator, including an empty layer.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL with `_ka_multilayer_velocity_at_states` undefined.

**Step 3: Implement the modal point path**

Use the concatenated state segment buffers from `_MultilayerWorkspace`. For
each vertical mode, repack PV with `eigenvectors_inv[mode, layer]`, launch the
concrete Euler/QG KA kernel for a one-element target, and store the modal
velocity in small device arrays. Project the modal values into `N` physical
layer results on-device, then copy back the `2N` returned scalars. Do not create
a host `MultiLayerContourProblem` or materialize layers.

**Step 4: Route public GPU dispatch and verify GREEN**

Replace the multi-layer GPU point method with the new helper and add CUDA
equivalence coverage.

Run: `julia --project=. test/test_device.jl`

Expected: PASS.

**Step 5: Commit**

```sh
git add test/test_device.jl test/test_cuda_surgery.jl src/accel/ka/velocity.jl src/velocity/common.jl
git commit -m "fix: evaluate multilayer gpu point velocity on device"
```

### Task 3: Periodic close-pair detection and admissibility

**Files:**
- Modify: `test/test_device.jl`
- Modify: `src/accel/ka/surgery/pairs.jl`

**Step 1: Write failing cross-seam tests**

Build compatible closed contours on opposite periodic boundaries. Assert that
`_device_admissible_close_segment_buffer(state, δ, domain, CPU())` returns the
same canonical pairs as `find_close_segments` and detects a pair absent from
the unbounded call. Add nested-contour cases to verify periodic local-interior
vorticity filtering.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL because no periodic device method exists.

**Step 3: Implement minimum-image kernels**

Add scalar helpers for wrapping and shifting a segment into the image nearest a
reference point. Pass `periodic`, `Lx`, and `Ly` into candidate and
admissibility kernels. Apply consistent images in contact distance, interior
probe wrapping, and ray casting. Preserve the existing unbounded wrappers by
passing `periodic=false` and zero box lengths.

**Step 4: Run and verify GREEN**

Run: `julia --project=. test/test_device.jl`

Expected: PASS.

**Step 5: Commit**

```sh
git add test/test_device.jl src/accel/ka/surgery/pairs.jl
git commit -m "feat: detect periodic surgery pairs on device"
```

### Task 4: Periodic pair ranking and cross-seam topology rewrite

**Files:**
- Modify: `test/test_device.jl`
- Modify: `src/accel/ka/surgery/types.jl`
- Modify: `src/accel/ka/surgery/pairs.jl`
- Modify: `src/accel/ka/surgery/rewrite.jl`
- Modify: `src/accel/ka/surgery/driver.jl`

**Step 1: Write failing split/merge tests**

Compare device pair selection and rewrite against CPU `reconnect!` for a
same-contour periodic split and a different-contour cross-seam merge. Assert
node order, corner flags, PV, wrap, and minimum-image geometry equivalence.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL because ranking and rewrite use unshifted coordinates.

**Step 3: Make selection domain-aware**

Thread the domain through `_device_reconnection_plan` and
`_device_select_reconnection_pair_buffer`. Compute ranking distance from
minimum-image segment frames.

**Step 4: Record and apply merge translations**

Extend `DeviceTopologyRewritePlan` with `merge_shift_x` and `merge_shift_y`.
In `_topology_rewrite_size_kernel!`, compute the second-contour translation
from the contact nodes after any orientation reversal, use translated endpoints
for stitch selection, and store the translation. In
`_materialize_rewrite_outputs_kernel!`, add it to every node sourced from the
second contour of a merge.

**Step 5: Run and verify GREEN**

Run: `julia --project=. test/test_device.jl`

Expected: PASS.

**Step 6: Commit**

```sh
git add test/test_device.jl src/accel/ka/surgery/types.jl src/accel/ka/surgery/pairs.jl src/accel/ka/surgery/rewrite.jl src/accel/ka/surgery/driver.jl
git commit -m "feat: rewrite periodic surgery topology on device"
```

### Task 5: Route periodic surgery through the device pipeline

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_cuda_surgery.jl`
- Modify: `src/accel/ka/surgery/driver.jl`

**Step 1: Replace the host-boundary regression with failing device tests**

Change the periodic multi-layer test to call `_device_multilayer_surgery!`.
Add single-layer healthy, filament-removal, and cross-seam reconnection cases.
Assert equivalence with CPU surgery and authoritative state.

**Step 2: Run and verify RED**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL because the pipeline and multi-layer helper accept only
`UnboundedDomain`.

**Step 3: Generalize the pipeline**

Accept `AbstractDomain` in the state reconnect loop and surgery pipeline, pass
the domain into pair selection and rewrite, and make spanning-proximity checks
minimum-image aware. Delete `_host_boundary_surgery!`. Route supported periodic
single- and multi-layer GPU methods to the device pipeline.

**Step 4: Verify focused and CUDA-guarded suites**

Run:

```sh
julia --project=. test/test_device.jl
julia --project=. test/test_cuda_surgery.jl
```

Expected: backend-neutral tests pass; CUDA bodies pass when hardware exists and
otherwise report the guard pass.

**Step 5: Commit**

```sh
git add test/test_device.jl test/test_cuda_surgery.jl src/accel/ka/surgery/driver.jl
git commit -m "fix: keep periodic gpu surgery on device"
```

### Task 6: Enforce and document the no-fallback contract

**Files:**
- Modify: `test/test_device.jl`
- Modify: `docs/src/api/devices.md`
- Modify: `docs/src/api/velocity.md`
- Modify: `docs/src/architecture.md`
- Modify: `docs/src/tutorial_qg.md`
- Modify: `docs/src/index.md`

**Step 1: Add a failing dispatch/source guard**

Add tests that inspect the lowered GPU point and surgery methods (when CUDA is
available) and assert that they contain no call to `materialize_contours`,
`_host_boundary_surgery!`, or construction of a `CPU()` problem. Also scan the
advertised GPU source files for obsolete fallback helper names.

**Step 2: Run and verify RED if stale fallback remains**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL until all obsolete fallback code is removed.

**Step 3: Update documentation**

Describe device-native point probes and periodic surgery. State that scalar
counts/results and explicit output materialization are the only host
boundaries. Retain the explicit OrdinaryDiffEq rejection for GPU problems.

**Step 4: Run complete verification**

Run:

```sh
git diff --check
julia --project=. test/test_device.jl
julia --project=. test/test_allocations.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Expected: all tests pass and CUDA-specific bodies are hardware-guarded on hosts
without a functional NVIDIA runtime.

**Step 5: Commit**

```sh
git add test/test_device.jl docs/src/api/devices.md docs/src/api/velocity.md docs/src/architecture.md docs/src/tutorial_qg.md docs/src/index.md
git commit -m "docs: enforce gpu no-fallback contract"
```
