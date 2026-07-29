# GPU Completeness Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete and verify the package's advertised NVIDIA CUDA behavior without expanding to cross-vendor GPUs or CPU-only extensions.

**Architecture:** Keep `DeviceContourState` authoritative, validate public GPU outputs from current state, and replace per-call multi-layer energy packing with a task-local reusable workspace. Exercise device-generic kernels on the KernelAbstractions CPU backend and update guarded CUDA tests to enforce the same contract on NVIDIA hardware.

**Tech Stack:** Julia 1.10+, CUDA.jl extension, KernelAbstractions, StaticArrays, Test.

### Task 1: Validate multi-layer GPU output from authoritative state

**Files:**
- Modify: `test/test_device.jl`
- Modify: `src/velocity/common.jl`

**Step 1: Write the failing test**

Create a CPU-backed device-state fixture whose per-layer node counts differ
from the host shadow, then call a small validation helper with buffers sized to
the current states. Assert that current sizes pass and undersized layer buffers
throw `DimensionMismatch`.

**Step 2: Run the test to verify it fails**

Run: `julia --project=. test/test_device.jl`

Expected: FAIL because `_validate_multilayer_state_velocity_buffer!` does not
exist or because validation consults `prob.layers`.

**Step 3: Write the minimal implementation**

Add `_validate_multilayer_state_velocity_buffer!(vel, states)` using
`_device_state_nnodes(states[layer])`. Route GPU tuple `velocity!` directly
through this validation and `_ka_multilayer_velocity_to_host!`; retain the
existing CPU policy.

**Step 4: Run the focused test and commit**

Run: `julia --project=. test/test_device.jl`

Expected: all device tests pass.

Commit: `fix: validate gpu multilayer output from device state`

### Task 2: Reuse multi-layer device energy buffers

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_allocations.jl`
- Modify: `src/diagnostics/ka_energy.jl`
- Modify: `src/accel/ka/velocity.jl`

**Step 1: Write failing numerical and scaling tests**

Add direct-versus-device energy comparisons for two-layer unbounded and
periodic problems. Add a warm allocation test comparing 16 and 128 nodes per
layer and require the large call to remain within fixed backend overhead of the
small call.

**Step 2: Run tests to verify the regression**

Run: `julia --project=. test/test_allocations.jl`

Expected: FAIL because the large call allocates tens of kilobytes more.

**Step 3: Implement reusable packing**

In `ka_energy.jl`, add a task-local `_MultilayerEnergyWorkspace` containing:

- reusable per-layer compaction scratch;
- concatenated geometry, base/modal PV, contour identity, and reduction arrays;
- cached periodic Ewald arrays;
- a topology signature.

Add device kernels to append a packed layer with contour-id offsets and to
apply modal PV weights in place. Make `_pack_energy_workspace!` use the actual
state sizes so capacity-sized scratch can serve different layers. Evaluate raw
energy with the reusable partial buffer and rebuild only when the topology
signature changes. Add the new cache key to `clear_state_workspace_cache!`.

**Step 4: Run focused tests and commit**

Run:

```sh
julia --project=. test/test_allocations.jl
julia --project=. test/test_device.jl
```

Expected: numerical comparisons pass and warm allocation no longer grows with
node count.

Commit: `perf: reuse multilayer gpu energy workspace`

### Task 3: Reconcile tests and documentation with implemented support

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_cuda_surgery.jl`
- Modify: `src/velocity/common.jl`
- Modify: `src/accel/ka/surgery/driver.jl`
- Modify: `docs/src/api/devices.md`
- Modify: `docs/src/api/velocity.md`

**Step 1: Add or correct contract tests**

Add backend-neutral beta-plane velocity comparison and periodic host-boundary
surgery comparison. In CUDA-guarded tests, replace the obsolete multi-layer
energy exception with CPU equivalence for unbounded and periodic domains, and
replace the obsolete periodic multi-layer surgery exception with a successful
state comparison.

**Step 2: Run the focused tests**

Run: `julia --project=. test/test_device.jl`

Expected: backend-neutral tests pass; CUDA tests remain guarded on this host.

**Step 3: Correct messages and docs**

Include beta-plane QG in GPU velocity fallback messages, describe periodic
host-boundary surgery accurately, and state that energy is available for the
kernel/domain combinations listed by the diagnostics API (not beta-plane QG).

**Step 4: Commit**

Commit: `test: enforce complete gpu support matrix`

### Task 4: Verify the completed branch

**Files:**
- Verify all modified source, test, and documentation files.

**Step 1: Run formatting and diff checks**

Run:

```sh
git diff --check
git status --short
```

Expected: no whitespace errors and only intended changes.

**Step 2: Run focused suites**

Run:

```sh
julia --project=. test/test_device.jl
julia --project=. test/test_allocations.jl
```

Expected: all tests pass.

**Step 3: Run the full suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`

Expected: all tests pass. CUDA-specific bodies are skipped because this host
has no NVIDIA runtime.

**Step 4: Review and finish**

Use `superpowers:requesting-code-review`, address every actionable finding,
then use `superpowers:verification-before-completion` and
`superpowers:finishing-a-development-branch`.
