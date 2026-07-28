# Correctness and Allocation Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct five confirmed API/performance defects while preserving existing numerical results and allocation guarantees.

**Architecture:** Normalize high-level constructor inputs at the boundary, dispatch point queries through authoritative device state, and extend the existing task-local workspace pattern to host copy-back and energy reduction buffers. Validate Ewald tuning before constructing cache ranges.

**Tech Stack:** Julia 1.10+, StaticArrays, KernelAbstractions, Test.

### Task 1: Normalize `Problem` input precision

**Files:**
- Modify: `test/test_problem.jl`
- Modify: `src/core/problem_factory.jl`

**Steps:**
1. Add tests showing mismatched single- and multi-layer inputs become `PVContour{T}`, preserve values/corners, and already-matching vectors retain identity.
2. Run `julia --project=. test/test_problem.jl` and confirm the mismatched Euler test fails during timestepping.
3. Add typed contour/layer conversion helpers and call them before building the contour problem.
4. Re-run the focused test and confirm it passes.

### Task 2: Use authoritative state for GPU point queries

**Files:**
- Modify: `test/test_device.jl`
- Modify: `test/test_cuda_surgery.jl`
- Modify: `src/velocity/common.jl`

**Steps:**
1. Add a device-state regression that mutates packed coordinates and checks a point query against materialized current geometry; add CUDA public-API coverage when CUDA is available.
2. Run the focused device test and confirm the missing authoritative-state path fails.
3. Add state-based point-query helpers and explicit GPU dispatch for single- and multi-layer problems.
4. Re-run device tests and confirm current-state results.

### Task 3: Remove size-proportional multilayer GPU `velocity!` allocation

**Files:**
- Modify: `test/test_allocations.jl`
- Modify: `src/accel/ka/packing.jl`
- Modify: `src/accel/ka/velocity.jl`
- Modify: `src/velocity/common.jl`

**Steps:**
1. Add a CPU-backend device-state test comparing small/large warm allocations for host tuple output.
2. Run it and confirm allocation grows with total nodes.
3. Extend the multilayer task-local workspace with reusable flat `SVector` and CPU copy-back buffers, and route public GPU tuple output through it.
4. Re-run the allocation test and numerical equivalence checks.

### Task 4: Reuse device energy packing and partial buffers

**Files:**
- Modify: `test/test_allocations.jl`
- Modify: `src/diagnostics/ka_energy.jl`
- Modify: `src/accel/ka/velocity.jl`

**Steps:**
1. Add warm small/large allocation tests for single-layer device-state energy.
2. Run them and confirm allocation grows with node count.
3. Add task-local energy workspaces for topology metadata, packed segments, and reduction partials; refill in place and rebuild only when layout size changes.
4. Re-run allocation and energy equivalence tests.

### Task 5: Validate Ewald truncation parameters

**Files:**
- Modify: `test/test_periodic_qg_sqg.jl`
- Modify: `src/velocity/periodic/cache.jl`

**Steps:**
1. Add `ArgumentError` tests for negative `n_fourier` and `n_images` across builders/setup, retaining zero as valid.
2. Run the focused periodic test and confirm negative cases do not throw.
3. Add a shared non-negative validation helper at cache-construction entry points.
4. Re-run focused tests.

### Task 6: Verify and integrate

**Files:**
- Verify all modified source and test files.

**Steps:**
1. Run `git diff --check`.
2. Run allocation tests with one thread.
3. Run the complete `Pkg.test()` suite.
4. Confirm the implementation worktree is clean except intended changes and commit them.
5. Use `superpowers:finishing-a-development-branch` to integrate the implementation branch.
