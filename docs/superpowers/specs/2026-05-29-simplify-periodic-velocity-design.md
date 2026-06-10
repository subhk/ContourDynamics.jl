# Simplify Periodic Velocity Hot Paths — Dedup + Polish

**Date:** 2026-05-29
**Status:** Approved design, pending spec review
**Approach:** C (dedup + polish)

## Goal

Reduce duplication and improve readability in the periodic-domain velocity hot
paths without changing behavior or performance. The code is already
well-commented; the defect being fixed is **duplication** — the same Green's
function correction math is written two-to-three times, which is a latent
correctness hazard (fix a sign in one copy, miss the other).

## Invariant (non-negotiable)

Behavior-preserving refactor. Floating-point expressions and their evaluation
order are preserved wherever order affects the result. Verified by the full test
suite plus the allocation guard. **No performance regression.**

## Scope

In scope:

- `src/velocity/periodic/single_layer.jl` (521 lines)
- `src/velocity/periodic/cache.jl` (299 lines)
- `src/velocity/common.jl` (639 lines)

Polish only (if warranted): `src/core/contour_types.jl`,
`src/core/domain_types.jl`, `src/core/shapes.jl` — small struct files; docstring
and naming touch-ups only, no logic change.

Explicitly out of scope: GPU paths (`src/accel/gpu/*`), surgery, diagnostics
(unless a shared helper is trivially reused), and any algorithmic change.

## Current-state findings

1. **Green's correction written twice per kernel.** The straight
   `segment_velocity(kernel, ::PeriodicDomain, x, a, b, cache)` methods inline
   the full real-space Ewald sum + Fourier sum:
   - Euler: `single_layer.jl` ~L79–123
   - QG: ~L190–201
   - SQG: ~L268–307

   The identical math also lives in the single-point helpers used by the curved
   path:
   - `_periodic_euler_green_correction` ~L315–360
   - `_periodic_qg_green_correction` ~L362–387
   - `_periodic_sqg_green_correction` ~L389–431

   The curved methods (`curved_segment_velocity`, ~L433–521) already delegate to
   these helpers. Only the straight methods still inline. ~150 redundant lines.

2. **Threaded/serial loop bodies duplicated.** `_direct_velocity!`
   (`common.jl` ~L162–226) copies the entire source-summation loop into both a
   `Threads.@threads` branch (~L185–198) and a serial branch (~L206–219).
   `_multilayer_mode_velocity!` (~L383–465) does the same (~L412–434 vs
   ~L435–458). ~70 redundant lines.

3. **Ewald cache builders repeat setup.** The three `build_ewald_cache` methods
   (`cache.jl` L21–109) recompute `alpha`, `kx`, `ky`, `area` identically and
   differ only in the coefficient formula. The two `setup_ewald_cache!` methods
   (L195–241) duplicate the FIFO evict-and-store block.

## Design

### Part 1 — One source of truth for the periodic correction

Rewrite each straight `segment_velocity(kernel, ::PeriodicDomain, …, cache)` to
loop Gauss–Legendre nodes and call the existing
`_periodic_<kernel>_green_correction` helper, instead of inlining the math:

```julia
function segment_velocity(kernel::K, domain::PeriodicDomain{T},
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          cache::EwaldCache{T}) where {T, K}
    a, b = _nearest_periodic_segment_image(domain, x, a, b)
    ds = b - a
    sqrt(ds[1]^2 + ds[2]^2) < eps(T) && return zero(SVector{2,T})

    base = <per-kernel base velocity>          # see table below
    g_nodes, g_weights = _gl5_nodes_weights(T)
    mid = (a + b) / 2
    half_ds = ds / 2

    corr = zero(T)
    for q in eachindex(g_nodes)
        s_pt = mid + g_nodes[q] * half_ds
        corr += g_weights[q] *
            _periodic_<kernel>_green_correction(kernel, domain, cache, x, s_pt)
    end
    return base + half_ds * corr
end
```

Per-kernel base velocity (unchanged from current code):

| Kernel | Base velocity term |
|--------|--------------------|
| Euler  | `segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)` |
| QG     | `segment_velocity(EulerKernel(), domain, x, a, b, euler_cache)` (periodic Euler) |
| SQG    | `segment_velocity(kernel, UnboundedDomain(), x, a, b)` |

The three `_periodic_*_green_correction` helpers become the only place the
real-space + Fourier correction math lives; both the straight and curved paths
consume them. The thin convenience methods (`(x,a,b)` cache-fetch wrappers and
`(x,a,b,::Nothing)` forwarders) are retained as-is.

**Performance note / risk.** The helpers are currently plain functions (only the
curved path calls them). Routing the straight hot path through them adds 5 calls
per segment. Mark the three helpers `@inline` so the straight loop stays
allocation-free and is not deoptimized. This is the one part that requires a
before/after micro-benchmark, not just correctness tests.

**FP-equivalence note.** The helper computes the same expressions in the same
order as the current inline code (verified term by term against the findings
above), so results are identical up to compiler reassociation, which the
tolerance-based tests already accommodate.

### Part 2 — Function barriers for threaded/serial duplication

Extract the per-target-point source summation into a single `@inline` helper and
call it from both the threaded and serial branches.

Single-layer (`_direct_velocity!`):

```julia
@inline function _accumulate_node_velocity(kernel, domain, contours,
                                           source_curvatures, ewald,
                                           xi::SVector{2,T}) where {T}
    v = zero(SVector{2,T})
    for (source_ci, c) in pairs(contours)
        nc = nnodes(c); nc < 2 && continue
        pv = c.pv
        κ = source_curvatures[source_ci]
        @inbounds for j in 1:nc
            a = c.nodes[j]; b = next_node(c, j)
            v += pv * _segment_velocity_with_geometry(
                kernel, domain, xi, a, b, κ[j], κ[mod1(j + 1, nc)], ewald)
        end
    end
    return v
end
```

Both branches of `_direct_velocity!` then become `vel[i] = _accumulate_node_velocity(...)`.
Apply the same extraction to `_multilayer_mode_velocity!` (the source-layer sum).

Function barriers are the standard Julia idiom for sharing a loop body between a
threaded and serial branch; with `@inline` and a concrete kernel type this is
performance-neutral. Guarded by `test_threading.jl` and the allocation test.

### Part 3 — DRY the Ewald cache builders

```julia
function _ewald_wavenumbers(domain::PeriodicDomain{T}, n_fourier::Int) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    alpha = sqrt(T(π)) / sqrt(Lx * Ly)
    kx = [T(2π * m) / (2 * Lx) for m in -n_fourier:n_fourier]
    ky = [T(2π * n) / (2 * Ly) for n in -n_fourier:n_fourier]
    area = 4 * Lx * Ly
    return alpha, kx, ky, area
end
```

Each `build_ewald_cache` method calls `_ewald_wavenumbers`, then fills
`fourier_coeffs` with its kernel-specific formula — making the actual difference
between kernels obvious.

```julia
function _store_ewald_cache!(caches, order, key, cache)
    if !haskey(caches, key)
        while length(caches) >= _EWALD_CACHE_MAX && !isempty(order)
            delete!(caches, popfirst!(order))
        end
        push!(order, key)
    end
    caches[key] = cache
    return cache
end
```

Both `setup_ewald_cache!` methods use `_store_ewald_cache!`. Cold path, pure
clarity win, no perf consideration.

### Part 4 — Polish

- Add one file-level overview comment at the top of `single_layer.jl` describing
  the shared singular-subtraction strategy (analytic or periodic-Euler base term
  plus a smooth Ewald correction integrated by Gauss–Legendre quadrature), so
  individual methods no longer re-derive it.
- Unify `@inbounds` usage and local naming (`r_vec0`/`r_vec`, `corr`) — mostly
  automatic once Part 1 routes everything through the helpers.
- Optional, only if obviously beneficial: docstring/naming touch-ups in the small
  core type files. No logic change.

## Verification

Run after **every** part, and require green before proceeding:

- `julia --project=. -e 'using Pkg; Pkg.test()'` — full suite. Relevant coverage:
  `test_periodic_qg_sqg.jl`, the "Periodic Domain (Ewald)", "Euler Kernel",
  "QG Kernel", and "Multi-Layer QG" testsets in `runtests.jl`,
  `test_threading.jl`, `test_conservation.jl`, `test_merger.jl`.
- `test_allocations.jl` — `velocity!` and `energy` must stay within their
  existing post-warm-up thresholds (≈8 KB). This is the primary guard against a
  refactor deoptimizing a hot loop.

Part 1 additionally:

- A before/after micro-benchmark on a representative periodic `velocity!` (e.g.,
  a 128-node circular patch on a `PeriodicDomain` for each of Euler/QG/SQG).
  Require runtime within noise (≈≤3%) and unchanged allocation count.

## Work order

Lowest-risk first; the high-value, high-scrutiny correction dedup lands after the
scaffolding is proven green.

1. **Part 3** (cache builders) — cold path, verify suite + allocations.
2. **Part 2** (function barriers) — verify suite + `test_threading.jl` + allocations.
3. **Part 1** (correction dedup) — verify suite + allocations + micro-benchmark.
4. **Part 4** (polish) — verify suite.

## Risks and mitigations

- **Hot-loop deopt from helper calls (Part 1).** Mitigate with `@inline` on the
  three helpers; confirm via the allocation test and micro-benchmark.
- **FP drift from reassociation.** Preserve expression order; rely on
  tolerance-based correctness tests. If any test tightens below tolerance,
  treat as a regression and investigate.
- **Multi-layer/threading correctness (Part 2).** Covered by `test_threading.jl`
  and the multi-layer testset; the extracted helper is a literal lift of the
  existing body.

## Out of scope / explicitly not doing

- No algorithmic or numerical-method changes.
- No metaprogramming (`@eval` loops) to collapse the thin dispatch wrappers —
  it would hurt readability for the scientific audience.
- No changes to GPU, surgery, or diagnostics code.
- No commits without explicit user approval.
