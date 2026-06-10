# A `velocity!` Call, Step by Step

This page traces a single `velocity!` call through the real code, one hop at a
time, so you can build a mental model of the velocity engine without
reverse-engineering it. It follows the most common case:

```julia
prob = Problem(; contours=[circular_patch(1.0, 128, 2pi)], dt=0.01)
vel  = zeros(SVector{2,Float64}, total_nodes(prob.contour_problem))
velocity!(vel, prob.contour_problem)   # single-layer, Euler, Unbounded, CPU
```

Everything below lives in `src/velocity/common.jl` unless noted otherwise.

## The call graph

```text
velocity!(vel, prob)                         # public entry (CPU method)
└─ _velocity_policy!(vel, prob)              # shared CPU+GPU skeleton
   ├─ _validate_velocity_buffer!(vel, prob) # buffer big enough?
   └─ _small_velocity!(vel, prob)           # device branch: CPU vs GPU
      └─ _direct_velocity!(vel, prob)        # the CPU reference loop
         └─ _accumulate_node_velocity(...)   # sum over source segments, per target node
            └─ _segment_velocity_with_geometry(...)   # straight vs curved segment
               └─ segment_velocity(kernel, domain, x, a, b)   # the Green's-function math
```

Five thin hops, then the math leaf. Each hop exists for one reason — spelled out
below.

## Hop 1 — `velocity!` (public entry)

```julia
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) where {T}
    return _velocity_policy!(vel, prob)
end
```

This method matches **CPU** problems (the `CPU` in the type parameters). A second
`velocity!` method matches `…, T, GPU`. Both immediately hand off to the same
`_velocity_policy!` so the validation-then-dispatch skeleton is written once.

`prob` is a [`ContourProblem`](@ref): it bundles the **kernel** (`EulerKernel`,
`QGKernel`, …), the **domain** (`UnboundedDomain` or `PeriodicDomain`), the
**contours** (vectors of `PVContour` nodes), and reusable scratch buffers.

## Hop 2 — `_velocity_policy!` (shared skeleton)

```julia
function _velocity_policy!(vel, prob)
    _validate_velocity_buffer!(vel, prob)
    _small_velocity!(vel, prob)
    return vel
end
```

The single point shared by the CPU and GPU public methods. It does two things in
order: check the output buffer, then compute. Keeping it separate means the
device-specific `velocity!` methods stay one line each.

## Hop 3 — `_validate_velocity_buffer!`

`velocity!` accepts an **oversized** reusable buffer (handy when the node count
shrinks after surgery), so this just asserts `length(vel) >= total_nodes(prob)`
and returns `N`. Only the first `N` slots get written.

## Hop 4 — `_small_velocity!` (the device branch)

```julia
# CPU: every single-layer kernel uses the direct evaluator.
_small_velocity!(vel, prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) =
    _direct_velocity!(vel, prob)

# GPU: supported kernels run through the KernelAbstractions path.
_small_velocity!(vel, prob::ContourProblem{K, D, T, GPU}) =
    _ka_velocity!(vel, prob, prob.dev)
```

**This is the one place the device matters.** Julia's multiple dispatch picks the
method by the `CPU`/`GPU` type parameter. On CPU there is a single method for all
kernels (Euler/QG/SQG, unbounded or periodic) — they all use the allocation-free
direct evaluator. The GPU method routes to the KA kernels in `src/accel/ka/`.

> Why a whole method just to pick a function? Because it is the device seam. One
> public `velocity!`, one `_velocity_policy!`, and this method is where CPU and
> GPU part ways — without it, every public method would repeat the validate +
> dispatch logic.

## Hop 5 — `_direct_velocity!` (the CPU reference loop)

This is where work actually happens. In order:

1. **Prefetch the Ewald cache** — `_prefetch_ewald(domain, kernel)`. Returns
   `nothing` for `UnboundedDomain` (no periodic images); for `PeriodicDomain` it
   fetches the cached Fourier/Ewald data once, not per node.
2. **Prepare curvature buffers** — `_prepare_curvature_buffers!` fills per-node
   signed curvatures (reused scratch, so no allocation). Curvature decides
   straight-vs-curved integration later.
3. **Loop over every target node**, writing `vel[i]`. Two branches:
   - **Serial** (small problems): a plain nested loop over contours and nodes.
   - **Threaded** (`N ≥ 128`, multiple threads): `Threads.@threads` over a flat
     node index `i`. The flat index is mapped back to `(contour, local node)`
     with `searchsortedlast` over precomputed `offsets` — that lookup is the
     price of threading a ragged set of contours with one flat loop.

   Both branches call the same inner helper for each target node.

## Hop 6 — `_accumulate_node_velocity` (sum over sources)

```julia
function _accumulate_node_velocity(kernel, domain, contours, source_curvatures, ewald, xi)
    v = zero(SVector{2,T})
    for (source_ci, c) in pairs(contours)
        pv = c.pv
        κ  = source_curvatures[source_ci]
        for j in 1:nnodes(c)
            a = c.nodes[j]
            b = next_node(c, j)        # wraps to node 1 on closed contours
            v += pv * _segment_velocity_with_geometry(
                     kernel, domain, xi, a, b, κ[j], κ[mod1(j+1, nc)], ewald)
        end
    end
    return v
end
```

This is the O(N²) heart: the velocity at one target node `xi` is the sum, over
**every segment of every contour**, of that segment's induced velocity, weighted
by the source contour's PV jump `pv`. A segment is the pair `(a, b)` of
consecutive nodes.

## Hop 7 — `_segment_velocity_with_geometry` (straight vs curved)

```julia
ds_len = norm(b - a)
max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
    return segment_velocity(kernel, domain, x, a, b, ewald)   # nearly straight: analytic
return curved_segment_velocity(kernel, domain, x, a, b, κa, κb, ewald)  # curved: quadrature
```

A dimensionless curvature test. Nearly straight segments take the cheaper,
more accurate **analytic** path; curved segments use cubic interpolation with
fixed Gauss–Legendre quadrature.

## The math leaf — `segment_velocity`

The kernel × domain combination selects the actual Green's-function formula by
dispatch:

| kernel | domain | file |
|--------|--------|------|
| Euler / QG / SQG | `UnboundedDomain` | `src/velocity/unbounded/single_layer.jl` |
| Euler / QG / SQG | `PeriodicDomain`  | `src/velocity/periodic/single_layer.jl` |

For example, `segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)`
integrates the 2D Euler Green's function `log|x − x'|²` analytically over the
segment. QG adds a `K₀` (Bessel) correction on top of the Euler term; SQG uses a
regularised `1/√(r²+δ²)` kernel. The math is documented inline in those files and
in [Theory & Method](theory.md).

## Why so many layers?

Each hop removes a different kind of duplication:

- **`_velocity_policy!`** — one validate-then-compute skeleton for both devices.
- **`_small_velocity!`** — the *only* CPU/GPU branch; everything above it is
  device-agnostic.
- **`_direct_velocity!`** — owns buffer reuse and the serial/threaded choice, so
  the inner math never thinks about threading.
- **`_accumulate_node_velocity`** — the segment sum lives in one place, shared by
  the serial and threaded branches.
- **`_segment_velocity_with_geometry`** — the straight/curved decision, so each
  `segment_velocity` method only implements one clean formula.

If you only remember one thing: **CPU velocity is `_direct_velocity!`**, an
O(N²) double loop over segments. The layers above it are dispatch plumbing; the
files below it are physics.
