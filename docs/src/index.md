```@raw html
---
layout: home

hero:
  name: "ContourDynamics.jl"
  text: "Lagrangian Vortex Patch Simulations"
  tagline: Simulate vortex patches by tracking their boundaries directly, with contour surgery for long runs in 2D Euler, SQG, QG, and multi-layer QG flows
  actions:
    - theme: brand
      text: Get Started
      link: /tutorial_euler
    - theme: alt
      text: Theory & Method
      link: /theory
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/subhk/ContourDynamics.jl

features:
  - title: 2D Euler, SQG & QG Kernels
    details: Built-in kernels for Euler, SQG, QG, and multi-layer QG. Straight Euler/SQG segment integrals use analytic antiderivatives, curved Dritschel arcs use fixed quadrature, and QG uses stable singular subtraction.
    link: /theory
    linkText: Theory
  - title: Contour Surgery
    details: Adaptive remeshing, reconnection, and filament removal keep contours well resolved during long integrations.
    link: /tutorial_euler
    linkText: Tutorial
  - title: Doubly-Periodic Domains
    details: Periodic domains use Ewald summation and automatic node wrapping. The package also supports beta-plane PV staircases for geophysical examples.
    link: /tutorial_qg#periodic-domains-and-beta-staircases
    linkText: Periodic example
  - title: Analytical Diagnostics
    details: Compute energy, enstrophy, circulation, angular momentum, and ellipse moments directly from the contour geometry, with support depending on the kernel/domain combination.
    link: /api/diagnostics
    linkText: View diagnostics API
  - title: Ecosystem Integration
    details: Optional extensions connect to DifferentialEquations.jl, Makie.jl, RecordedArrays.jl, and JLD2.jl.
  - title: High Performance
    details: Fast contour kernels, threaded CPU execution, device-resident GPU state for supported paths, and low-allocation timestepping.
  - title: GPU Acceleration
    details: Pass `dev=GPU()` to keep supported single-layer velocity, timestepping, surgery, and diagnostics on an NVIDIA GPU. CPU copies are explicit output boundaries via `materialize_contours`, snapshots, and animation frames.
    link: /api/devices
    linkText: Devices API
---
```

## Quick Start

Minimal setup:

```@repl index_quickstart
using ContourDynamics

# Create a circular vortex patch and set up the problem
prob = Problem(; contours=[circular_patch(1.0, 32, 2π)], dt=0.01)

# Evolve with RK4 + surgery
evolve!(prob; nsteps=5)

# Check conserved quantities
println("Energy: $(round(energy(prob); digits=6))")
println("Circulation: $(round(circulation(prob); digits=6))");
```

Summary:

- creates one circular vortex patch
- evolves it forward in time with the default RK4 stepper and surgery settings
- prints two basic diagnostics at the end

For lower-level control, build `ContourProblem`, `RK4Stepper`, and
`SurgeryParams` directly. The tutorials use the high-level `Problem` wrapper as
the default entry point.

!!! tip "GPU Acceleration"
Pass `dev=GPU()` to keep supported single-layer simulation work on an NVIDIA GPU:
    ```julia
    using CUDA
prob = Problem(; contours=[circular_patch(1.0, 128, 2π)], dt=0.01, dev=GPU())
    ```
    GPU problems keep the active contour state in device buffers. Use
    `materialize_contours(prob)` only when you need a CPU copy for output,
    plotting, file writing, or inspection. Unsupported GPU operations throw
    instead of silently falling back to CPU work. `BetaPlaneQGKernel` currently
    requires `dev=CPU()`, and multi-layer GPU support is velocity-only.

## Installation

```julia
using Pkg
Pkg.add("ContourDynamics")
```

Requires Julia 1.10 or later.

## What is Contour Dynamics?

Contour dynamics is a **Lagrangian method** for inviscid flow with piecewise-constant potential vorticity. Instead of solving for vorticity on a grid, it tracks the boundaries of PV patches directly.

For these problems, the velocity can be written as a boundary integral over the patch edges:

```math
\mathbf{u}(\mathbf{x}) = \sum_j \frac{q_j}{2\pi} \oint_{C_j} G(|\mathbf{x} - \mathbf{x}'|) \times d\mathbf{x}'
```

where ``G`` is the Green's function for the model. In practice, the package computes the velocity from contour segments directly, without smearing sharp boundaries onto a grid.

**Contour surgery** keeps the method practical over long integrations by remeshing stretched contours, reconnecting close segments, and removing tiny filaments when needed.

Typical workflow:

1. create one or more contours with helpers like `circular_patch` or `elliptical_patch`
2. build a `Problem`
3. call `evolve!`
4. inspect diagnostics such as `energy`, `circulation`, or `vortex_area`

### When to use contour dynamics

Contour dynamics is ideal when:

- sharp patch boundaries are important
- the flow is well described by piecewise-constant PV
- mergers, filamentation, or long-time patch dynamics matter
- the problem is set in 2D Euler, SQG, or quasi-geostrophic dynamics

It is less suitable for smooth PV distributions (use spectral/pseudospectral methods) or 3D flows.

## Navigation

- [Euler tutorial](tutorial_euler.md) for the standard unbounded single-layer workflow
- [QG tutorial](tutorial_qg.md) for deformation-radius effects, periodic domains, and multi-layer cases
- [Examples](examples.md) for short runnable setups
- [API Reference](api.md) for function and type documentation
