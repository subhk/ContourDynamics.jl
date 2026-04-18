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
    details: Built-in kernels for Euler, SQG, QG, and multi-layer QG. Euler and SQG segment integrals are handled analytically, and QG uses a stable high-order quadrature scheme.
    link: /theory
    linkText: Learn the math
  - title: Contour Surgery
    details: Adaptive remeshing, reconnection, and filament removal keep contours well resolved during long integrations.
    link: /tutorial_euler
    linkText: Try a tutorial
  - title: Doubly-Periodic Domains
    details: Periodic domains use Ewald summation and automatic node wrapping. The package also supports beta-plane PV staircases for geophysical examples.
    link: /tutorial_qg#periodic-domains-and-beta-staircases
    linkText: See periodic example
  - title: Analytical Diagnostics
    details: Compute energy, enstrophy, circulation, angular momentum, and ellipse moments directly from the contour geometry.
    link: /api#Diagnostics
    linkText: View diagnostics API
  - title: Ecosystem Integration
    details: Optional extensions connect to DifferentialEquations.jl, Makie.jl, RecordedArrays.jl, and JLD2.jl.
  - title: High Performance
    details: Fast contour kernels, threaded CPU execution, GPU support for the direct Euler path, and low-allocation timestepping.
  - title: GPU Acceleration
    details: Pass `dev=GPU()` to offload supported velocity computations to an NVIDIA GPU through CUDA.jl. Surgery and diagnostics still run on CPU.
    link: /tutorial_euler
    linkText: Try a tutorial
---
```

## Quick Start

If you want the shortest path to a running simulation, start here:

```julia
using ContourDynamics

# Create a circular vortex patch and set up the problem
prob = Problem(; contours=[circular_patch(1.0, 128, 2π)], dt=0.01)

# Evolve with RK4 + surgery
evolve!(prob; nsteps=1000)

# Check conserved quantities
println("Energy: $(energy(prob))")
println("Circulation: $(circulation(prob))")
```

What this does:

- creates one circular vortex patch
- evolves it forward in time with the default RK4 stepper and surgery settings
- prints two basic diagnostics at the end

If you prefer the lower-level API, you can build `ContourProblem`, `RK4Stepper`,
and `SurgeryParams` yourself. The tutorials start with the high-level `Problem`
wrapper because it is the easiest way to get a working simulation.

!!! tip "GPU Acceleration"
    Pass `dev=:gpu` to run velocity computations on an NVIDIA GPU:
    ```julia
    using CUDA
    prob = Problem(; contours=[circular_patch(1.0, 128, 2π)], dt=0.01, dev=:gpu)
    ```
    See the [tutorial](/tutorial_euler) for details.

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

In practice, most workflows look like this:

1. create one or more contours with helpers like `circular_patch` or `elliptical_patch`
2. build a `Problem`
3. call `evolve!`
4. inspect diagnostics such as `energy`, `circulation`, or `vortex_area`

### When to use contour dynamics

Contour dynamics is ideal when:

- You want to keep patch boundaries sharp
- Your flow is well described by piecewise-constant PV
- You care about mergers, filamentation, or long-time patch dynamics
- You are working in 2D Euler, SQG, or quasi-geostrophic settings

It is less suitable for smooth PV distributions (use spectral/pseudospectral methods) or 3D flows.

## Where To Start

- If you are new to the package, start with the [Euler tutorial](/tutorial_euler).
- If you want geophysical flows or periodic domains, go to the [QG tutorial](/tutorial_qg).
- If you want runnable recipes first, open the [Examples](/examples) page.
- If you already know what you need, use the [API Reference](/api).
