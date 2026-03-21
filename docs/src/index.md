```@raw html
---
layout: home

hero:
  name: "ContourDynamics.jl"
  text: "Lagrangian Vortex Patch Simulations"
  tagline: Track vortex boundaries, not grid points — contour dynamics and surgery for 2D Euler, SQG, QG, and N-layer quasi-geostrophic flows in Julia
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
  - icon: 🌀
    title: 2D Euler, SQG & QG Kernels
    details: Exact log(r) antiderivative for Euler. Regularized 1/r with arcsinh antiderivative for SQG. K₀(r/Ld) with singular subtraction and 5-point Gauss-Legendre for QG. N-layer coupling via eigenmode decomposition — all without numerical diffusion.
    link: /theory
    linkText: Learn the math
  - icon: ✂️
    title: Contour Surgery
    details: Full Dritschel algorithm — adaptive node redistribution, automated reconnection (merge & split) via spatial indexing, and filament removal for arbitrarily long integrations.
    link: /tutorial_euler
    linkText: Try a tutorial
  - icon: 🔁
    title: Doubly-Periodic Domains
    details: Ewald summation with precomputed Fourier coefficients, lazy thread-safe caching, and automatic node wrapping. Supports beta-plane PV staircases for geophysical applications.
    link: /tutorial_qg#periodic-domains-and-beta-staircases
    linkText: See periodic example
  - icon: 📊
    title: Analytical Diagnostics
    details: Energy, enstrophy, circulation, angular momentum, and ellipse moments computed directly from contour geometry via Green's theorem — no interpolation, no gridding.
    link: /api#Diagnostics
    linkText: View diagnostics API
  - icon: 🔌
    title: Ecosystem Integration
    details: Package extensions for DifferentialEquations.jl, Makie.jl, GeophysicalFlows.jl, RecordedArrays.jl, and JLD2.jl — plug into the broader Julia ecosystem.
  - icon: ⚡
    title: High Performance
    details: StaticArrays for node positions, multi-threaded velocity and energy computation, O(log C) spatial indexing for surgery, and zero-allocation time stepping.
---
```

## Quick Start

```julia
using ContourDynamics
using StaticArrays

# Create a circular vortex patch
R, N, pv = 1.0, 128, 1.0
nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])

# Evolve with RK4 + surgery
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 20)
evolve!(prob, stepper, params; nsteps=1000)

# Check conserved quantities
println("Energy: ", energy(prob))
println("Circulation: ", circulation(prob))
```

## Installation

```julia
using Pkg
Pkg.add("ContourDynamics")
```

Requires Julia 1.10 or later.

## What is Contour Dynamics?

Contour dynamics is a **Lagrangian numerical method** for simulating inviscid, incompressible vortex flows. Instead of solving the vorticity equation on a grid, the method tracks the **boundaries of uniform potential vorticity (PV) patches**.

The key idea: for piecewise-constant PV distributions, the velocity at any point can be written as a **boundary integral** over the contour edges:

```math
\mathbf{u}(\mathbf{x}) = \sum_j \frac{q_j}{2\pi} \oint_{C_j} G(|\mathbf{x} - \mathbf{x}'|) \times d\mathbf{x}'
```

where ``G`` is the Green's function (``-\log r`` for 2D Euler, ``-1/r`` for SQG, ``K_0(r/L_d)`` for QG). Each segment integral is computed **analytically** (Euler, SQG) or with **high-order quadrature** (QG), so the method introduces **no numerical diffusion**.

**Contour surgery** (Dritschel, 1988) extends this to long-time integrations by automatically handling topological changes — vortex mergers, contour splitting, and filament removal — that would otherwise cause the contour to develop unresolvable complexity.

### When to use contour dynamics

Contour dynamics is ideal when:

- You need **exact PV conservation** (no diffusion, no dissipation)
- The flow is well-described by **piecewise-constant PV** (vortex patches, PV staircases)
- You want to study **vortex mergers, filamentation, and long-time dynamics**
- You're working in **2D Euler or quasi-geostrophic** settings

It is less suitable for smooth PV distributions (use spectral/pseudospectral methods) or 3D flows.
