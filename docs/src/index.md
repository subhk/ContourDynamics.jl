```@raw html
---
layout: home

hero:
  name: "ContourDynamics.jl"
  text: "Vortex Patch Simulations in Julia"
  tagline: Contour dynamics and contour surgery for 2D Euler, single-layer QG, and N-layer quasi-geostrophic flows — all without gridding
  actions:
    - theme: brand
      text: Get Started
      link: /tutorial_euler
    - theme: alt
      text: API Reference
      link: /api
    - theme: alt
      text: View on GitHub
      link: https://github.com/subhk/ContourDynamics.jl

features:
  - title: 2D Euler & QG Kernels
    details: Classical log(r) Green's function for Euler, K₀(r/Ld) for QG with configurable deformation radius, and N-layer coupling via eigenmode decomposition.
  - title: Contour Surgery
    details: Full Dritschel surgery suite — node redistribution, contour reconnection, and filament removal for long-time simulations.
  - title: Ewald Summation
    details: Doubly-periodic domains via Ewald splitting with precomputed Fourier coefficients and lazy thread-safe caching.
  - title: Analytical Diagnostics
    details: Energy, enstrophy, circulation, angular momentum, and ellipse moments computed directly from contour geometry.
  - title: Ecosystem Integration
    details: Package extensions for DifferentialEquations.jl, Makie.jl, GeophysicalFlows.jl, and RecordedArrays.jl.
  - title: High Performance
    details: StaticArrays for contour nodes, threaded velocity computation, and type-stable dispatch on kernel traits.
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

## What is Contour Dynamics?

Contour dynamics is a Lagrangian numerical method for simulating inviscid, incompressible vortex flows. Instead of solving the vorticity equation on a grid, the method tracks the **boundaries of uniform potential vorticity (PV) patches**. All computations — velocities, diagnostics, interactions — are performed analytically on the contour boundaries.

This approach is exact for piecewise-constant PV distributions and avoids numerical diffusion entirely, making it ideal for studying vortex mergers, filamentation, and geophysical vortex dynamics.
