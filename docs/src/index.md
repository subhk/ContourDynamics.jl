# ContourDynamics.jl

A Julia package for simulating the evolution of vortex patches using the contour dynamics and contour surgery method.

## Features

- **2D Euler dynamics** with the classical log(r) Green's function
- **Single-layer quasi-geostrophic (QG)** with K₀(r/Ld) kernel and configurable deformation radius
- **N-layer QG** with layer coupling via SMatrix for baroclinic instability studies
- **Full Dritschel surgery suite**: node redistribution, contour reconnection, and filament removal
- **RK4 and leapfrog** time integrators with optional DifferentialEquations.jl integration
- **Ewald summation** for doubly-periodic domains
- **Analytical diagnostics**: energy, enstrophy, circulation, angular momentum, ellipse moments — all computed from contour geometry without gridding
- **Makie.jl recipes** for visualization and animation (via package extension)
- **GeophysicalFlows.jl interop** for grid-to-contour conversion (via package extension)

## Quick Start

```julia
using ContourDynamics
using StaticArrays

# Create a circular vortex patch
R, N, pv = 1.0, 128, 1.0
nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])

# Evolve
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 20)
evolve!(prob, stepper, params; nsteps=1000)

# Check diagnostics
println("Energy: ", energy(prob))
println("Circulation: ", circulation(prob))
```

## Contents

```@contents
Pages = ["tutorial_euler.md", "tutorial_qg.md", "api.md"]
```
