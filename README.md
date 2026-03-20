# ContourDynamics.jl

A Julia package for simulating the evolution of vortex patches using the contour dynamics and contour surgery method. Supports 2D Euler, single-layer QG, and N-layer QG dynamics.

## Installation

```julia
using Pkg
Pkg.add("ContourDynamics")
```

## Quick Start

```julia
using ContourDynamics, StaticArrays

# Create a circular vortex patch
R, N, pv = 1.0, 128, 1.0
nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])

# Evolve with RK4
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 20)
evolve!(prob, stepper, params; nsteps=1000)

# Diagnostics — computed analytically from contour geometry
energy(prob)       # kinetic energy
circulation(prob)  # total circulation
enstrophy(prob)    # enstrophy
```

## Features

- **Physics**: 2D Euler, single-layer QG (K₀ kernel), N-layer QG (coupled layers)
- **Surgery**: Full Dritschel suite — remeshing, reconnection, filament removal
- **Domains**: Unbounded and doubly-periodic (Ewald summation)
- **Diagnostics**: Energy, enstrophy, circulation, angular momentum, ellipse moments
- **Time stepping**: RK4, symplectic leapfrog, optional DifferentialEquations.jl integration
- **Visualization**: Makie.jl plot recipes (via package extension)
- **Ecosystem**: GeophysicalFlows.jl grid-contour conversion (via package extension)

## Citation

If you use ContourDynamics.jl in your research, please cite our JOSS paper (forthcoming).

## License

MIT
