# ContourDynamics.jl

[![CI](https://github.com/subhk/ContourDynamics.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/ContourDynamics.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://subhk.github.io/ContourDynamics.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/ContourDynamics.jl/dev/)

ContourDynamics.jl simulates vortex patches by tracking their boundaries. It
supports 2D Euler, surface quasi-geostrophic (SQG), single-layer QG, and
multilayer QG flows.

## Installation

Install the registered package with Julia's package manager:

```julia
using Pkg
Pkg.add("ContourDynamics")
```

ContourDynamics.jl requires Julia 1.10 or later.

## Quick start

```julia
using ContourDynamics

patch = circular_patch(1.0, 128, 2π)
problem = Problem(; contours=[patch], dt=0.01)

evolve!(problem; nsteps=1000)

println("Energy: ", energy(problem))
println("Circulation: ", circulation(problem))
```

`Problem` provides the simplest interface. The lower-level APIs give direct
control over kernels, domains, time steppers, and contour surgery.

## Features

- Unbounded and doubly periodic domains
- Adaptive remeshing, reconnection, and filament removal
- Energy, circulation, enstrophy, and geometry diagnostics
- Threaded CPU execution
- Optional NVIDIA GPU support through CUDA.jl
- Optional plotting, snapshots, diagnostic recording, and DifferentialEquations.jl integration

## Contributing

Bug reports and contributions are welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup and testing instructions.

## License

ContourDynamics.jl is available under the [MIT license](LICENSE).
