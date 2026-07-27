# ContourDynamics.jl

[![CI](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/ContourDynamics.jl/dev/)

ContourDynamics.jl is a Julia package for Lagrangian simulations of vortex
patches and piecewise-constant potential-vorticity (PV) regions. It tracks patch
boundaries directly and evaluates the velocity from contour integrals, avoiding
the numerical diffusion that can arise when sharp PV interfaces are represented
on a grid.

The package supports 2D Euler, surface quasi-geostrophic (SQG), single-layer QG,
and multilayer QG dynamics on unbounded and doubly-periodic domains. It includes
Dritschel-style contour surgery for long-time integrations, analytical
diagnostics from contour geometry, optional CUDA acceleration for supported
velocity paths, and documented examples for merger, filamentation, beta-plane,
SQG, and multilayer QG workflows.


## Installation

ContourDynamics.jl requires Julia 1.10 or later. Dependencies are managed by
Julia's package manager.

Install from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/subhk/ContourDynamics.jl")
```

If the package is installed in the Julia General registry, the registered release
can be installed with:

```julia
using Pkg
Pkg.add("ContourDynamics")
```

For local development from a clone:

```bash
git clone https://github.com/subhk/ContourDynamics.jl.git
cd ContourDynamics.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Optional functionality is loaded through Julia package extensions:

| Optional package | Provides |
|------------------|----------|
| `CUDA.jl` | device-resident GPU velocity, timestepping, surgery, and diagnostics |
| `Makie.jl` | contour animation helpers |
| `JLD2.jl` | snapshot and restart files |
| `RecordedArrays.jl` | diagnostic time-series recording |
| `OrdinaryDiffEq.jl` | conversion to DifferentialEquations.jl problems |

## Quick Start

```julia
using ContourDynamics

# Circular Euler vortex patch with RK4 timestepping and standard surgery.
prob = Problem(; contours=[circular_patch(1.0, 128, 2pi)], dt=0.01)

evolve!(prob; nsteps=1000)

snapshot = materialize_contours(prob)
println("Energy: ", energy(prob))
println("Circulation: ", circulation(prob))
println("Area: ", vortex_area(snapshot[1]))
```

The high-level `Problem` interface is the recommended starting point. For full
control over kernels, domains, timesteppers, and surgery parameters, use the
lower-level `ContourProblem`, `MultiLayerContourProblem`, `RK4Stepper`, and
`SurgeryParams` APIs.

## Examples

Complete runnable scripts are available in [`examples/`](examples/), with
matching documentation pages under the online docs.

### Vortex Merger

Two co-rotating patches merge through contour surgery when their separation is
below the classical critical distance.

```julia
using ContourDynamics, StaticArrays

R, N, pv = 0.5, 128, 2pi
circle(cx, cy) = [
    SVector(cx + R*cos(2pi*k/N), cy + R*sin(2pi*k/N))
    for k in 0:N-1
]

c1 = PVContour(circle(-0.45, 0.0), pv)
c2 = PVContour(circle(+0.45, 0.0), pv)

prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.005, 0.02, 0.2, 1e-6, 5)

evolve!(prob, stepper, params; nsteps=500)
final_contours = materialize_contours(prob)
println("$(length(final_contours)) contour(s) after merger")
```

### Two-Layer QG

```julia
using ContourDynamics, StaticArrays

Ld = SVector(1.0)
F = 1.0 / (2 * Ld[1]^2)
coupling = SMatrix{2,2}(-F, F, F, -F)
kernel = MultiLayerQGKernel(Ld, coupling)

nodes = [
    SVector(0.5*cos(2pi*k/100), 0.5*sin(2pi*k/100))
    for k in 0:99
]

prob = MultiLayerContourProblem(
    kernel,
    UnboundedDomain(),
    ([PVContour(nodes, 2pi)], PVContour{Float64}[]),
)

stepper = RK4Stepper(0.01, total_nodes(prob))
# SurgeryParams(δ, μ, Δ_max, area_min, n_surgery): keep δ ≤ μ/4, and
# n_surgery < nsteps so surgery actually runs during the integration.
params = SurgeryParams(0.001, 0.005, 0.2, 1e-6, 10)
evolve!(prob, stepper, params; nsteps=200)
```

## Core Functionality

### Physics Kernels

| Kernel | Scalar Green's function | Use case |
|--------|-------------------------|----------|
| `EulerKernel()` | `-(1 / 2pi) log(r)` | 2D incompressible Euler vortex patches |
| `QGKernel(Ld)` | `(1 / 2pi) K0(r / Ld)` | single-layer QG with deformation radius `Ld` |
| `SQGKernel(delta)` | `1 / (2pi * sqrt(r^2 + delta^2))` | regularized SQG |
| `MultiLayerQGKernel(Ld, C)` | eigenmode decomposition | N-layer baroclinic QG |

### Domains and Algorithms

- `UnboundedDomain()` for free-space patch dynamics.
- `PeriodicDomain(Lx, Ly)` for doubly-periodic domains using Ewald summation.
- Dritschel-style remeshing, reconnection, and filament cleanup through
  `surgery!`.
- Analytical contour diagnostics for conserved quantities and vortex geometry.
- CPU threaded execution for direct velocity paths.
- Optional CUDA acceleration with `dev=GPU()`, keeping contour state on the
  device across velocity, timestepping, surgery, and diagnostics.

## Community Guidelines

Contributions, bug reports, documentation improvements, examples, and feature
requests are welcome.

- Open a GitHub issue for questions, bug reports, or support requests:
  <https://github.com/subhk/ContourDynamics.jl/issues>
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing, and pull
  request guidelines.
- The project follows the [ColPrac](https://github.com/SciML/ColPrac) guide for
  collaborative Julia package development.

Before opening a pull request, please run `Pkg.test()` and update documentation
or examples when public behavior changes.

## License

ContourDynamics.jl is distributed under the MIT license. See [LICENSE](LICENSE).
