# ContourDynamics.jl

[![CI](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/ContourDynamics.jl/dev/)

A Julia package for simulating the evolution of vortex patches using the **contour dynamics** and **contour surgery** method. Tracks boundaries of uniform potential vorticity (PV) patches — all computations are performed analytically on contour boundaries without gridding.

Supports **2D Euler**, **surface quasi-geostrophic (SQG)**, **single-layer quasi-geostrophic (QG)**, and **N-layer QG** dynamics on unbounded and doubly-periodic domains.

## Installation

```julia
using Pkg
Pkg.add("ContourDynamics")
```

Requires Julia 1.10 or later.

## Quick Start

```julia
using ContourDynamics, StaticArrays

# Create a circular vortex patch with 128 boundary nodes
R, N, pv = 1.0, 128, 2π
nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
contour = PVContour(nodes, pv)

# Set up problem with 2D Euler kernel on unbounded domain
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [contour])

# Evolve with RK4 time stepping and contour surgery
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams{Float64}()
evolve!(prob, stepper, params; nsteps=1000, surgery_interval=10)

# Diagnostics — computed analytically from contour geometry
energy(prob)            # kinetic energy
circulation(prob)       # total circulation
enstrophy(prob)         # enstrophy
angular_momentum(prob)  # angular momentum
```

## Features

### Physics Kernels

| Kernel | Green's Function | Use Case |
|--------|-----------------|----------|
| `EulerKernel()` | G(r) = -1/(2π) log(r) | 2D incompressible Euler |
| `QGKernel(Ld)` | G(r) = -1/(2π) K₀(r/Ld) | Single-layer QG with deformation radius Ld |
| `SQGKernel(δ)` | G(r) = -1/(2π√(r²+δ²)) | Surface QG (fractional Laplacian) |
| `MultiLayerQGKernel(Ld, C, H)` | Eigenmode decomposition | N-layer QG with coupling matrix C |

The Euler kernel uses an exact antiderivative for the segment integral (no quadrature). The QG kernel uses singular subtraction: the log singularity is handled analytically (same as Euler), and the smooth remainder K₀(r/Ld) + log(r/Ld) is integrated via 5-point Gauss-Legendre quadrature.

### Contour Surgery

Full implementation of the Dritschel surgery algorithm:

- **Node redistribution** (`remesh`) — maintains resolution along contour boundaries
- **Contour reconnection** — merges and splits contours when segments approach within a critical distance
- **Filament removal** — removes thin filaments below an area threshold

### Domains

- **`UnboundedDomain()`** — free-space Green's function
- **`PeriodicDomain(Lx, Ly)`** — doubly-periodic via Ewald summation with precomputed Fourier coefficients and lazy thread-safe caching

### Diagnostics

All diagnostics are computed from contour geometry using Green's theorem — no gridding required:

| Function | Description |
|----------|-------------|
| `vortex_area(c)` | Patch area (shoelace formula) |
| `centroid(c)` | Centroid of a contour |
| `ellipse_moments(c)` | Aspect ratio and orientation angle |
| `circulation(prob)` | Total circulation Γ = Σ qᵢ Aᵢ |
| `enstrophy(prob)` | Enstrophy Z = Σ qᵢ² Aᵢ |
| `energy(prob)` | Kinetic energy via double boundary integral |
| `angular_momentum(prob)` | Angular momentum |

### Time Integration

- **`RK4Stepper`** — classical 4th-order Runge-Kutta
- **`LeapfrogStepper`** — symplectic leapfrog (2nd-order)
- **`evolve!`** — main time loop with periodic surgery and buffer management

## Quasi-Geostrophic Example

```julia
using ContourDynamics, StaticArrays

# QG vortex with Rossby deformation radius Ld = 1.0
Ld = 1.0
nodes = [SVector(0.5*cos(2π*k/128), 0.5*sin(2π*k/128)) for k in 0:127]
prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, 2π)])

stepper = RK4Stepper(0.01, total_nodes(prob))
for step in 1:500
    timestep!(prob, stepper)
end
```

## Two-Layer QG Example

```julia
using ContourDynamics, StaticArrays, LinearAlgebra

Ld = SVector(1.0)                                  # deformation radius
H = SVector(0.5, 0.5)                              # layer depths
coupling = SMatrix{2,2}([-1.0, 1.0, 1.0, -1.0])   # 2-layer coupling matrix

kernel = MultiLayerQGKernel(Ld, coupling, H)

nodes = [SVector(0.5*cos(2π*k/100), 0.5*sin(2π*k/100)) for k in 0:99]

layer1 = ContourProblem(kernel, UnboundedDomain(), [PVContour(nodes, 2π)])
layer2 = ContourProblem(kernel, UnboundedDomain(), PVContour{Float64}[])
prob = MultiLayerContourProblem((layer1, layer2))

stepper = RK4Stepper(0.01, total_nodes(prob))
for step in 1:200
    timestep!(prob, stepper)
end
```

## Vortex Merger with Surgery

```julia
using ContourDynamics, StaticArrays

# Two co-rotating patches close enough to merge
R, sep, pv = 0.5, 0.9, 2π
c1 = PVContour([SVector(R*cos(θ) - sep/2, R*sin(θ)) for θ in range(0, 2π, 129)[1:128]], pv)
c2 = PVContour([SVector(R*cos(θ) + sep/2, R*sin(θ)) for θ in range(0, 2π, 129)[1:128]], pv)

prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams{Float64}()

for step in 1:500
    timestep!(prob, stepper)
    if step % 5 == 0
        surgery!(prob, params)
        resize_buffers!(stepper, prob)
    end
end

println("$(length(prob.contours)) contour(s) after merger")
```

## Package Extensions

ContourDynamics.jl provides optional integrations via Julia package extensions (requires Julia 1.10+):

| Extension | Trigger Package | Functionality |
|-----------|----------------|---------------|
| `ContourDynamicsDiffEqExt` | OrdinaryDiffEq | `to_ode_problem` — bridge to DifferentialEquations.jl solvers |
| `ContourDynamicsMakieExt` | Makie | `record_evolution` — animated contour evolution videos |
| `ContourDynamicsGeophysicalFlowsExt` | GeophysicalFlows | `contours_from_gridfield` / `gridfield_from_contours` — grid-contour conversion |
| `ContourDynamicsRecordedArraysExt` | RecordedArrays | `recorded_diagnostics` — time-series recording callback |

Load any extension by importing the trigger package alongside ContourDynamics:

```julia
using ContourDynamics, OrdinaryDiffEq

ode_prob = to_ode_problem(prob, (0.0, 10.0))
sol = solve(ode_prob, Tsit5())
```

## Method

Contour dynamics is a Lagrangian numerical method for inviscid, incompressible vortex flows. Instead of solving the vorticity equation on a grid, the method tracks the **boundaries of uniform PV patches**. The velocity of each boundary node is computed as a sum of boundary integrals over all contour segments:

```
u(x) = Σⱼ (qⱼ / 2π) ∮_Cⱼ G(|x - x'|) × dx'
```

where G is the appropriate Green's function (log for Euler, K₀ for QG). This approach is exact for piecewise-constant PV distributions and avoids numerical diffusion entirely.

Contour surgery (Dritschel, 1988) extends the method to long-time integrations by automatically handling topological changes — vortex merger, splitting, and filament removal — that would otherwise cause the contour to develop unresolvable complexity.

## References

- Dritschel, D. G. (1988). Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics. *J. Comput. Phys.*, 77(1), 240-266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D. G. (1989). Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows. *Comput. Phys. Rep.*, 10(3), 77-146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Dritschel, D. G. & Ambaum, M. H. P. (1997). A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields. *Q. J. R. Meteorol. Soc.*, 123(540), 1097-1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)

## Citation

If you use ContourDynamics.jl in your research, please cite our JOSS paper (forthcoming).

## License

MIT
