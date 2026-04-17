# ContourDynamics.jl

[![CI](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/subhk/ContourDynamics.jl/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/ContourDynamics.jl/dev/)

Lagrangian vortex dynamics via **contour dynamics** and **contour surgery** in Julia.

Track boundaries of piecewise-constant potential vorticity (PV) patches with analytical boundary integrals — no grids, no numerical diffusion.

**Supported physics:** 2D Euler, surface quasi-geostrophic (SQG), single-layer QG, and N-layer QG on unbounded and doubly-periodic domains.

## Key Features

- **Exact segment integrals** — closed-form antiderivatives for Euler and SQG; singular subtraction + Gauss-Legendre quadrature for QG
- **Contour surgery** — automatic remeshing, topological reconnection, and filament removal for long-time stability ([Dritschel, 1988](https://doi.org/10.1016/0021-9991(88)90165-9))
- **Periodic domains** — Ewald summation with precomputed, thread-safe caching
- **Multi-layer QG** — eigenmode decomposition of the coupling matrix for N-layer baroclinic dynamics
- **GPU acceleration** — CUDA-accelerated O(N^2) velocity via KernelAbstractions.jl
- **Treecode** — O(N log N) velocity evaluation for large problems
- **Analytical diagnostics** — energy, circulation, enstrophy, angular momentum, and vortex geometry from contour integrals

## Installation

```julia
using Pkg
Pkg.add("ContourDynamics")
```

Requires Julia 1.10 or later.

## Quick Start

```julia
using ContourDynamics, StaticArrays

# Circular vortex patch: 128 nodes, radius 1, PV jump 2pi
R, N, pv = 1.0, 128, 2pi
nodes = [SVector(R * cos(2pi * k / N), R * sin(2pi * k / N)) for k in 0:N-1]

prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.002, 0.01, 0.2, 1e-6, 10)

evolve!(prob, stepper, params; nsteps=1000)

# Analytical diagnostics — no gridding
energy(prob)
circulation(prob)
enstrophy(prob)
```

## Physics Kernels

| Kernel | Scalar kernel | Use case |
|--------|-----------------|----------|
| `EulerKernel()` | -1/(2pi) log(r) | 2D incompressible Euler |
| `QGKernel(Ld)` | 1/(2pi) K_0(r/Ld) | Single-layer QG with deformation radius Ld |
| `SQGKernel(delta)` | -1/(2pi sqrt(r^2 + delta^2)) | Surface QG (regularized fractional Laplacian) |
| `MultiLayerQGKernel(Ld, C)` | Eigenmode decomposition | N-layer QG with coupling matrix C |

## Examples

### Vortex merger

Two co-rotating patches merge via contour surgery when their separation is below the critical distance (~3.3R):

```julia
using ContourDynamics, StaticArrays

R, N, pv = 0.5, 128, 2pi
circle(cx, cy) = [SVector(cx + R*cos(2pi*k/N), cy + R*sin(2pi*k/N)) for k in 0:N-1]

c1 = PVContour(circle(-0.45, 0.0), pv)
c2 = PVContour(circle(+0.45, 0.0), pv)

prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.005, 0.02, 0.2, 1e-6, 5)

evolve!(prob, stepper, params; nsteps=500)
println("$(length(prob.contours)) contour(s) after merger")
```

### Two-layer QG

Upper-layer vortex in a two-layer baroclinic system:

```julia
using ContourDynamics, StaticArrays, LinearAlgebra

Ld = SVector(1.0)
F = 1.0 / (2 * Ld[1]^2)
coupling = SMatrix{2,2}(-F, F, F, -F)
kernel = MultiLayerQGKernel(Ld, coupling)

nodes = [SVector(0.5*cos(2pi*k/100), 0.5*sin(2pi*k/100)) for k in 0:99]

prob = MultiLayerContourProblem(
    kernel, UnboundedDomain(),
    ([PVContour(nodes, 2pi)], PVContour{Float64}[])
)

stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 201)
evolve!(prob, stepper, params; nsteps=200)
```

See the [`examples/`](examples/) directory for complete scripts with JLD2 output (vortex merger, filamentation, beta-plane drift, two-layer QG).

## GPU Acceleration

GPU velocity evaluation is currently available for the Euler and SQG kernels on
unbounded domains. Pass `dev=GPU()` — no other code changes needed:

```julia
using ContourDynamics, CUDA

prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours; dev=GPU())
stepper = RK4Stepper(0.01, total_nodes(prob); dev=GPU())
evolve!(prob, stepper, params; nsteps=1000)
```

Surgery stays on CPU. Requires NVIDIA GPU with CUDA.jl v5+.

## Package Extensions

Optional integrations loaded on demand (Julia 1.10+):

| Extension | Trigger package | Provides |
|-----------|----------------|----------|
| `ContourDynamicsDiffEqExt` | OrdinaryDiffEq | `to_ode_problem` — DifferentialEquations.jl bridge |
| `ContourDynamicsMakieExt` | Makie | `record_evolution` — animated contour videos |
| `ContourDynamicsRecordedArraysExt` | RecordedArrays | `recorded_diagnostics` — time-series callbacks |
| `ContourDynamicsJLD2Ext` | JLD2 | `save_snapshot` / `load_snapshot` / `jld2_recorder` — checkpointing |

## Documentation

Full documentation is available at [subhk.github.io/ContourDynamics.jl](https://subhk.github.io/ContourDynamics.jl/dev/), including tutorials, API reference, and mathematical background.

## References

- Zabusky, N.J., Hughes, M.H. & Roberts, K.V. (1979). Contour dynamics for the Euler equations in two dimensions. *J. Comput. Phys.* **30**(1), 96-106. [doi:10.1016/0021-9991(79)90089-5](https://doi.org/10.1016/0021-9991(79)90089-5)
- Dritschel, D.G. (1988). Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics. *J. Comput. Phys.* **77**(1), 240-266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). Contour dynamics and contour surgery. *Comput. Phys. Rep.* **10**(3), 77-146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995). Surface quasi-geostrophic dynamics. *J. Fluid Mech.* **282**, 1-20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)
