# Tutorial: Quasi-Geostrophic Dynamics

## Single-Layer QG

The QG kernel replaces the Euler log(r) Green's function with K₀(r/Ld),
where Ld is the Rossby deformation radius.

```julia
using ContourDynamics
using StaticArrays

# QG vortex with deformation radius Ld = 2.0
Ld = 2.0
R, N, pv = 1.0, 128, 1.0

nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.05, total_nodes(prob))
params = SurgeryParams(0.02, 0.01, 0.3, 1e-5, 10)
evolve!(prob, stepper, params; nsteps=2000)
```

## Multi-Layer QG

For N-layer QG, specify the coupling matrix and layer thicknesses:

```julia
using StaticArrays

# 2-layer setup
Ld = SVector(1.5)  # deformation radius at interface
H = SVector(1.0, 1.0)
F = 1.0 / Ld[1]^2
coupling = SMatrix{2,2}(1.0 + F, -F, -F, 1.0 + F)

kernel = MultiLayerQGKernel(Ld, coupling, H)
```

## Periodic Domains

For doubly-periodic domains, use `PeriodicDomain`. The velocity computation
automatically switches to Ewald summation:

```julia
prob = ContourProblem(QGKernel(2.0), PeriodicDomain(π, π), [PVContour(nodes, pv)])
```
