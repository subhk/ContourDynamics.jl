# Tutorial: 2D Euler Vortex Patch

This tutorial walks through a basic 2D Euler vortex dynamics simulation.

## Setting Up a Vortex Patch

```julia
using ContourDynamics
using StaticArrays

# Create an elliptical vortex patch (Kirchhoff ellipse)
a, b = 2.0, 1.0  # semi-axes
N = 128           # number of boundary nodes
pv = 1.0          # potential vorticity jump

nodes = [SVector(a*cos(2π*i/N), b*sin(2π*i/N)) for i in 0:N-1]
contour = PVContour(nodes, pv)
```

## Creating the Problem

```julia
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [contour])
```

## Time Integration

```julia
dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 50)

evolve!(prob, stepper, params; nsteps=5000)
```

## Diagnostics

All diagnostics are computed analytically from the contour geometry:

```julia
E = energy(prob)                    # kinetic energy
Z = enstrophy(prob)                 # enstrophy
Γ = circulation(prob)               # total circulation
A = vortex_area(prob.contours[1])   # patch area
λ, θ = ellipse_moments(prob.contours[1])  # aspect ratio, orientation
```

## The Kirchhoff Ellipse

A key validation: an elliptical vortex patch in 2D Euler rotates steadily without
changing shape. The angular velocity is:

```math
\Omega = \frac{ab}{(a+b)^2} q
```

where `a, b` are the semi-axes and `q` is the PV jump.
