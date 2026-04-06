# Tutorial: Quasi-Geostrophic Dynamics

This tutorial covers single-layer QG, periodic domains with beta-plane PV staircases, and multi-layer QG dynamics.

## Physical Background

Quasi-geostrophic (QG) dynamics describes rotating stratified flows where the Rossby number is small. The key difference from 2D Euler is the **Rossby deformation radius** ``L_d``, which sets the scale at which rotation effects become important.

The QG Green's function is:

```math
G(r) = -\frac{1}{2\pi} K_0\!\left(\frac{r}{L_d}\right)
```

where ``K_0`` is the modified Bessel function of the second kind. Key properties:
- For ``r \ll L_d``: ``K_0(r/L_d) \approx -\log(r/L_d)`` — matches the Euler Green's function
- For ``r \gg L_d``: ``K_0(r/L_d) \sim \sqrt{\pi L_d/(2r)} \, e^{-r/L_d}`` — exponential decay

This means vortices smaller than ``L_d`` behave like Euler vortices, while larger vortices are screened by rotation.

## Single-Layer QG

```julia
using ContourDynamics
using StaticArrays

# QG vortex with deformation radius Ld = 2.0
Ld = 2.0
R, N, pv = 1.0, 128, 1.0

nodes = [SVector(R*cos(2π*i/N), R*sin(2π*i/N)) for i in 0:N-1]
prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, pv)])
```

!!! tip "GPU Support"
    To run this tutorial on GPU, add `using CUDA` and pass `dev=GPU()` when
    constructing the problem and stepper. All other code remains the same.

### Comparing Euler and QG

When ``L_d \gg R`` (patch radius), QG velocities approach the Euler limit. When ``L_d \lesssim R``, QG velocities are weaker due to rotational screening:

```julia
prob_euler = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])
prob_qg = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, pv)])

vel_euler = zeros(SVector{2, Float64}, N)
vel_qg = zeros(SVector{2, Float64}, N)
velocity!(vel_euler, prob_euler)
velocity!(vel_qg, prob_qg)

# Compare speeds at the first node
speed_euler = sqrt(vel_euler[1][1]^2 + vel_euler[1][2]^2)
speed_qg = sqrt(vel_qg[1][1]^2 + vel_qg[1][2]^2)
println("Euler speed: $speed_euler")
println("QG speed (Ld=$Ld): $speed_qg")
println("Ratio: $(speed_qg / speed_euler)")
```

### Evolving a QG Vortex

```julia
stepper = RK4Stepper(0.05, total_nodes(prob))
params = SurgeryParams(0.02, 0.01, 0.3, 1e-5, 10)
evolve!(prob, stepper, params; nsteps=2000)

# QG also conserves circulation and energy
println("Circulation: $(circulation(prob))")
println("Energy: $(energy(prob))")
```

## Periodic Domains and Beta Staircases

For geophysical applications, doubly-periodic domains are essential. ContourDynamics.jl uses **Ewald summation** to handle the periodic Green's function efficiently.

### Setting Up a Periodic Domain

```julia
# Domain: [-π, π) × [-π, π)
domain = PeriodicDomain(Float64(π), Float64(π))

# A vortex patch in the periodic domain
R = 0.3
nodes = [SVector(R*cos(2π*k/64), R*sin(2π*k/64)) for k in 0:63]
prob = ContourProblem(QGKernel(2.0), domain, [PVContour(nodes, 1.0)])
```

The Ewald cache is built automatically on first use. For custom accuracy, pre-build with `setup_ewald_cache!`:

```julia
# Higher accuracy: more Fourier modes and periodic images
setup_ewald_cache!(domain, QGKernel(2.0); n_fourier=16, n_images=4)
```

### Beta-Plane PV Staircase

The background PV gradient ``\beta y`` on a beta plane can be represented as a **PV staircase** — a set of horizontal spanning contours that discretize the continuous gradient:

```julia
T = Float64
L = 3.0
domain = PeriodicDomain(T(L))

# Discretize βy into 12 staircase steps
beta = T(1.0)
staircase = beta_staircase(beta, domain, 12; nodes_per_contour=64)

println("Number of spanning contours: $(length(staircase))")
println("PV jump per contour: $(staircase[1].pv)")
println("Is spanning: $(is_spanning(staircase[1]))")
```

Each spanning contour has a `wrap` vector that connects the last node back to the first node shifted by one period — this encodes the cross-domain topology.

### Beta Drift of a Cyclone

A cyclone on a beta plane drifts north-westward. We can simulate this by combining the PV staircase with a circular vortex patch:

```julia
# Circular vortex at the origin
R_vortex = 0.3
N_vortex = 64
vortex = PVContour(
    [SVector{2,T}(R_vortex * cos(2π*k/N_vortex), R_vortex * sin(2π*k/N_vortex))
     for k in 0:N_vortex-1],
    T(2π)
)

# Combine staircase + vortex
contours = vcat(staircase, [vortex])
prob = ContourProblem(QGKernel(T(1.0)), domain, contours)

# Evolve
stepper = RK4Stepper(T(0.005), total_nodes(prob))
params = SurgeryParams(T(0.02), T(0.01), T(0.3), T(1e-6), 401)

c0 = centroid(vortex)
evolve!(prob, stepper, params; nsteps=400)

# Find the vortex (largest non-spanning contour)
vortex_final = argmax(
    c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)),
    prob.contours
)
cf = centroid(prob.contours[vortex_final])
println("Vortex drift: Δx=$(cf[1] - c0[1]), Δy=$(cf[2] - c0[2])")
println("(Cyclones drift north-westward on a beta plane)")
```

## Multi-Layer QG

For ``N``-layer QG dynamics, the layers are coupled through interface deformation. The coupling is encoded in a **coupling matrix** that relates PV in each layer to the streamfunction.

### Two-Layer Setup

```julia
using LinearAlgebra

# Deformation radius of the baroclinic mode
Ld = SVector(1.5)
# Stretching operator with one barotropic (λ = 0) and one baroclinic mode
F = 1.0 / (2 * Ld[1]^2)
coupling = SMatrix{2,2}(-F, F, F, -F)

kernel = MultiLayerQGKernel(Ld, coupling)
println("Number of layers: $(nlayers(kernel))")
```

The constructor automatically eigen-decomposes the coupling matrix. Each eigenmode is evolved independently using either the Euler kernel (barotropic mode) or a QG kernel with the appropriate modal deformation radius.

### Creating a Multi-Layer Problem

```julia
R, N_nodes = 0.5, 100
nodes = [SVector(R*cos(2π*k/N_nodes), R*sin(2π*k/N_nodes)) for k in 0:N_nodes-1]

# Vortex in layer 1, no vortex in layer 2
layer1_contours = [PVContour(nodes, 2π)]
layer2_contours = PVContour{Float64}[]

prob = MultiLayerContourProblem(kernel, UnboundedDomain(),
                                (layer1_contours, layer2_contours))

println("Total nodes: $(total_nodes(prob))")
println("Layers: $(nlayers(prob))")
```

### Evolving the Multi-Layer System

```julia
stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 50)

E0 = energy(prob)
Γ0 = circulation(prob)

evolve!(prob, stepper, params; nsteps=500)

println("Energy change: $(abs(energy(prob) - E0) / abs(E0))")
println("Circulation change: $(abs(circulation(prob) - Γ0) / abs(Γ0))")
```

## Next Steps

- See complete runnable scripts in the [Examples](/examples) section
- Read the [Theory & Method](/theory) page for the mathematics behind Ewald summation and modal decomposition
- Check the [API Reference](/api) for all available functions
