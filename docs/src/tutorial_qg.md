# Tutorial: Quasi-Geostrophic Dynamics

This tutorial introduces the main quasi-geostrophic workflows in ContourDynamics.jl: single-layer QG, periodic domains with beta-plane staircases, and multi-layer QG.

## Physical Background

Quasi-geostrophic (QG) dynamics describes rotating stratified flows at small Rossby number. Compared with 2D Euler, the main extra parameter is the **Rossby deformation radius** ``L_d``, which sets the scale where rotation and stratification matter.

With the sign convention used by `ContourDynamics.jl`, the QG scalar kernel in
the contour integral is:

```math
G(r) = \frac{1}{2\pi} K_0\!\left(\frac{r}{L_d}\right)
```

where ``K_0`` is the modified Bessel function of the second kind. Two useful limits are:
- For ``r \ll L_d``: ``K_0(r/L_d) \approx -\log(r/L_d)``, so the kernel behaves like Euler at small scales
- For ``r \gg L_d``: ``K_0(r/L_d)`` decays exponentially, so interactions are screened at large scales

This means vortices smaller than ``L_d`` behave like Euler vortices, while larger vortices are screened by rotation.

## Single-Layer QG

```julia
using ContourDynamics

# QG vortex with deformation radius Ld = 2.0
Ld = 2.0
R, N, pv = 1.0, 128, 1.0

contour = circular_patch(R, N, pv)
prob = Problem(; contours=[contour], dt=0.05, kernel=:qg, Ld=Ld)
```

!!! note "GPU Support"
    GPU velocity evaluation is currently available for single-layer Euler, QG,
    and SQG on unbounded or periodic domains, and for direct multi-layer QG on
    unbounded or periodic domains.

### Comparing Euler and QG

When ``L_d \gg R`` (patch radius), QG velocities approach the Euler limit. When ``L_d \lesssim R``, QG velocities are weaker due to rotational screening:

```julia
using StaticArrays   # needed for SVector velocity buffers below

contour = circular_patch(R, N, pv)
prob_euler = Problem(; contours=[contour], dt=0.05)
prob_qg    = Problem(; contours=[contour], dt=0.05, kernel=:qg, Ld=Ld)

# Low-level velocity! call with explicit SVector buffers
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
evolve!(prob; nsteps=2000)

# QG also conserves circulation and energy
println("Circulation: $(circulation(prob))")
println("Energy: $(energy(prob))")
```

## Periodic Domains and Beta Staircases

For geophysical applications, doubly-periodic domains are essential. ContourDynamics.jl uses **Ewald summation** to handle the periodic Green's function efficiently.

### Setting Up a Periodic Domain

```julia
# A vortex patch in a periodic domain [-π, π) × [-π, π)
R = 0.3
contour = circular_patch(R, 64, 1.0)
L = Float64(pi)
prob = Problem(; contours=[contour], dt=0.05,
               kernel=:qg, Ld=2.0, domain=:periodic, Lx=L, Ly=L)
```

The Ewald cache is built automatically on first use. For custom accuracy, pre-build with `setup_ewald_cache!`:

```julia
# Higher accuracy: more Fourier modes and periodic images
setup_ewald_cache!(domain(prob), kernel(prob); n_fourier=16, n_images=4)
```

### Beta-Plane PV Staircase

The background PV gradient ``\beta y`` on a beta plane can be represented as a **PV staircase**, a set of horizontal spanning contours that discretize the continuous gradient:

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

Each spanning contour has a `wrap` vector that connects the last node back to the first node shifted by one period. That is how the package represents contours that cross the periodic boundary.

### Beta Drift of a Cyclone

A cyclone on a beta plane drifts north-westward. We can simulate this by combining the PV staircase with a circular vortex patch:

```julia
# Circular vortex at the origin
vortex = circular_patch(T(0.3), 64, T(2π))

# Combine staircase + vortex
all_contours = vcat(staircase, [vortex])
prob = Problem(; contours=all_contours, dt=T(0.005),
               kernel=:qg, Ld=T(1.0), domain=:periodic, Lx=T(L), Ly=T(L))

c0 = centroid(vortex)
evolve!(prob; nsteps=400)

# Find the vortex (largest non-spanning contour)
vortex_final = argmax(
    c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)),
    contours(prob)
)
cf = centroid(contours(prob)[vortex_final])
println("Vortex drift: dx=$(cf[1] - c0[1]), dy=$(cf[2] - c0[2])")
println("(Cyclones drift north-westward on a beta plane)")
```

## Multi-Layer QG

For ``N``-layer QG dynamics, the layers are coupled through interface deformation. The coupling is encoded in a **coupling matrix** that relates PV in each layer to the streamfunction.

### Two-Layer Setup

```julia
using LinearAlgebra
using StaticArrays   # needed for SVector/SMatrix coupling matrix

# Deformation radius of the baroclinic mode
Ld = SVector(1.5)
# Stretching operator with one barotropic mode (eigenvalue 0)
# and one baroclinic mode
F = 1.0 / (2 * Ld[1]^2)
coupling = SMatrix{2,2}(-F, F, F, -F)

kernel = MultiLayerQGKernel(Ld, coupling)
println("Number of layers: $(nlayers(kernel))")
```

The constructor automatically eigen-decomposes the coupling matrix. Each eigenmode is evolved independently using either the Euler kernel (barotropic mode) or a QG kernel with the appropriate modal deformation radius.

### Creating a Multi-Layer Problem

```julia
R, N_nodes = 0.5, 100

# Vortex in layer 1, no vortex in layer 2
layer1_contours = [circular_patch(R, N_nodes, 2π)]
layer2_contours = PVContour{Float64}[]

prob = Problem(; layers=(layer1_contours, layer2_contours),
               dt=0.01, kernel=:multilayer_qg,
               Ld=Ld, coupling=coupling)

println("Total nodes: $(total_nodes(prob))")
println("Layers: $(nlayers(prob))")
```

### Evolving the Multi-Layer System

```julia
energy0 = energy(prob)
circulation0 = circulation(prob)

evolve!(prob; nsteps=500)

println("Energy change: $(abs(energy(prob) - energy0) / abs(energy0))")
println("Circulation change: $(abs(circulation(prob) - circulation0) / abs(circulation0))")
```

## Next Steps

- See complete runnable scripts in the [Examples](/examples) section
- Read the [Theory overview](/theory), [Ewald Summation](/theory/ewald_summation), or [Multi-Layer QG](/theory/multilayer_qg) for the relevant mathematics
- Check the [API Reference](/api) for all available functions
