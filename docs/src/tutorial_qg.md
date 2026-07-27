# Tutorial: Quasi-Geostrophic Dynamics

This tutorial introduces the main quasi-geostrophic workflows in ContourDynamics.jl: single-layer QG, periodic domains with beta-plane staircases, and multi-layer QG.

Suggested reading order:

1. start with the single-layer QG example
2. then look at the periodic-domain example
3. finally move to the multi-layer example

Each section uses the same overall workflow:

- build contours
- construct a `Problem`
- evolve it in time
- inspect diagnostics or geometry

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

```@repl tutorial_qg_single
using ContourDynamics

# QG vortex with deformation radius Ld = 2.0
Ld = 2.0
R, N, pv = 1.0, 32, 1.0

contour = circular_patch(R, N, pv)
prob = Problem(; contours=[contour], dt=0.05, kernel=:qg, Ld=Ld)

println("Nodes: $(total_nodes(prob))")
println("Circulation: $(round(circulation(prob); digits=6))");
```

!!! note "GPU Support"
    To run these QG workflows on GPU, add `using CUDA` and pass `dev=GPU()`
    when constructing the `Problem`.
    
    Single-layer Euler, QG, and SQG (unbounded or periodic), beta-plane QG
    (periodic), and multi-layer QG all support device-resident velocity,
    RK4/leapfrog timestepping, periodic wrapping, surgery, and diagnostics.
    Surgery on unbounded domains runs entirely on the device; periodic surgery
    materializes at the host boundary, runs the CPU surgery pass, and reloads
    the device state.

At this stage, the only new ingredient compared with the Euler tutorial is the
deformation radius `Ld`. Everything else about the high-level workflow is the
same.

### Comparing Euler and QG

When ``L_d \gg R`` (patch radius), QG velocities approach the Euler limit. When ``L_d \lesssim R``, QG velocities are weaker due to rotational screening:

```@repl tutorial_qg_compare
using StaticArrays   # needed for SVector velocity buffers below
using ContourDynamics

Ld = 2.0
R, N, pv = 1.0, 32, 1.0
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
println("Euler speed: $(round(speed_euler; digits=6))")
println("QG speed (Ld=$Ld): $(round(speed_qg; digits=6))")
println("Ratio: $(round(speed_qg / speed_euler; digits=6))");
```

This comparison illustrates the role of `Ld`:

- large `Ld` means weak screening, so QG looks close to Euler
- smaller `Ld` means stronger screening, so distant interactions weaken

### Evolving a QG Vortex

```@repl tutorial_qg_single
evolve!(prob; nsteps=5)

# QG also conserves circulation and energy
println("Circulation: $(round(circulation(prob); digits=6))")
println("Energy: $(round(energy(prob); digits=6))");
```

## Periodic Domains and Beta Staircases

For geophysical applications, doubly-periodic domains are essential. ContourDynamics.jl uses **Ewald summation** to handle the periodic Green's function efficiently.

### Setting Up a Periodic Domain

```@repl tutorial_qg_periodic
using ContourDynamics

# A vortex patch in a periodic domain [-π, π) × [-π, π)
R = 0.3
contour = circular_patch(R, 24, 1.0)
L = Float64(pi)
prob = Problem(; contours=[contour], dt=0.05,
               kernel=:qg, Ld=2.0, domain=:periodic, Lx=L, Ly=L)

println("Domain half-widths: ($(L), $(L))")
println("Nodes: $(total_nodes(prob))");
```

The Ewald cache is built automatically on first use. For custom accuracy, pre-build with `setup_ewald_cache!`:

```@repl tutorial_qg_periodic
# Higher accuracy: more Fourier modes and periodic images
setup_ewald_cache!(domain(prob), kernel(prob); n_fourier=16, n_images=4)
println("Ewald cache configured with n_fourier=16 and n_images=4");
```

In most cases, `setup_ewald_cache!` does not need to be called explicitly. The
default cache settings are adequate for typical problem sizes. Manual setup is
primarily relevant when higher periodic-kernel accuracy is required.

### Beta-Plane PV Staircases

The package includes a `beta_staircase` helper for constructing horizontal
spanning contours in a periodic domain. With [`BetaPlaneQGKernel`](@ref), these
contours represent material regular-PV interfaces on a beta plane. The helper
uses the Lam-Dritschel mid-step placement: `n_beta` contours at
`-Ly + (k - 1/2) * 2Ly / n_beta`, each with jump `beta * 2Ly / n_beta`.
The kernel keeps a reference copy of the initial straight staircase, subtracts
that reference from the contour sum, then adds the analytic zonal correction for
`reference staircase - beta*y`.

```@repl tutorial_qg_staircase
using ContourDynamics

T = Float64
L = 3.0
domain = PeriodicDomain(T(L))

# Construct material full-PV jumps for beta-plane contour dynamics
beta = T(1.0)
staircase = beta_staircase(beta, domain, 8; nodes_per_contour=16)
vortex = circular_patch(T(0.3), 32, T(2π))

prob = Problem(; contours=vcat(staircase, [vortex]),
               dt=T(0.005),
               kernel=:beta_plane_qg,
               beta=beta,
               Ld=T(1.0),
               domain=:periodic,
               Lx=L,
               Ly=L,
               surgery=:none)

println("Number of spanning contours: $(length(staircase))")
println("PV jump per contour: $(staircase[1].pv)")
println("Kernel: $(kernel(prob))");
```

Each spanning contour has a `wrap` vector that connects the last node back to the first node shifted by one period. That is how the package represents contours that cross the periodic boundary.

That means the staircase contours are not ordinary closed loops sitting inside
the box. They represent interfaces that continue across the periodic boundary.

The beta-plane setup combines three features:

- a periodic Green's function through the Ewald machinery
- active spanning-contour geometry for material beta contours
- reference straight beta staircase subtraction in `BetaPlaneQGKernel`
- analytic contour-only correction for the continuous `beta*y` term

The analytic correction is the finite-`n_beta` sawtooth residual from the
discretized `q_r - beta*y` inversion. It keeps the implementation contour-based
without replacing velocity evaluation by a grid solve.

## Multi-Layer QG

For ``N``-layer QG dynamics, the layers are coupled through interface deformation. The coupling is encoded in a **coupling matrix** that relates PV in each layer to the streamfunction.

### Two-Layer Setup

```@repl tutorial_qg_multilayer
using ContourDynamics
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
println("Modal deformation radius: $(Ld[1])");
```

The constructor automatically eigen-decomposes the coupling matrix. Each eigenmode is evolved independently using either the Euler kernel (barotropic mode) or a QG kernel with the appropriate modal deformation radius.

The package handles the eigen-decomposition and the projection back to physical
layers internally.

### Creating a Multi-Layer Problem

```@repl tutorial_qg_multilayer
using ContourDynamics
using StaticArrays

R, N_nodes = 0.5, 32

# Vortex in layer 1, no vortex in layer 2
layer1_contours = [circular_patch(R, N_nodes, 2π)]
layer2_contours = PVContour{Float64}[]

prob = Problem(; layers=(layer1_contours, layer2_contours),
               dt=0.01, kernel=:multilayer_qg,
               Ld=Ld, coupling=coupling)

println("Total nodes: $(total_nodes(prob))")
println("Layers: $(nlayers(prob))");
```

This is the multi-layer analogue of the earlier single-layer `Problem` call:
the only extra pieces are the `layers` tuple, `Ld`, and the coupling matrix.

### Evolving the Multi-Layer System

```@repl tutorial_qg_multilayer
energy0 = energy(prob)
circulation0 = circulation(prob)

evolve!(prob; nsteps=5)

println("Energy change: $(round(abs(energy(prob) - energy0) / abs(energy0); digits=8))")
println("Circulation change: $(round(abs(circulation(prob) - circulation0) / abs(circulation0); digits=8))");
```

## Next Steps

- [Examples](examples.md) for complete runnable scripts
- [Theory overview](theory.md), [Ewald Summation](theory/ewald_summation.md), or [Multi-Layer QG](theory/multilayer_qg.md) for the underlying mathematics
- [API overview](api.md) for the relevant functions and types
