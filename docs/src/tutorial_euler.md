# Tutorial: 2D Euler Vortex Dynamics

This tutorial walks through simulating a 2D Euler vortex patch — from setup to diagnostics. By the end you'll understand how to create contour problems, evolve them in time, and verify conservation laws.

## Physical Background

In 2D Euler dynamics, an inviscid, incompressible fluid has its velocity determined entirely by the vorticity field. For a **vortex patch** — a region of uniform vorticity ``q`` surrounded by irrotational flow — the velocity at any point can be computed as a contour integral around the patch boundary.

The Green's function is ``G(r) = -\frac{1}{2\pi} \log r``, and the velocity at a boundary node is:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

Each segment of this integral is computed **analytically** — no quadrature error.

## Setting Up a Vortex Patch

Let's create a **Kirchhoff ellipse** — an elliptical vortex patch that rotates steadily in 2D Euler flow without changing shape.

```julia
using ContourDynamics
using StaticArrays

# Kirchhoff ellipse: semi-axes a > b, uniform PV jump
a, b = 2.0, 1.0   # aspect ratio = 2
N = 128            # boundary nodes
pv = 1.0           # potential vorticity jump

nodes = [SVector(a*cos(2π*i/N), b*sin(2π*i/N)) for i in 0:N-1]
contour = PVContour(nodes, pv)
```

The `PVContour` stores the boundary nodes and the PV jump across the contour. Positive PV induces counterclockwise circulation.

## Creating the Problem

Combine a kernel, a domain, and one or more contours into a `ContourProblem`:

```julia
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [contour])
```

!!! tip "GPU Support"
    To run this tutorial on GPU, add `using CUDA` and pass `dev=GPU()` when
    constructing the problem and stepper. All other code remains the same.

You can check initial diagnostics right away:

```julia
A = vortex_area(contour)       # should be ≈ π*a*b = 2π
Γ = circulation(prob)          # should be ≈ pv * A = 2π
λ, θ = ellipse_moments(contour)  # aspect ratio ≈ 2.0, angle ≈ 0
println("Area = $A, Circulation = $Γ, Aspect ratio = $λ")
```

## Time Integration

Set up an RK4 time stepper and surgery parameters:

```julia
dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))

# Surgery parameters:
#   delta     = 0.01   proximity threshold for reconnection
#   mu        = 0.005  minimum segment length
#   Delta_max = 0.2    maximum segment length
#   area_min  = 1e-6   filament removal threshold
#   n_surgery = 50     steps between surgery passes
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 50)
```

## Tracking Conservation Laws

Use callbacks to record diagnostics at each time step:

```julia
times = Float64[]
energies = Float64[]
circulations = Float64[]
aspect_ratios = Float64[]

function diagnostics_callback(prob, step)
    push!(times, step * dt)
    push!(energies, energy(prob))
    push!(circulations, circulation(prob))
    λ, _ = ellipse_moments(prob.contours[1])
    push!(aspect_ratios, λ)
end

evolve!(prob, stepper, params; nsteps=5000, callbacks=[diagnostics_callback])
```

## Verifying the Kirchhoff Solution

The Kirchhoff ellipse is a key validation case. An elliptical vortex patch with semi-axes ``a`` and ``b`` and PV jump ``q`` rotates rigidly at angular velocity:

```math
\Omega = \frac{ab}{(a+b)^2} \, q
```

For our parameters (``a=2, b=1, q=1``): ``\Omega = 2/9 \approx 0.222``.

After evolution, you can verify:

```julia
# The aspect ratio should remain ≈ 2 (steady rotation, no deformation)
println("Final aspect ratio: $(aspect_ratios[end])")
println("Aspect ratio drift: $(abs(aspect_ratios[end] - 2.0))")

# Energy and circulation should be conserved
ΔE = abs(energies[end] - energies[1]) / abs(energies[1])
ΔΓ = abs(circulations[end] - circulations[1]) / abs(circulations[1])
println("Relative energy change: $ΔE")
println("Relative circulation change: $ΔΓ")
```

## Computing Velocity at Arbitrary Points

You can evaluate the velocity field at any point, not just on contour nodes:

```julia
# Velocity at the origin (should be zero by symmetry for a centered patch)
v_origin = velocity(prob, SVector(0.0, 0.0))
println("Velocity at origin: $v_origin")

# Velocity at a point outside the patch
v_far = velocity(prob, SVector(5.0, 0.0))
println("Velocity at (5,0): $v_far")
```

## Next Steps

- Learn about [quasi-geostrophic dynamics](@ref "Tutorial: Quasi-Geostrophic Dynamics") with finite deformation radii
- See complete runnable scripts in the [Examples](@ref) section
- Read the [Theory & Method](@ref) page for mathematical details
