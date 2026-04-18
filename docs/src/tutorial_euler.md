# Tutorial: 2D Euler Vortex Dynamics

This tutorial shows the basic ContourDynamics.jl workflow on a 2D Euler vortex patch. You will create a contour, build a problem, evolve it in time, and check a few diagnostics.

## Physical Background

In 2D Euler flow, the velocity is determined by the vorticity field. For a **vortex patch**, a region of uniform vorticity surrounded by irrotational flow, that velocity can be written as a contour integral around the patch boundary.

The Green's function is ``G(r) = -\frac{1}{2\pi} \log r``, and the velocity at a boundary node is:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

Each segment contribution is computed analytically, so this part of the method does not introduce quadrature error.

## Setting Up a Vortex Patch

We will start with a **Kirchhoff ellipse**, a standard test case that rotates steadily without changing shape.

```julia
using ContourDynamics

# Kirchhoff ellipse: semi-axes a > b, uniform PV jump
a, b = 2.0, 1.0   # aspect ratio = 2
N = 128            # boundary nodes
pv = 1.0           # potential vorticity jump

contour = elliptical_patch(a, b, N, pv)
```

The `elliptical_patch` helper creates a `PVContour` with evenly spaced boundary nodes and the given PV jump. Positive PV induces counterclockwise circulation.

## Creating the Problem

Create a `Problem` by specifying contours and a time step:

```julia
dt = 0.01
prob = Problem(; contours=[contour], dt=dt)
```

This high-level constructor chooses sensible defaults for the rest of the
simulation:

- `kernel=:euler`
- `domain=:unbounded`
- `stepper=:RK4`
- `surgery=:standard`

That is enough for many first runs. You only need to specify more options when
you want a different physical model, a periodic domain, or different surgery
settings.

!!! tip "GPU Support"
    To run this tutorial on GPU, add `using CUDA` and pass `dev=GPU()` when
    constructing the `Problem`. All other code remains the same.

You can check initial diagnostics right away:

```julia
area0 = vortex_area(contours(prob)[1])          # should be about π*a*b = 2π
circulation0 = circulation(prob)                # should be about pv * area0 = 2π
aspect_ratio0, angle0 = ellipse_moments(contours(prob)[1])
println("Area = $area0, Circulation = $circulation0, Aspect ratio = $aspect_ratio0")
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
    aspect_ratio, _ = ellipse_moments(contours(prob)[1])
    push!(aspect_ratios, aspect_ratio)
end

evolve!(prob; nsteps=5000, callbacks=[diagnostics_callback])
```

This callback receives the current `prob` and the current step number. It is a
convenient way to build time series without changing the solver itself.

## Verifying the Kirchhoff Solution

The Kirchhoff ellipse is a standard validation case. An elliptical vortex patch with semi-axes ``a`` and ``b`` and PV jump ``q`` rotates rigidly at angular velocity:

```math
\Omega = \frac{ab}{(a+b)^2} \, q
```

For our parameters (``a=2, b=1, q=1``), the predicted angular velocity is ``2/9 \approx 0.222``.

After evolution, you can verify:

```julia
# The aspect ratio should remain ≈ 2 (steady rotation, no deformation)
println("Final aspect ratio: $(aspect_ratios[end])")
println("Aspect ratio drift: $(abs(aspect_ratios[end] - 2.0))")

# Energy and circulation should stay nearly constant
rel_energy_change = abs(energies[end] - energies[1]) / abs(energies[1])
rel_circulation_change = abs(circulations[end] - circulations[1]) / abs(circulations[1])
println("Relative energy change: $rel_energy_change")
println("Relative circulation change: $rel_circulation_change")
```

## Computing Velocity at Arbitrary Points

You can evaluate the velocity field at any point, not just on contour nodes:

```julia
using StaticArrays

# Velocity at the origin (should be zero by symmetry for a centered patch)
v_origin = velocity(prob, SVector(0.0, 0.0))
println("Velocity at origin: $v_origin")

# Velocity at a point outside the patch
v_far = velocity(prob, SVector(5.0, 0.0))
println("Velocity at (5,0): $v_far")
```

## Next Steps

- Learn about [quasi-geostrophic dynamics](/tutorial_qg) with finite deformation radii
- See complete runnable scripts in the [Examples](/examples) section
- Read the [Theory overview](/theory) or jump straight to [Contour Dynamics](/theory/contour_dynamics) for mathematical details
