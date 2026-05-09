# Tutorial: 2D Euler Vortex Dynamics

This tutorial shows the basic ContourDynamics.jl workflow on a 2D Euler vortex patch. The workflow creates a contour, builds a problem, evolves it in time, and checks a few diagnostics.

## Physical Background

In 2D Euler flow, the velocity is determined by the vorticity field. For a **vortex patch**, a region of uniform vorticity surrounded by irrotational flow, that velocity can be written as a contour integral around the patch boundary.

The Green's function is ``G(r) = -\frac{1}{2\pi} \log r``, and the velocity at a boundary node is:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

Straight segment contributions are computed analytically. When remeshing
supplies nonzero Dritschel endpoint curvature, the same contour integral is
evaluated on the cubic interpolation arc with fixed Gauss-Legendre quadrature.

## Setting Up a Vortex Patch

The example starts with a **Kirchhoff ellipse**, a standard test case that
rotates steadily without changing shape.

```@example tutorial_euler
using ContourDynamics

# Kirchhoff ellipse: semi-axes a > b, uniform PV jump
a, b = 2.0, 1.0   # aspect ratio = 2
N = 32             # boundary nodes (kept small so docs build quickly)
pv = 1.0           # potential vorticity jump

contour = elliptical_patch(a, b, N, pv);
```

The `elliptical_patch` helper creates a `PVContour` with evenly spaced boundary nodes and the given PV jump. Positive PV induces counterclockwise circulation.

## Creating the Problem

Create a `Problem` by specifying contours and a time step:

```@example tutorial_euler
dt = 0.01
prob = Problem(; contours=[contour], dt=dt);
```

This high-level constructor chooses sensible defaults for the rest of the
simulation:

- `kernel=:euler`
- `domain=:unbounded`
- `stepper=:RK4`
- `surgery=:standard`

These defaults are sufficient for the standard unbounded Euler workflow.
Additional options are only needed for a different physical model, a periodic
domain, or different surgery settings.

!!! tip "GPU Support"
    To run this tutorial on GPU, add `using CUDA` and pass `dev=GPU()` when
    constructing the `Problem`. All other code remains the same.

Initial diagnostics:

```@repl tutorial_euler
initial_contours = materialize_contours(prob)
area0 = vortex_area(initial_contours[1])        # should be about π*a*b = 2π
circulation0 = circulation(prob)                # should be about pv * area0 = 2π
aspect_ratio0, angle0 = ellipse_moments(initial_contours[1])
println("Area = $(round(area0; digits=6))")
println("Circulation = $(round(circulation0; digits=6))")
println("Aspect ratio = $(round(aspect_ratio0; digits=6))");
```

## Tracking Conservation Laws

Use callbacks to record diagnostics at each time step:

```@example tutorial_euler
times = Float64[]
energies = Float64[]
circulations = Float64[]
aspect_ratios = Float64[]

function diagnostics_callback(prob, step)
    push!(times, step * dt)
    push!(energies, energy(prob))
    push!(circulations, circulation(prob))
    snapshot = materialize_contours(prob)
    aspect_ratio, _ = ellipse_moments(snapshot[1])
    push!(aspect_ratios, aspect_ratio)
end

evolve!(prob; nsteps=5, callbacks=[diagnostics_callback])

println("Recorded steps: $(length(times))")
println("Final time: $(round(times[end]; digits=3))");
```

This callback receives the current `prob` and the current step number. It
provides a simple way to record time series without modifying the solver.

## Verifying the Kirchhoff Solution

The Kirchhoff ellipse is a standard validation case. An elliptical vortex patch with semi-axes ``a`` and ``b`` and PV jump ``q`` rotates rigidly at angular velocity:

```math
\Omega = \frac{ab}{(a+b)^2} \, q
```

For our parameters (``a=2, b=1, q=1``), the predicted angular velocity is ``2/9 \approx 0.222``.

After evolution, the following checks verify the solution quality:

```@repl tutorial_euler
# The aspect ratio should remain ≈ 2 (steady rotation, no deformation)
println("Final aspect ratio: $(round(aspect_ratios[end]; digits=6))")
println("Aspect ratio drift: $(round(abs(aspect_ratios[end] - 2.0); digits=6))")

# Energy and circulation should stay nearly constant
rel_energy_change = abs(energies[end] - energies[1]) / abs(energies[1])
rel_circulation_change = abs(circulations[end] - circulations[1]) / abs(circulations[1])
println("Relative energy change: $(round(rel_energy_change; digits=8))")
println("Relative circulation change: $(round(rel_circulation_change; digits=8))");
```

## Computing Velocity at Arbitrary Points

The velocity field can be evaluated at arbitrary points, not only on contour
nodes:

```@repl tutorial_euler
using StaticArrays

# Velocity at the origin (should be zero by symmetry for a centered patch)
v_origin = velocity(prob, SVector(0.0, 0.0))
println("Velocity at origin: $v_origin")

# Velocity at a point outside the patch
v_far = velocity(prob, SVector(5.0, 0.0))
println("Velocity at (5,0): $v_far");
```

## Next Steps

- [QG tutorial](tutorial_qg.md) for finite deformation-radius dynamics
- [Examples](examples.md) for complete runnable scripts
- [Theory overview](theory.md) or [Contour Dynamics](theory/contour_dynamics.md) for mathematical details
