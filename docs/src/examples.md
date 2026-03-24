# Examples

Complete runnable scripts demonstrating ContourDynamics.jl capabilities. Full scripts with file output are available in the [`examples/`](https://github.com/subhk/ContourDynamics.jl/tree/main/examples) directory.

## Vortex Merger

Two co-rotating circular vortex patches placed close enough to merge via contour surgery. When the separation is less than about 3.3 radii, the patches exchange filaments and eventually merge into a single vortex.

```julia
using ContourDynamics
using StaticArrays

T = Float64
N = 128          # nodes per contour
R = 0.5          # patch radius
sep = 1.8 * R    # centre-to-centre separation (< 3.3R triggers merger)
pv = 2π          # uniform PV jump

# Two circular patches offset along x
function circular_nodes(cx, cy, R, N)
    [SVector{2,T}(cx + R * cos(2π * k / N),
                  cy + R * sin(2π * k / N)) for k in 0:(N-1)]
end

c1 = PVContour(circular_nodes(-sep / 2, 0.0, R, N), pv)
c2 = PVContour(circular_nodes(+sep / 2, 0.0, R, N), pv)

prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])

dt = 0.01
surgery_params = SurgeryParams(T(0.01), T(0.005), T(0.2), T(1e-6), 5)
stepper = RK4Stepper(dt, total_nodes(prob))

Γ0 = circulation(prob)
evolve!(prob, stepper, surgery_params; nsteps=500)

println("Final: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")
println("Circulation conserved: |ΔΓ/Γ₀| = $(abs(circulation(prob) - Γ0) / abs(Γ0))")
```

## Filamentation

An elliptical vortex patch with high aspect ratio sheds thin filaments that are automatically removed by surgery. This demonstrates the interplay between the Kirchhoff rotation and the surgery algorithm's filament removal.

```julia
using ContourDynamics
using StaticArrays

N = 200
a, b = 1.0, 0.3  # semi-axes (aspect ratio ≈ 3.3)
pv = 2π

nodes = [SVector(a * cos(2π * k / N), b * sin(2π * k / N)) for k in 0:(N-1)]
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.005, total_nodes(prob))
surgery_params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 10)

A0 = vortex_area(prob.contours[1])
evolve!(prob, stepper, surgery_params; nsteps=1000)

println("Final: $(length(prob.contours)) contour(s)")
println("Area of largest contour: $(maximum(c -> abs(vortex_area(c)), prob.contours))")
println("Original area: $A0")
```

## QG Vortex with Deformation Radius

A circular vortex patch evolving under quasi-geostrophic dynamics with a finite Rossby deformation radius. The QG dynamics screens the far-field velocity, so the vortex rotates more slowly than its Euler counterpart.

```julia
using ContourDynamics
using StaticArrays

R, Ld, pv = 0.5, 1.0, 2π
N = 128

nodes = [SVector(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.01, total_nodes(prob))

# Track the centroid — a circular patch should remain centered
c0 = centroid(prob.contours[1])
for step in 1:500
    timestep!(prob, stepper)
end

c1 = centroid(prob.contours[1])
println("Centroid drift: ($(c1[1] - c0[1]), $(c1[2] - c0[2]))")
println("Area: $(vortex_area(prob.contours[1]))")
```

## Beta-Plane Vortex Drift

A cyclonic vortex on a beta plane (``\beta y`` background PV gradient) drifts north-westward due to the planetary vorticity gradient. The background gradient is represented as a PV staircase of spanning contours on a periodic domain.

```julia
using ContourDynamics
using StaticArrays

T = Float64
beta = T(1.0)         # planetary vorticity gradient
Ld = T(1.0)           # deformation radius
R = T(0.3)            # vortex radius
L = T(3.0)            # domain half-width

domain = PeriodicDomain(L)

# PV staircase: 12 levels discretizing βy
staircase = beta_staircase(beta, domain, 12; nodes_per_contour=64)

# Cyclonic vortex at origin
N_vortex = 64
vortex = PVContour(
    [SVector{2,T}(R * cos(2π*k/N_vortex), R * sin(2π*k/N_vortex))
     for k in 0:N_vortex-1],
    T(2π)
)

prob = ContourProblem(QGKernel(Ld), domain, vcat(staircase, [vortex]))

stepper = RK4Stepper(T(0.005), total_nodes(prob))
params = SurgeryParams(T(0.02), T(0.01), T(0.3), T(1e-6), 401)  # no surgery during run

c0 = centroid(vortex)
evolve!(prob, stepper, params; nsteps=400)

# Find vortex (largest non-spanning contour)
idx = argmax(c -> is_spanning(c) ? zero(T) : abs(vortex_area(c)), prob.contours)
cf = centroid(prob.contours[idx])
println("Vortex drift: Δx=$(round(cf[1]-c0[1]; digits=4)), Δy=$(round(cf[2]-c0[2]; digits=4))")
println("(Cyclones drift north-westward on a beta plane)")
```

## SQG Elliptical Vortex

An elliptical surface buoyancy patch evolving under SQG dynamics. The fractional Laplacian Green's function `G(r) = -1/(2πr)` produces sharper fronts and stronger filamentation than the Euler kernel. The regularization parameter `delta` smooths the velocity singularity at the patch boundary.

```julia
using ContourDynamics
using StaticArrays

N = 200
a, b_ax = 1.0, 0.5   # semi-axes (aspect ratio 2)
pv = 2π
delta = 0.01          # regularization length ≈ segment spacing

nodes = [SVector(a * cos(2π * k / N), b_ax * sin(2π * k / N)) for k in 0:(N-1)]
prob = ContourProblem(SQGKernel(delta), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.002, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 10)

Γ0 = circulation(prob)
evolve!(prob, stepper, params; nsteps=500)

println("Final: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")
println("Circulation conserved: |ΔΓ/Γ₀| = $(abs(circulation(prob) - Γ0) / abs(Γ0))")
```

## Two-Layer QG

A vortex patch in the upper layer of a two-layer quasi-geostrophic system with baroclinic coupling. The coupling matrix connects the PV in each layer to the streamfunction, and the solver uses eigenmode decomposition for efficient computation.

```julia
using ContourDynamics
using StaticArrays
using LinearAlgebra

R, pv = 0.5, 2π
N = 100

# Two-layer coupling
Ld = SVector(1.0)                               # interface deformation radius
F = 1.0 / Ld[1]^2
coupling = SMatrix{2,2}(1.0+F, -F, -F, 1.0+F)  # symmetric coupling matrix

kernel = MultiLayerQGKernel(Ld, coupling)

nodes = [SVector(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]

# Upper-layer vortex, quiescent lower layer
prob = MultiLayerContourProblem(
    kernel, UnboundedDomain(),
    ([PVContour(nodes, pv)], PVContour{Float64}[])
)

stepper = RK4Stepper(0.01, total_nodes(prob))
params = SurgeryParams(0.01, 0.005, 0.2, 1e-6, 50)

E0 = energy(prob)
Γ0 = circulation(prob)
evolve!(prob, stepper, params; nsteps=200)

println("Energy: $(energy(prob))  (change: $(abs(energy(prob)-E0)/abs(E0)))")
println("Circulation: $(circulation(prob))  (change: $(abs(circulation(prob)-Γ0)/abs(Γ0)))")
```
