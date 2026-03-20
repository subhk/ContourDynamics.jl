# Examples

Complete runnable scripts demonstrating ContourDynamics.jl capabilities. All examples are available in the [`examples/`](https://github.com/subhk/ContourDynamics.jl/tree/main/examples) directory.

## Vortex Merger

Two co-rotating circular vortex patches placed close enough to merge via contour surgery.

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
surgery_params = SurgeryParams{T}()
stepper = RK4Stepper(dt, total_nodes(prob))

for step in 1:500
    timestep!(prob, stepper)
    if step % 5 == 0
        surgery!(prob, surgery_params)
        resize_buffers!(stepper, prob)
    end
end

println("Final: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")
```

## Filamentation

An elliptical vortex patch with high aspect ratio sheds thin filaments that are removed by surgery.

```julia
using ContourDynamics
using StaticArrays

N = 200
a, b = 1.0, 0.3  # semi-axes (aspect ratio ~3.3)
pv = 2π

nodes = [SVector(a * cos(2π * k / N), b * sin(2π * k / N)) for k in 0:(N-1)]
prob = ContourProblem(EulerKernel(), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.005, total_nodes(prob))
surgery_params = SurgeryParams{Float64}()

for step in 1:1000
    timestep!(prob, stepper)
    if step % 10 == 0
        surgery!(prob, surgery_params)
        resize_buffers!(stepper, prob)
    end
end

println("Final: $(length(prob.contours)) contour(s)")
```

## QG Vortex with Deformation Radius

A circular vortex patch evolving under quasi-geostrophic dynamics with a finite Rossby deformation radius.

```julia
using ContourDynamics
using StaticArrays

R, Ld, pv = 0.5, 1.0, 2π
N = 128

nodes = [SVector(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
prob = ContourProblem(QGKernel(Ld), UnboundedDomain(), [PVContour(nodes, pv)])

stepper = RK4Stepper(0.01, total_nodes(prob))

for step in 1:500
    timestep!(prob, stepper)
end

c = centroid(prob.contours[1])
println("Centroid after 500 steps: ($( c[1]), $(c[2]))")
```

## Two-Layer QG

A vortex patch in the upper layer of a two-layer quasi-geostrophic system with baroclinic coupling.

```julia
using ContourDynamics
using StaticArrays
using LinearAlgebra

R, pv = 0.5, 2π
N = 100

Ld = SVector(1.0)                                    # deformation radius
H = SVector(0.5, 0.5)                                # equal layer depths
coupling = SMatrix{2,2}([-1.0, 1.0, 1.0, -1.0])     # 2-layer coupling

kernel = MultiLayerQGKernel(Ld, coupling, H)

nodes = [SVector(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]

layer1 = ContourProblem(kernel, UnboundedDomain(), [PVContour(nodes, pv)])
layer2 = ContourProblem(kernel, UnboundedDomain(), PVContour{Float64}[])
prob = MultiLayerContourProblem((layer1, layer2))

stepper = RK4Stepper(0.01, total_nodes(prob))

for step in 1:200
    timestep!(prob, stepper)
end

println("Energy: $(energy(prob))")
println("Circulation: $(circulation(prob))")
```
