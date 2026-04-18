# Example: SQG elliptical vortex
#
# Purpose:
#   Demonstrate regularized SQG contour dynamics for an initially elliptical
#   buoyancy patch. Compared with Euler, the SQG kernel tends to produce sharper
#   fronts and stronger filamentation.
#
# Run:
#   julia --project examples/sqg_elliptical_vortex.jl
#
# Optional GPU:
#   Add `using CUDA` and pass `dev=GPU()`:
#   prob = Problem(; contours=[c], dt=0.002, kernel=:sqg, delta_sqg=delta, dev=GPU())

using ContourDynamics
using JLD2

N = 200
a = 1.0
b = 0.5
pv = 2π
delta = 0.01

c = elliptical_patch(a, b, N, pv)
prob = Problem(; contours=[c], dt=0.002, kernel=:sqg, delta_sqg=delta)

circulation0 = circulation(prob)

evolve!(prob; nsteps=500)

println("Final: $(length(contours(prob))) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(abs(circulation(prob) - circulation0) / abs(circulation0))")

jldsave("sqg_elliptical_vortex.jld2";
        contours = contours(prob),
        circulation0 = circulation0,
        circulationf = circulation(prob),
        total_nodes = total_nodes(prob),
        delta = delta)
