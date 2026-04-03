# Two-Layer QG Example
#
# A circular vortex patch in the upper layer of a two-layer QG system.
# The inter-layer coupling introduces baroclinic effects through the
# eigenmode decomposition of the coupling matrix.
#
# Multi-layer quasi-geostrophic contour dynamics and the modal
# decomposition used here are described in:
#
#   Dritschel, D.G. (1989). "Contour dynamics and contour surgery: numerical
#   algorithms for extended, high-resolution modelling of vortex dynamics in
#   two-dimensional, inviscid, incompressible flows."
#   Comput. Phys. Rep. 10(3), 77–146. doi:10.1016/0167-7977(89)90004-X
#
#   Polvani, L.M., Zabusky, N.J. & Flierl, G.R. (1989). "Two-layer
#   geostrophic vortex dynamics. Part 1. Upper-layer V-states and merger."
#   J. Fluid Mech. 205, 215–242. doi:10.1017/S0022112089002016

# Note: GPU acceleration is not yet available for multi-layer problems.
# This example runs on CPU only.

using ContourDynamics
using StaticArrays
using LinearAlgebra
using JLD2

T = Float64
N = 100
R = 0.5
pv = 2π

# --- Two-layer parameters ---
Ld = SVector{1,T}(1.0)                          # single deformation radius (N-1 = 1)
coupling = SMatrix{2,2,T}([-1.0, 1.0, 1.0, -1.0])  # standard 2-layer coupling

kernel = MultiLayerQGKernel(Ld, coupling)

println("Two-layer QG kernel:")
println("  Eigenvalues: $(kernel.eigenvalues)")
println("  Deformation radius: Ld = $(Ld[1])")

# Upper-layer vortex patch
nodes = [SVector{2,T}(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
c_upper = PVContour(nodes, pv)

domain = UnboundedDomain()

# Layer 1 has the vortex, layer 2 is empty
prob = MultiLayerContourProblem(kernel, domain, ([c_upper], PVContour{T}[]); dev=CPU())

dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob); dev=CPU())
nsteps = 200

println("\nRunning $nsteps steps (dt=$dt), saving every t=0.5...")

# Save based on physical time interval
recorder = jld2_recorder("two_layer_qg.jld2"; save_dt=0.5, dt=dt)

evolve!(prob, stepper, SurgeryParams(0.01, 0.005, 0.2, 1e-6, nsteps + 1);
        nsteps=nsteps, callbacks=[recorder])

println("\nDone. Final state:")
for (i, contours) in enumerate(prob.layers)
    n = sum(nnodes, contours; init=0)
    println("  Layer $i: $(length(contours)) contour(s), $n nodes")
end

# --- Inspect saved data ---
snaps = load_simulation("two_layer_qg.jld2")
println("\nSaved $(length(snaps)) snapshots to two_layer_qg.jld2")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Γ=$(round(d.circulation; digits=4))")
end
