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
using JLD2

N = 100
R = 0.5
pv = 2π

# --- Two-layer parameters ---
Ld = SVector(1.0)                                # single deformation radius (N-1 = 1)
F = 1.0 / (2 * Ld[1]^2)                         # stretching coefficient
coupling = SMatrix{2,2}(-F, F, F, -F)            # [-F F; F -F] (column-major order)

c_upper = circular_patch(R, N, pv)
nsteps = 200

prob = Problem(; kernel=:multilayer_qg, Ld=Ld, coupling=coupling,
                 layers=([c_upper], PVContour{Float64}[]),
                 dt=0.01,
                 surgery=SurgeryParams(0.01, 0.005, 0.2, 1e-6, nsteps + 1))
display(prob); println()

println("Running $nsteps steps, saving every t=0.5...")

# Save based on physical time interval
recorder = jld2_recorder("two_layer_qg.jld2"; save_dt=0.5, dt=prob.stepper.dt)

evolve!(prob; nsteps=nsteps, callbacks=[recorder])

println("\nDone. Final state:")
for (i, layer_contours) in enumerate(prob.contour_problem.layers)
    n = sum(nnodes, layer_contours; init=0)
    println("  Layer $i: $(length(layer_contours)) contour(s), $n nodes")
end

# --- Inspect saved data ---
snaps = load_simulation("two_layer_qg.jld2")
println("\nSaved $(length(snaps)) snapshots to two_layer_qg.jld2")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Γ=$(round(d.circulation; digits=4))")
end
