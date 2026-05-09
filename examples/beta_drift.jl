# Beta-Plane Drift Example (QG) — PV Staircase Method
#
# A circular vortex patch drifts on a beta plane via the PV staircase
# approach: the background PV gradient βy is discretized into horizontal
# spanning contours in a periodic domain. The vortex patch deforms the
# staircase contours, producing beta drift and Rossby wave radiation
# without any explicit beta term in the equations.
#
# The PV staircase technique and its application to beta-plane contour
# dynamics are described in:
#
#   Dritschel, D.G. (1988). "Contour surgery: a topological reconnection
#   scheme for extended integrations using contour dynamics."
#   J. Comput. Phys. 77(1), 240–266. doi:10.1016/0021-9991(88)90165-9
#
#   Dritschel, D.G. (1989). "Contour dynamics and contour surgery: numerical
#   algorithms for extended, high-resolution modelling of vortex dynamics in
#   two-dimensional, inviscid, incompressible flows."
#   Comput. Phys. Rep. 10(3), 77–146. doi:10.1016/0167-7977(89)90004-X
#
# The nondimensional parameters below follow case D of:
#
#   Lam, J.S.-L. & Dritschel, D.G. (2001). "On the beta-drift of an
#   initially circular vortex patch." J. Fluid Mech. 436, 107–129.
#   doi:10.1017/S0022112001003974
#
# Their common model parameters are β = 1, Rd = 1, and periodic half-width
# l = 5π. Case D uses R = 1, PV anomaly ω0 = 5, and nβ = 50 β-contours.
# The full paper runs to t = 28 with CASL; this direct-contour
# example runs a shorter, lower-node demonstration by default.

# Optional GPU:
#   add `using CUDA` and pass `dev=GPU()`
#   single-layer periodic QG velocity can run on GPU; periodic surgery/output
#   still use the CPU reference path

using ContourDynamics
using JLD2
using StaticArrays
include("visualization.jl")

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "beta_drift")

# --- Lam & Dritschel (2001), Table 1, vortex D ---
beta = 1.0                    # β = df/dy
Ld = 1.0                      # Rossby deformation radius Rd
R = 1.0                       # vortex patch radius
omega0 = 5.0                  # uniform positive PV anomaly → cyclone
n_beta = 50                   # number of β-contours

# --- Periodic domain ---
L = 5π                        # half-width l; domain is [-l, l] × [-l, l]

# --- Build PV staircase (discretized βy background) ---
nodes_per_beta_contour = parse(Int, get(ENV, "BETA_DRIFT_BETA_NODES", "4"))
function lam_dritschel_beta_staircase(beta, L, n_beta; nodes_per_contour)
    # The staircase contours are spanning: their closing segment crosses one
    # periodic image in x via the wrap vector instead of closing locally.
    d_beta = 2L / n_beta
    dq = beta * d_beta
    wrap = SVector(2L, 0.0)
    return [begin
        y = (k - 0.5) * d_beta - L
        nodes = [SVector(-L + 2L * (i - 1) / nodes_per_contour, y)
                 for i in 1:nodes_per_contour]
        PVContour(nodes, dq, wrap)
    end for k in 1:n_beta]
end
staircase = lam_dritschel_beta_staircase(beta, L, n_beta;
                                         nodes_per_contour=nodes_per_beta_contour)
println("Beta staircase: $(length(staircase)) spanning contours, Δq = $(round(staircase[1].pv; digits=4))")

# --- Circular vortex patch at origin ---
vortex_nodes = parse(Int, get(ENV, "BETA_DRIFT_VORTEX_NODES", "64"))
vortex = circular_patch(R, vortex_nodes, omega0)

# Combine: staircase contours + vortex patch
all_contours = vcat(staircase, [vortex])

dt = 0.005
t_final = parse(Float64, get(ENV, "BETA_DRIFT_T_FINAL", "0.1"))
nsteps = parse(Int, get(ENV, "BETA_DRIFT_NSTEPS", string(round(Int, t_final / dt))))
save_dt = parse(Float64, get(ENV, "BETA_DRIFT_SAVE_DT", "0.025"))
surgery_delta = 1e-3          # Lam & Dritschel CASL cutoff scale
surgery_mu = 0.01
max_segment = 0.3
prob = Problem(; kernel=:qg, Ld=Ld,
                 domain=:periodic, Lx=L, Ly=L,
                 contours=all_contours, dt=dt,
                 surgery=SurgeryParams(surgery_delta, surgery_mu, max_segment, 1e-6, nsteps + 1))
display(prob); println()

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "beta_drift.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "beta_drift")
println("Writing outputs under $OUTDIR")
println("Lam & Dritschel case D: β=$beta, Rd=$Ld, l=$L, R=$R, ω0=$omega0, nβ=$n_beta")
println("Running $nsteps steps to t=$(nsteps * dt), saving every t=$save_dt...")

# Save the initial condition so the animation includes frame 0.
save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)
recorder = jld2_recorder(outfile; save_dt=save_dt, dt=prob.stepper.dt)

# Track the vortex patch centroid (last contour)
initial_contours = materialize_contours(prob)
vortex_idx = length(initial_contours)
c0 = centroid(initial_contours[vortex_idx])

evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)

# Find the vortex among final contours (non-spanning, largest area)
# Surgery is disabled in this short run, but selecting by area keeps the
# diagnostic meaningful if users enable surgery or extend the integration.
final_contours = materialize_contours(prob)
vortex_final = argmax(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), final_contours)
cf = centroid(vortex_final)
println("\nVortex drift: Δx=$(round(cf[1] - c0[1]; digits=4)), Δy=$(round(cf[2] - c0[2]; digits=4))")
println("(Cyclones drift north-westward on a beta plane)")

# --- Inspect saved data ---
snaps = load_simulation(outfile)
println("\nSaved $(length(snaps)) snapshots to $outfile")
for s in snaps
    # Find largest non-spanning contour at each snapshot
    non_spanning = filter(c -> !is_spanning(c), s.contours)
    if !isempty(non_spanning)
        main_c = argmax(c -> abs(vortex_area(c)), non_spanning)
        ctr = centroid(main_c)
        println("  t=$(round(s.time; digits=2))  vortex centroid=($(round(ctr[1]; digits=4)), $(round(ctr[2]; digits=4)))")
    end
end

save_animation(mediabase, snaps; title="Beta-plane drift")
