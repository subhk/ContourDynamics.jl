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
# Their model problem fixes β = 1 and Rd = 1 in equation (2.5), uses
# a periodic square with half-width l = 5π in section 3.1, and Table 1
# vortex D has R = 1, PV anomaly ω0 = 5, and nβ = 50 beta contours.
# Table 1 also lists the CASL grid parameters n̄h = 512 and mg = 2; those
# are not direct-contour node counts and are reported below as metadata.
# Use BETA_DRIFT_PRESET=paper to run the paper's t = 28 time window. The
# default preset keeps a shorter demonstration run.

# Optional GPU:
#   add `using CUDA` and pass `dev=GPU()`
#   single-layer periodic QG velocity can run on GPU; periodic surgery/output
#   still use the CPU reference path

using ContourDynamics
using StaticArrays

# --- Run controls ---
envint(name, default) = parse(Int, get(ENV, name, string(default)))
envfloat(name, default) = parse(Float64, get(ENV, name, string(default)))
envbool(name, default) = parse(Bool, lowercase(get(ENV, name, string(default))))

dry_run = envbool("BETA_DRIFT_DRY_RUN", false)
if !dry_run
    using JLD2
    include("visualization.jl")
end

preset = lowercase(get(ENV, "BETA_DRIFT_PRESET", "demo"))
preset in ("demo", "paper") || error("BETA_DRIFT_PRESET must be either demo or paper")
paper_preset = preset == "paper"

# --- Output ---
OUTDIR = get(ENV, "BETA_DRIFT_OUTDIR", joinpath(@__DIR__, "output", "beta_drift"))

# --- Lam & Dritschel (2001), Eq. (2.5) and Table 1, vortex D ---
beta = 1.0                    # β = df/dy; set to 1 in Eq. (2.5)
Ld = 1.0                      # Rossby deformation radius Rd; set to 1 in Eq. (2.5)
R = 1.0                       # vortex patch radius
omega0 = 5.0                  # uniform positive PV anomaly → cyclone
n_beta = 50                   # number of β-contours
paper_t_final = 28.0          # Section 3.1 finite-domain reliability window

# --- Periodic domain, Lam & Dritschel (2001), section 3.1 ---
L = 5π                        # half-width l; domain is [-l, l] × [-l, l]
casl_grid_nh = 512            # Table 1 CASL grid parameter n̄h
casl_grid_mg = 2              # Table 1 CASL grid parameter mg
casl_surgery_delta = 1e-3     # Table 1 cutoff scale δ

# Rounded Table 1 resolution ratios for vortex D.
@assert isapprox(L / (n_beta * R), 0.31; atol=0.005)
@assert isapprox(L / (n_beta * abs(omega0)), 6.3e-2; atol=5e-4)

# --- Direct-contour resolution ---
#
# Lam & Dritschel's nβ = 50 is the number of β-contours. Their n̄h = 512 and
# mg = 2 are CASL grid parameters, not the number of direct nodes per spanning
# contour. Keep this explicit: increase BETA_DRIFT_BETA_NODES only for direct
# contour convergence checks.
nodes_per_beta_contour = envint("BETA_DRIFT_BETA_NODES", 4)
vortex_nodes = envint("BETA_DRIFT_VORTEX_NODES", 64)

nodes_per_beta_contour >= 2 || error("BETA_DRIFT_BETA_NODES must be at least 2")
vortex_nodes >= 3 || error("BETA_DRIFT_VORTEX_NODES must be at least 3")

# --- Build PV staircase (discretized βy background), Eqs. (3.2)--(3.4) ---

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
vortex = circular_patch(R, vortex_nodes, omega0)

# Combine: staircase contours + vortex patch
all_contours = vcat(staircase, [vortex])

dt = envfloat("BETA_DRIFT_DT", 0.01)
t_final = envfloat("BETA_DRIFT_T_FINAL", paper_preset ? paper_t_final : 1.0)
nsteps = envint("BETA_DRIFT_NSTEPS", round(Int, t_final / dt))
save_dt = envfloat("BETA_DRIFT_SAVE_DT", 0.25)
surgery_delta = envfloat("BETA_DRIFT_SURGERY_DELTA", casl_surgery_delta)
surgery_mu = envfloat("BETA_DRIFT_SURGERY_MU", 0.01)
max_segment = envfloat("BETA_DRIFT_MAX_SEGMENT", 0.3)
surgery_every = envint("BETA_DRIFT_SURGERY_EVERY", nsteps + 1)

dt > 0 || error("BETA_DRIFT_DT must be positive")
nsteps >= 0 || error("BETA_DRIFT_NSTEPS must be non-negative")
save_dt > 0 || error("BETA_DRIFT_SAVE_DT must be positive")
surgery_delta > 0 || error("BETA_DRIFT_SURGERY_DELTA must be positive")
surgery_mu > 0 || error("BETA_DRIFT_SURGERY_MU must be positive")
max_segment > 0 || error("BETA_DRIFT_MAX_SEGMENT must be positive")
surgery_every >= 1 || error("BETA_DRIFT_SURGERY_EVERY must be at least 1")

prob = Problem(; kernel=:qg, Ld=Ld,
                 domain=:periodic, Lx=L, Ly=L,
                 contours=all_contours, dt=dt,
                 surgery=SurgeryParams(surgery_delta, surgery_mu, max_segment, 1e-6, surgery_every))

display(prob); println()

println("Output directory: $OUTDIR")
println("Preset: $preset")
println("Lam & Dritschel case D: β=$beta, Rd=$Ld, l=$L, R=$R, ω0=$omega0, nβ=$n_beta")
println("Lam & Dritschel Table 1 CASL metadata: n̄h=$casl_grid_nh, mg=$casl_grid_mg, δ=$casl_surgery_delta")
println("Table 1 checks: l/(nβ R)=$(round(L / (n_beta * R); digits=2)), l/(nβ |ω0|)=$(round(L / (n_beta * abs(omega0)); digits=3))")
total_nodes = sum(nnodes(c) for c in all_contours)
pair_work = Float64(total_nodes)^2
println("Direct contour resolution: β nodes/contour=$nodes_per_beta_contour, vortex nodes=$vortex_nodes, total nodes=$total_nodes")
println("Estimated RK4 pair work: $(round(4 * pair_work / 1e6; digits=2)) million segment interactions/step, " *
        "$(round(4 * pair_work * nsteps / 1e9; digits=2)) billion for this run")
if surgery_every > nsteps
    println("Surgery cadence: disabled for this run (set BETA_DRIFT_SURGERY_EVERY to enable)")
else
    println("Surgery cadence: every $surgery_every steps")
end
println("Running $nsteps steps to t=$(nsteps * dt), saving every t=$save_dt...")

if dry_run
    println("Dry run complete; skipping output, evolution, and media export.")
    exit()
end

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "beta_drift.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "beta_drift")

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
