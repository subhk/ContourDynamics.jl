# Beta-Plane Drift Example (QG) - Contour Dynamics
#
# A circular vortex patch drifts on a beta plane using contour dynamics.  The
# live contours encode full PV: a beta staircase plus the vortex anomaly.  The
# BetaPlaneQGKernel subtracts the frozen straight staircase from the QG
# inversion, so only regular PV drives the flow while the staircase contours
# remain material and can deform.
#
# This follows the contour-dynamics structure of Lam & Dritschel (2001), but it
# is still a compact package example rather than a CASL reproduction.

using ContourDynamics

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

OUTDIR = get(ENV, "BETA_DRIFT_OUTDIR", joinpath(@__DIR__, "output", "beta_drift"))

# Lam & Dritschel (2001), case D metadata.
beta = 1.0
Ld = 1.0
R = 1.0
omega0 = 5.0
L = 5π
paper_t_final = 28.0
casl_grid_nh = 512
casl_grid_mg = 2
casl_surgery_delta = 1e-3

n_beta = envint("BETA_DRIFT_NBETA", paper_preset ? 50 : 16)
nodes_per_beta_contour = envint("BETA_DRIFT_BETA_NODES", paper_preset ? 64 : 16)
vortex_nodes = envint("BETA_DRIFT_VORTEX_NODES", paper_preset ? 128 : 48)

dt = envfloat("BETA_DRIFT_DT", paper_preset ? 0.005 : 0.01)
t_final = envfloat("BETA_DRIFT_T_FINAL", paper_preset ? paper_t_final : 1.0)
nsteps = envint("BETA_DRIFT_NSTEPS", round(Int, t_final / dt))
save_dt = envfloat("BETA_DRIFT_SAVE_DT", paper_preset ? 0.25 : 0.1)

surgery_delta = envfloat("BETA_DRIFT_SURGERY_DELTA", casl_surgery_delta)
surgery_mu = envfloat("BETA_DRIFT_SURGERY_MU", 0.01)
max_segment = envfloat("BETA_DRIFT_MAX_SEGMENT", 0.3)
surgery_every = envint("BETA_DRIFT_SURGERY_EVERY", nsteps + 1)

n_beta >= 1 || error("BETA_DRIFT_NBETA must be at least 1")
nodes_per_beta_contour >= 3 || error("BETA_DRIFT_BETA_NODES must be at least 3")
vortex_nodes >= 3 || error("BETA_DRIFT_VORTEX_NODES must be at least 3")
dt > 0 || error("BETA_DRIFT_DT must be positive")
nsteps >= 0 || error("BETA_DRIFT_NSTEPS must be non-negative")
save_dt > 0 || error("BETA_DRIFT_SAVE_DT must be positive")
surgery_every >= 1 || error("BETA_DRIFT_SURGERY_EVERY must be at least 1")

domain = PeriodicDomain(L, L)
staircase = beta_staircase(beta, domain, n_beta + 1;
                           nodes_per_contour=nodes_per_beta_contour)
vortex = circular_patch(R, vortex_nodes, omega0)
all_contours = vcat(staircase, [vortex])

surgery = SurgeryParams(surgery_delta, surgery_mu, max_segment, 1e-6, surgery_every)

prob = Problem(; contours=all_contours,
                 dt=dt,
                 kernel=:beta_plane_qg,
                 beta=beta,
                 Ld=Ld,
                 domain=:periodic,
                 Lx=L,
                 Ly=L,
                 surgery=surgery)

println("Output directory: $OUTDIR")
println("Preset: $preset")
println("Method: contour beta-plane QG, full PV contours minus reference straight beta staircase")
println("Lam & Dritschel case D metadata: beta=$beta, Rd=$Ld, l=$L, R=$R, omega0=$omega0")
println("Lam & Dritschel Table 1 CASL metadata: nbar_h=$casl_grid_nh, mg=$casl_grid_mg, delta=$casl_surgery_delta")
println("Contour resolution: beta contours=$(length(staircase)), beta nodes/contour=$nodes_per_beta_contour, vortex nodes=$vortex_nodes")
println("Total nodes: $(total_nodes(prob)); dt=$dt, nsteps=$nsteps, save_dt=$save_dt")
if surgery_every > nsteps
    println("Surgery cadence: disabled for this run (set BETA_DRIFT_SURGERY_EVERY to enable)")
else
    println("Surgery cadence: every $surgery_every steps")
end

if dry_run
    println("Dry run complete; skipping evolution and media export.")
    exit()
end

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "beta_drift.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "beta_drift")

save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)
recorder = jld2_recorder(outfile; save_dt=save_dt, dt=prob.stepper.dt)

c0 = centroid(vortex)
evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)

final_contours = materialize_contours(prob)
vortex_final = argmax(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), final_contours)
cf = centroid(vortex_final)
println("\nVortex centroid displacement: dx=$(round(cf[1] - c0[1]; digits=4)), dy=$(round(cf[2] - c0[2]; digits=4))")

snaps = load_simulation(outfile)
println("\nSaved $(length(snaps)) contour snapshots to $outfile")
for s in snaps
    non_spanning = filter(c -> !is_spanning(c), s.contours)
    isempty(non_spanning) && continue
    main_c = argmax(c -> abs(vortex_area(c)), non_spanning)
    ctr = centroid(main_c)
    println("  t=$(round(s.time; digits=3))  vortex centroid=($(round(ctr[1]; digits=4)), $(round(ctr[2]; digits=4)))")
end

save_animation(mediabase, snaps; title="Contour beta-plane drift",
               periodic_box=(-L, L, -L, L))
