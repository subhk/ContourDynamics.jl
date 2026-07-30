# Beta-Plane Drift Example (QG) - Contour Dynamics
#
# A circular vortex patch drifts on a beta plane using contour dynamics.  The
# live contours encode full PV: a beta staircase plus the vortex anomaly.  The
# BetaPlaneQGKernel subtracts the frozen straight staircase and adds the
# analytic `reference staircase - beta*y` correction, so the regular beta-plane
# inversion remains contour-based while the staircase contours stay material.
#
# Literature case: vortex D in Table 1 and Figure 5 of:
#
#   Lam, J.S.-L. & Dritschel, D.G. (2001). "On the beta-drift of an
#   initially circular vortex patch." J. Fluid Mech. 436, 107-129.
#   doi:10.1017/S0022112001003974
#
# The physical initial condition is their equation (2.4), nondimensionalized
# with beta = Rd = 1 as in equation (2.5), using the case-D values R = 1 and
# omega0 = 5. The "paper" preset also uses their n_beta = 50 and t_final = 28.
# The "demo" preset only reduces contour resolution and duration. This package
# uses direct contour dynamics rather than the paper's 512x512 CASL inversion,
# so nodes per contour and surgery cadence are implementation choices rather
# than parameters quoted from the paper.

using ContourDynamics

dry_run = false
if !dry_run
    using JLD2
    include("visualization.jl")
end

preset = "demo"
preset in ("demo", "paper") || error("preset must be either demo or paper")
paper_preset = preset == "paper"

OUTDIR = joinpath(@__DIR__, "output", "beta_drift")

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

n_beta = paper_preset ? 50 : 16
nodes_per_beta_contour = paper_preset ? 64 : 16
vortex_nodes = paper_preset ? 128 : 48

dt = paper_preset ? 0.005 : 0.01
t_final = paper_preset ? paper_t_final : 1.0
nsteps = round(Int, t_final / dt)
save_dt = paper_preset ? 0.25 : 0.1

surgery_delta = casl_surgery_delta
surgery_mu = 0.01
max_segment = 0.3
surgery_every = nsteps + 1

n_beta >= 2 || error("n_beta must be at least 2")
nodes_per_beta_contour >= 3 || error("nodes_per_beta_contour must be at least 3")
vortex_nodes >= 3 || error("vortex_nodes must be at least 3")
dt > 0 || error("dt must be positive")
nsteps >= 0 || error("nsteps must be non-negative")
save_dt > 0 || error("save_dt must be positive")
surgery_every >= 1 || error("surgery_every must be at least 1")

domain = PeriodicDomain(L, L)
staircase = beta_staircase(beta, domain, n_beta;
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
println("Method: contour beta-plane QG with analytic beta*y correction")
println("Lam & Dritschel case D metadata: beta=$beta, Rd=$Ld, l=$L, R=$R, omega0=$omega0")
println("Lam & Dritschel Table 1 CASL metadata: nbar_h=$casl_grid_nh, mg=$casl_grid_mg, delta=$casl_surgery_delta")
println("Contour resolution: beta contours=$(length(staircase)), beta nodes/contour=$nodes_per_beta_contour, vortex nodes=$vortex_nodes")
println("Total nodes: $(total_nodes(prob)); dt=$dt, nsteps=$nsteps, save_dt=$save_dt")
if surgery_every > nsteps
    println("Surgery cadence: disabled for this run (lower surgery_every to enable)")
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
