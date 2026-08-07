# Beta-Plane Drift Example (QG) - Contour Dynamics
#
# A circular vortex patch drifts on a beta plane using contour dynamics.  The
# live contours encode full PV: a beta staircase plus the vortex anomaly.  The
# BetaPlaneQGKernel subtracts the frozen straight staircase and adds the
# analytic `reference staircase - beta*y` correction, so the full beta-plane
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
# omega0 = 5, n_beta = 50, output times t = 7, 14, 21, 28, and t_final = 28.
# This package uses direct contour dynamics rather than the paper's 512x512
# CASL inversion, so the direct-CD time step and node counts are convergence
# choices rather than parameters quoted from the paper.

using ContourDynamics

dry_run_value = lowercase(get(ENV, "BETA_DRIFT_DRY_RUN", "false"))
dry_run = dry_run_value in ("1", "true", "yes", "on")
dry_run_value in ("0", "false", "no", "off", "1", "true", "yes", "on") ||
    error("BETA_DRIFT_DRY_RUN must be a boolean value, got $dry_run_value")
if !dry_run
    using JLD2
    include("visualization.jl")
end

OUTDIR = joinpath(@__DIR__, "output", "beta_drift")

# Lam & Dritschel (2001), case D metadata.
beta = 1.0
Ld = 1.0
R = 1.0
omega0 = 5.0
L = 5π
n_beta = 50
t_final = 28.0
save_dt = 7.0
casl_grid_nh = 512
casl_grid_mg = 2
casl_surgery_δ = 1e-3

# Direct-CD resolution choices; the paper does not specify these because it
# computes velocity on a CASL grid.
nodes_per_beta_contour = 64
vortex_nodes = 128
dt = 0.005
nsteps = round(Int, t_final / dt)

n_beta >= 2 || error("n_beta must be at least 2")
nodes_per_beta_contour >= 3 || error("nodes_per_beta_contour must be at least 3")
vortex_nodes >= 3 || error("vortex_nodes must be at least 3")
dt > 0 || error("dt must be positive")
nsteps >= 0 || error("nsteps must be non-negative")
save_dt > 0 || error("save_dt must be positive")

domain = PeriodicDomain(L, L)
staircase = beta_staircase(beta, domain, n_beta;
                           nodes_per_contour=nodes_per_beta_contour)
vortex = circular_patch(R, vortex_nodes, omega0)
all_contours = vcat(staircase, [vortex])

# Lam & Dritschel use CASL contour surgery at δ=1e-3. Its contour-to-grid
# algorithm and surgery cadence do not map one-to-one to this direct-CD solver,
# so do not present arbitrary direct-CD surgery settings as paper parameters.
# A converged direct-CD reproduction must choose and document its own remeshing
# and surgery study.
surgery = :none

prob = Problem(; contours=all_contours,
                 dt=dt,
                 kernel=:beta_plane_qg,
                 beta=beta,
                 Ld=Ld,
                 domain=:periodic,
                 Lx=L,
                 Ly=L,
                 surgery=surgery)

d_beta = 2L / n_beta
beta_jump = beta * d_beta
size_resolution_ratio = L / (n_beta * R)
jump_strength_ratio = abs(beta) * L / (n_beta * abs(omega0))

beta == 1.0 || error("Lam-Dritschel equation (2.5) requires beta=1")
Ld == 1.0 || error("Lam-Dritschel equation (2.5) requires Rd=1")
R == 1.0 && omega0 == 5.0 || error(
    "Lam-Dritschel Table 1 case D requires R=1 and omega0=5")
L == 5π || error("Lam-Dritschel section 3.1 requires l=5pi")
n_beta == 50 || error("Lam-Dritschel case D requires n_beta=50")
t_final == 28.0 || error("Lam-Dritschel Figure 5 ends at t=28")
save_dt == 7.0 || error("Lam-Dritschel Figure 5 is shown at 7-time-unit intervals")
nsteps * dt == t_final || error("The time step does not land exactly on t=28")
expected_levels = [-L + (k - 0.5) * d_beta for k in 1:n_beta]
actual_levels = [c.nodes[1][2] for c in staircase]
actual_levels ≈ expected_levels || error(
    "Beta contours do not match equations (3.2)-(3.3)")
all(c -> c.pv ≈ beta_jump, staircase) || error(
    "Beta-contour jumps do not match equation (3.4)")

println("Output directory: $OUTDIR")
println("Method: contour beta-plane QG with analytic beta*y correction")
println("Lam & Dritschel case D metadata: beta=$beta, Rd=$Ld, l=$L, R=$R, omega0=$omega0")
println("Lam & Dritschel Table 1 CASL metadata: nbar_h=$casl_grid_nh, mg=$casl_grid_mg, δ=$casl_surgery_δ")
println("Equation (3.3) spacing d_beta=$d_beta; equation (3.4) PV jump=$beta_jump")
println("Paper criteria: l/(n_beta*R)=$size_resolution_ratio, beta*l/(n_beta*abs(omega0))=$jump_strength_ratio")
println("Contour resolution: beta contours=$(length(staircase)), beta nodes/contour=$nodes_per_beta_contour, vortex nodes=$vortex_nodes")
println("Total nodes: $(total_nodes(prob)); dt=$dt, nsteps=$nsteps, save_dt=$save_dt")
println("Surgery: disabled (the paper's CASL surgery metadata is reported above, not imitated)")

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

media_snaps = filter(s -> s.time > 0, snaps)
save_animation(mediabase, media_snaps;
               title="Lam-Dritschel case D (direct contour dynamics)",
               limits=(-L, L, -0.25L, 0.75L),
               periodic_box=(-L, L, -L, L))
