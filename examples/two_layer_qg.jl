# Two-Layer QG Example
#
# Literature case: the upper-layer vortex merger in Figure 19 of Polvani,
# Zabusky & Flierl (1989): two initially circular, equal-PV, unit-radius
# vortices in the upper layer, with nondimensional initial centroid distance
# d_i = 2.2, depth ratio delta = 0.2, and gamma = R/L_d = 5.
#
# The paper's lower layer has constant geostrophic PV. For delta > 0 we use a
# symmetric similarity transform of the two-layer Phillips stretching operator,
# gamma^2[-1 1; delta -delta], so the package's symmetric modal solver gives
# the same upper-layer streamfunction while keeping the lower layer empty.
#
#   Polvani, L.M., Zabusky, N.J. & Flierl, G.R. (1989). "Two-layer
#   geostrophic vortex dynamics. Part 1. Upper-layer V-states and merger."
#   J. Fluid Mech. 205, 215-242. doi:10.1017/S0022112089002016
#
# The initial condition and nondimensional physical parameters match the paper.
# Contour resolution, timestep, automatic surgery settings, and the shorter
# t_final = 20 run are package choices; the paper displays Figure 19 through
# t = 74 and used a different adaptive contour implementation.

using ContourDynamics
using StaticArrays
using JLD2
using LinearAlgebra

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "two_layer_qg")

# --- Polvani, Zabusky & Flierl (1989), upper-layer merger case ---
nodes_per_contour = 128
R = 1.0
pv = 1.0
depth_ratio = 0.2
gamma = 5.0
initial_distance = 2.2

# --- Numerical controls ---
dt = 0.01
t_final = 20.0
nsteps = round(Int, t_final / dt)
save_dt = 0.5
surgery_delta = 0.0025
surgery_mu = 0.01
max_segment = 0.12
surgery_every = 25
save_media = true
save_media && include("visualization.jl")

function polvani_upper_layer_merger_problem(; nodes_per_contour=nodes_per_contour,
                                             R=R,
                                             pv=pv,
                                             depth_ratio=depth_ratio,
                                             gamma=gamma,
                                             initial_distance=initial_distance,
                                             dt=dt,
                                             surgery_delta=surgery_delta,
                                             surgery_mu=surgery_mu,
                                             max_segment=max_segment,
                                             surgery_every=surgery_every)
    depth_ratio > 0 || throw(ArgumentError("depth_ratio must be positive for the two-layer solver. Use a single-layer QG kernel for the equivalent-barotropic delta = 0 limit."))
    sqrt_delta = sqrt(depth_ratio)
    Ld = SVector(1.0 / (gamma * sqrt(1 + depth_ratio)))
    coupling = gamma^2 * SMatrix{2,2}(-1.0, sqrt_delta,
                                      sqrt_delta, -depth_ratio)

    c_left = circular_patch(R, nodes_per_contour, pv; cx=-initial_distance / 2)
    c_right = circular_patch(R, nodes_per_contour, pv; cx=initial_distance / 2)
    surgery = SurgeryParams(surgery_delta, surgery_mu, max_segment, 1e-8, surgery_every)

    prob = Problem(; kernel=:multilayer_qg,
                     Ld=Ld,
                     coupling=coupling,
                     layers=([c_left, c_right], PVContour{Float64}[]),
                     dt=dt,
                     surgery=surgery)

    return prob, Ld, coupling
end

function upper_layer_centroid_distance(prob)
    layer = prob.contour_problem.layers[1]
    length(layer) < 2 && return NaN
    return sqrt(sum((centroid(layer[2]) .- centroid(layer[1])).^2))
end

prob, Ld, coupling = polvani_upper_layer_merger_problem()
display(prob); println()

energy0 = energy(prob)
circulation0 = circulation(prob)
distance0 = upper_layer_centroid_distance(prob)

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "two_layer_qg.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "two_layer_qg")

println("Writing outputs under $OUTDIR")
println("Polvani et al. upper-layer merger: delta=$depth_ratio, gamma=$gamma, d_i=$initial_distance")
println("Modal deformation radius: $(Ld[1]); coupling eigenvalues: $(eigvals(Matrix(coupling)))")
println("Running $nsteps steps to t=$(nsteps * dt), saving every t=$save_dt...")

# Save the initial condition so the animation includes frame 0.
save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)
recorder = jld2_recorder(outfile; save_dt=save_dt, dt=prob.stepper.dt)

evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)

println("\nDone. Final state:")
for (i, layer_contours) in enumerate(prob.contour_problem.layers)
    n = sum(nnodes, layer_contours; init=0)
    println("  Layer $i: $(length(layer_contours)) contour(s), $n nodes")
end
println("Initial upper-layer centroid distance: $distance0")
println("Final upper-layer centroid distance: $(upper_layer_centroid_distance(prob))")
println("Relative energy change: $(abs(energy(prob) - energy0) / max(abs(energy0), eps(Float64)))")
println("Relative circulation change: $(abs(circulation(prob) - circulation0) / max(abs(circulation0), eps(Float64)))")

# --- Inspect saved data ---
snaps = load_simulation(outfile)
println("\nSaved $(length(snaps)) snapshots to $outfile")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Gamma=$(round(d.circulation; digits=4))")
end

if save_media
    save_animation(mediabase, snaps;
                   title="Two-layer QG upper-layer merger",
                   figure_size=(1200, 1200),
                   linewidth=2.5,
                   fillalpha=0.0,
                   framerate=15)
end
