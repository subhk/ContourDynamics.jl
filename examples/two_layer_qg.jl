# Two-Layer QG Example
#
# Literature case: the upper-layer vortex merger in Figure 19 of Polvani,
# Zabusky & Flierl (1989): two initially circular, equal-PV, unit-radius
# vortices in the upper layer, with nondimensional initial centroid distance
# d_i = 2.2, depth ratio δ = 0.2, and γ = R/L_d = 5.
#
# The paper's lower layer has constant geostrophic PV. For δ > 0 we use a
# physical two-layer Phillips stretching operator, γ^2[-1 1; δ -δ], and
# layer thicknesses proportional to (δ, 1). The package performs the weighted
# similarity transform internally; the lower layer has no PV contours.
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
include("two_layer_qg_setup.jl")

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "two_layer_qg")

# --- Polvani, Zabusky & Flierl (1989), upper-layer merger case ---
nodes_per_contour = 128
R = 1.0
pv = 1.0
depth_ratio = 0.2
γ = 5.0
initial_distance = 2.2

# --- Numerical controls ---
dt = 0.01
t_final = 20.0
nsteps = round(Int, t_final / dt)
save_dt = 0.5
surgery_δ = 0.0025
surgery_mu = 0.01
max_segment = 0.12
surgery_every = 25
save_media = true
save_media && include("visualization.jl")

function upper_layer_centroid_distance(prob)
    layer = prob.contour_problem.layers[1]
    length(layer) < 2 && return NaN
    return sqrt(sum((centroid(layer[2]) .- centroid(layer[1])).^2))
end

prob, Ld, coupling = polvani_upper_layer_merger_problem(;
    nodes_per_contour, R, pv, depth_ratio, γ, initial_distance, dt,
    surgery_δ, surgery_mu, max_segment, surgery_every)
display(prob); println()

energy0 = energy(prob)
circulation0 = circulation(prob)
distance0 = upper_layer_centroid_distance(prob)

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "two_layer_qg.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "two_layer_qg")

println("Writing outputs under $OUTDIR")
println("Polvani et al. upper-layer merger: δ=$depth_ratio, γ=$γ, d_i=$initial_distance")
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
