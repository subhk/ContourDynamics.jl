# Vortex Merger Example
#
# Literature case: the symmetric nested-vortex merger in Figure 13 of:
#
#   Dritschel, D.G. (1988). "Contour surgery: a topological reconnection
#   scheme for extended integrations using contour dynamics."
#   J. Comput. Phys. 77(1), 240-266. doi:10.1016/0021-9991(88)90165-9
#
# Figure 13 starts from two identical unit-radius vortices separated by three
# radii. Each vortex is represented by two nested contours approximating
#
#     omega(r) = 2pi * (1 - r^2)
#
# with radii r_j = sqrt(j / 2), j = 1, 2, and equal vorticity jumps pi.
# The paper used Dritschel's case-1 algorithm parameters from Table IV
# (node-density parameter p = pi/100, dt = 0.05, cutoff scale δ = 1e-4),
# later increasing p at t = 7. This script keeps the same physical initial
# condition, but uses smaller time steps and runs to t = 5, where the current
# surgery implementation remains well behaved.

# Optional GPU:
#   add `using CUDA` and pass `dev=GPU()`
#   unbounded single-layer Euler velocity and topology surgery can run on GPU

using ContourDynamics
using JLD2

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "vortex_merger")

# --- Parameters ---
R = 1.0
separation = 3.0 * R
nlevels = 2
profile_power = 2
nodes_per_contour = 128
dt = 0.005
nsteps = 1000
save_every = 10
δ = 1e-4
μ = 0.01
Δ_max = 0.15

pv_jump = 2π / nlevels
radii = [R * (j / nlevels)^(1 / profile_power) for j in 1:nlevels]

function nested_vortex(cx, cy)
    # Each annular PV level is represented as a separate contour with the same
    # PV jump. The signed areas of nested contours reconstruct the radial
    # vorticity profile used in the Dritschel benchmark.
    return [circular_patch(r, nodes_per_contour, pv_jump; cx=cx, cy=cy) for r in radii]
end

initial_contours = PVContour{Float64}[]
append!(initial_contours, nested_vortex(-separation / 2, 0.0))
append!(initial_contours, nested_vortex(+separation / 2, 0.0))

prob = Problem(; contours=initial_contours,
                 dt=dt,
                 surgery=SurgeryParams(δ, μ, Δ_max, 1e-8, 10))
display(prob);
println()

# --- Time integration with surgery + file output ---
mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "vortex_merger.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "vortex_merger")

println("Using ContourDynamics from $(pathof(ContourDynamics))")
println("Writing outputs under $OUTDIR")
println("Initial condition: two unit-radius nested vortices, center separation = $separation")
println("Contour radii per vortex: $(round.(radii; digits=6)); vorticity jump per contour = $pv_jump")
println("Running $nsteps steps to t = $(nsteps * dt), saving every $save_every iterations...")

save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)
recorder = jld2_recorder(outfile; save_every=save_every, dt=prob.stepper.dt)

# Circulation is insensitive to surgery topology changes when the contour areas
# are preserved, so it is a useful low-cost conservation check for this example.
circ0 = circulation(prob)
evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)
circ_final = circulation(prob)
final_contours = materialize_contours(prob)

println("\nDone. Circulation conserved: |ΔΓ/Γ0| = $(abs(circ_final - circ0) / abs(circ0))")
println("Final state: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")

snaps = load_simulation(outfile)
println("\nSaved $(length(snaps)) snapshots to $outfile")

diagfile = joinpath(OUTDIR, "vortex_merger_diagnostics.csv")
open(diagfile, "w") do io
    println(io, "step,time,ncontours,total_nodes,total_area,circulation,rel_circulation,energy,rel_energy")
    Γ0 = snaps[1].diagnostics.circulation
    E0 = snaps[1].diagnostics.energy
    for s in snaps
        d = s.diagnostics
        area = sum(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), s.contours)
        relΓ = abs(d.circulation - Γ0) / max(abs(Γ0), eps())
        relE = (E0 === nothing || d.energy === nothing) ? nothing :
               abs(d.energy - E0) / max(abs(E0), eps())
        println(io, join((s.step,
                          s.time,
                          length(s.contours),
                          d.total_nodes,
                          area,
                          d.circulation,
                          relΓ,
                          d.energy,
                          relE), ","))
    end
end
println("Saved Fig. 13 validation diagnostics: $diagfile")

for s in snaps
    d = s.diagnostics
    println("  step=$(s.step)  t=$(round(s.time; digits=3))  " *
            "contours=$(length(s.contours))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Γ=$(round(d.circulation; digits=4))")
end

include("visualization.jl")
save_animation(mediabase, snaps; title="Symmetric vortex merger")
