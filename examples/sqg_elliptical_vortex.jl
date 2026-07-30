# Example: SQG elliptical vortex
#
# Purpose:
#   Demonstrate regularized SQG contour dynamics for an initially smooth
#   elliptical buoyancy anomaly, approximated by nested temperature contours.
#   Compared with Euler, the SQG kernel tends to produce sharper fronts and
#   stronger filamentation.
#
# Literature case: equation (19) and Figure 2 of:
#
#   Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995).
#   "Surface quasi-geostrophic dynamics." J. Fluid Mech. 282, 1–20.
#   doi:10.1017/S0022112095000012
#
# Held et al. use the smooth surface-temperature field
#
#   Θ(x, y, 0) = exp(-x^2 - (4y)^2),
#
# in a doubly periodic square of side 2π and show the evolution at t = 0, 8,
# 16, and 26. This script preserves that literature initial condition and time
# window, but approximates the smooth field by nested constant-temperature
# contours. The default uses the unbounded SQG contour solver so the example
# remains practical; set `periodic_domain = true` below for the paper's domain.
# It is therefore a literature-derived contour approximation, not a reproduction
# of Held et al.'s 512×512 spectral model with hyperviscosity. The contour levels,
# node counts, surgery settings, timestep, and SQG regularization are package
# discretization choices rather than parameters quoted from the paper.
#
# Run:
#   julia --project=. examples/sqg_elliptical_vortex.jl
#
# Optional GPU:
#   add `using CUDA` and pass `dev=GPU()`:
#   prob = Problem(; contours=contours0, dt=dt, kernel=:sqg, delta_sqg=delta, dev=GPU())

using ContourDynamics
using JLD2

save_media = true
save_media && include("visualization.jl")

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "sqg_elliptical_vortex")

# --- Held et al. (1995) smooth elliptical vortex, contoured by level sets ---
N = 200
aspect_ratio = 4.0
levels = collect(0.1:0.1:0.9)

function held_gaussian_contours(levels, nodes_per_outer_contour::Int,
                                aspect_ratio::Real; T=Float64)

    issorted(levels) || throw(ArgumentError("levels must increase from outer to inner contour"))

    all(θ -> 0 < θ < 1, levels) || throw(ArgumentError("levels must lie between 0 and 1"))

    outer_radius = sqrt(-log(T(first(levels))))
    previous_level = zero(T)
    contours = PVContour{T}[]

    for level in levels
        θ = T(level)
        radius = sqrt(-log(θ))
        a = radius
        b = radius / T(aspect_ratio)
        n = max(32, round(Int, nodes_per_outer_contour * sqrt(radius / outer_radius)))
        push!(contours, elliptical_patch(a, b, n, θ - previous_level; T=T))
        previous_level = θ
    end

    return contours
end

# --- Parameters ---
delta   = 0.01
dt      = 0.05
t_final = 26.0
nsteps  = round(Int, t_final / dt)
save_dt = 0.2
L       = Float64(π)
periodic_domain = false
surgery_mode = "remesh"

surgery = if surgery_mode == "none"
    :none
elseif surgery_mode == "reconnect"
    SurgeryParams(0.0025, 0.01, 0.12, 1e-8, 10)
elseif surgery_mode == "remesh"
    # Remesh the stretched level sets, but use a tiny reconnection scale so the
    # nested temperature levels are not topologically reconnected.
    SurgeryParams(1e-5, 0.01, 0.12, 1e-10, 5)
else
    throw(ArgumentError("surgery_mode must be one of: remesh, reconnect, none"))
end

contours0 = held_gaussian_contours(levels, N, aspect_ratio)

prob = if periodic_domain
    Problem(; contours=contours0,
              dt=dt,
              kernel=:sqg,
              delta_sqg=delta,
              domain=:periodic,
              Lx=L,
              Ly=L,
              surgery=surgery)
else
    Problem(; contours=contours0,
              dt=dt,
              kernel=:sqg,
              delta_sqg=delta,
              surgery=surgery)
end

circulation0 = circulation(prob)
mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "sqg_elliptical_vortex.jld2")

rm(outfile; force=true)

mediabase = joinpath(OUTDIR, "sqg_elliptical_vortex")

println("Writing outputs under $OUTDIR")
println("Held-style SQG contour stack: levels=$(levels), aspect ratio=$aspect_ratio")
if periodic_domain
    println("Domain: [-$L, $L) × [-$L, $L); δ=$delta; initial contours=$(length(contours0)), nodes=$(total_nodes(prob))")
else
    println("Domain: unbounded; δ=$delta; initial contours=$(length(contours0)), nodes=$(total_nodes(prob))")
end
println("Contour surgery mode: $surgery_mode")
println("Running $nsteps steps to t=$(nsteps * dt), saving every t=$save_dt...")

# Save the initial condition so the animation includes frame 0.
save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)
recorder = jld2_recorder(outfile; save_dt=save_dt, dt=prob.stepper.dt)

evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)
final_contours = materialize_contours(prob)

println("Final: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(abs(circulation(prob) - circulation0) / abs(circulation0))")

snaps = load_simulation(outfile)
println("Saved $(length(snaps)) snapshots to $outfile")
if save_media
    save_animation(mediabase, snaps;
                   title="SQG elliptical vortex",
                   figure_size=(1200, 1200),
                   linewidth=3,
                   fillalpha=0.0,
                   framerate=15)
end
