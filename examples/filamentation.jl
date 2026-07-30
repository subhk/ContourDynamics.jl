# Filamentation Example
#
# Literature case: the perturbed elliptic-vortex calculation in equation (13),
# Figure 5, and Table IV, case 1, of:
#
#   Dritschel, D.G. (1988). "Contour surgery: a topological reconnection
#   scheme for extended integrations using contour dynamics."
#   J. Comput. Phys. 77(1), 240-266. doi:10.1016/0021-9991(88)90165-9
#
# The initial condition is a 4:1 ellipse with the small m = 3 eigenmode
# perturbation from Eq. (13),
#
#   x(theta, 0) = (cos(theta), lambda^-1 sin(theta))
#                 + epsilon * lambda^-1 * cos(m theta)
#                   / (sin(theta)^2 + lambda^-2 cos(theta)^2)
#                   * (lambda^-1 cos(theta), -sin(theta)),
#
# with lambda = 4, epsilon = 0.005, and peak vorticity 2pi. The paper's
# case-1 run used dt = 0.05, cutoff scale delta = 1e-4, and 153 initial
# nodes. The package's `mu` and `Delta_max` below are segment-length bounds
# for the implemented remeshing, not Dritschel's node-density parameter p.

# Optional GPU: set `use_gpu = true` below. This requires CUDA.jl and an NVIDIA
# GPU. The unbounded single-layer Euler setup runs velocity plus topology
# surgery/remeshing/reconnection on the GPU path.

using ContourDynamics
using JLD2
using StaticArrays

save_media = true
save_media && include("visualization.jl")
use_gpu = false

if use_gpu
    try
        @eval using CUDA
    catch err
        error("use_gpu=true requires CUDA.jl in the active environment. " *
              "Install CUDA.jl or set use_gpu=false. Original error: $err")
    end
    CUDA.functional() || error("use_gpu=true was requested, but CUDA is not functional on this machine.")
end
device = use_gpu ? GPU() : CPU()

# --- Output ---
OUTDIR = joinpath(@__DIR__, "output", "filamentation")

# --- Parameters ---
lambda = 4.0
epsilon = 0.005
mode = 3
N = 153
pv = 2π
dt = 0.05
t_final = 17.25
nsteps = round(Int, t_final / dt)
save_dt = 0.25
delta = 1e-4
mu = 0.01
Delta_max = 0.15
area_min = 1e-8
n_surgery = 10

function dritschel_perturbed_ellipse(lambda, epsilon, mode, N::Int, pv;
                                     T::Type{<:AbstractFloat}=Float64)
    lambda, epsilon, pv = T(lambda), T(epsilon), T(pv)
    N >= 3 || throw(ArgumentError("N must be >= 3, got $N"))
    lambda > one(T) || throw(ArgumentError("lambda must be > 1, got $lambda"))

    inv_lambda = inv(lambda)
    nodes = [begin
        theta = 2T(π) * T(i) / T(N)
        ctheta, stheta = cos(theta), sin(theta)
        denom = stheta^2 + inv_lambda^2 * ctheta^2
        amp = epsilon * inv_lambda * cos(T(mode) * theta) / denom
        SVector{2,T}(ctheta + amp * inv_lambda * ctheta,
                     inv_lambda * stheta - amp * stheta)
    end for i in 0:(N - 1)]

    return PVContour(nodes, pv)
end

function vortex_centroid(cs)
    weighted = SVector{2,Float64}(0.0, 0.0)
    total = 0.0
    for c in cs
        is_spanning(c) && continue
        weight = c.pv * vortex_area(c)
        weighted += weight * centroid(c)
        total += weight
    end
    abs(total) <= eps(Float64) && return weighted
    return weighted / total
end

relerr(value, reference) =
    value === nothing || reference === nothing ? nothing :
    abs(value - reference) / max(abs(reference), eps(Float64))

fmt(value; digits=6) = value === nothing ? "nothing" : string(round(value; digits=digits))

contour = dritschel_perturbed_ellipse(lambda, epsilon, mode, N, pv)

prob = Problem(; contours=[contour],
                 dt=dt,
                 surgery=SurgeryParams(delta, mu, Delta_max, area_min, n_surgery),
                 dev=device)
display(prob); println()

mkpath(OUTDIR)
outfile = joinpath(OUTDIR, "filamentation.jld2")
rm(outfile; force=true)
mediabase = joinpath(OUTDIR, "filamentation")
diagfile = joinpath(OUTDIR, "filamentation_diagnostics.csv")

println("Using ContourDynamics from $(pathof(ContourDynamics))")
println("Device: $device")
println("Writing outputs under $OUTDIR")
println("Initial condition: Dritschel Fig. 5 perturbed ellipse, lambda=$lambda, " *
        "epsilon=$epsilon, mode=$mode, N=$N")
println("Running $nsteps steps to t = $(nsteps * dt), saving every t = $save_dt...")

# Save the initial condition so the animation includes frame 0.
save_snapshot(outfile, prob.contour_problem, 0; dt=prob.stepper.dt)

# Save based on physical time interval
recorder = jld2_recorder(outfile; save_dt=save_dt, dt=prob.stepper.dt)

initial_contours = materialize_contours(prob)
area0 = sum(c -> abs(vortex_area(c)), initial_contours)
circulation0 = circulation(prob)
angular0 = angular_momentum(prob)
energy0 = energy(prob)
centroid0 = vortex_centroid(initial_contours)
evolve!(prob; nsteps=nsteps, callbacks=[recorder])
save_snapshot(outfile, prob.contour_problem, nsteps; dt=prob.stepper.dt)

final_contours = materialize_contours(prob)
area_final = sum(c -> abs(vortex_area(c)), final_contours)
centroid_final = vortex_centroid(final_contours)
centroid_drift = sqrt(sum((centroid_final .- centroid0).^2))
println("\nDone. Area change: |dA/A0| = $(abs(area_final - area0) / abs(area0))")
println("Circulation change: |dGamma/Gamma0| = $(relerr(circulation(prob), circulation0))")
println("Angular momentum change: |dJ/J0| = $(relerr(angular_momentum(prob), angular0))")
println("Energy change: |dE/E0| = $(relerr(energy(prob), energy0))")
println("Centroid drift: $centroid_drift")
println("Final state: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")

# --- Inspect saved data ---
snaps = load_simulation(outfile)
println("\nSaved $(length(snaps)) snapshots to $outfile")

open(diagfile, "w") do io
    println(io, "step,time,ncontours,total_nodes,total_area,circulation,rel_circulation," *
                "angular_momentum,rel_angular_momentum,energy,rel_energy," *
                "centroid_x,centroid_y,centroid_drift")
    for s in snaps
        d = s.diagnostics
        area = sum(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), s.contours)
        ctr = vortex_centroid(s.contours)
        drift = sqrt(sum((ctr .- centroid0).^2))
        println(io, join((s.step,
                          s.time,
                          length(s.contours),
                          d.total_nodes,
                          area,
                          d.circulation,
                          relerr(d.circulation, circulation0),
                          d.angular_momentum,
                          relerr(d.angular_momentum, angular0),
                          d.energy,
                          relerr(d.energy, energy0),
                          ctr[1],
                          ctr[2],
                          drift), ","))
    end
end
println("Saved Fig. 5-style diagnostics: $diagfile")

for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  contours=$(length(s.contours))  " *
            "nodes=$(d.total_nodes)  E=$(fmt(d.energy))  " *
            "Gamma=$(fmt(d.circulation; digits=4))")
end

if save_media
    save_animation(mediabase, snaps; title="Dritschel Fig. 5 filamentation")
else
    println("Skipped media export because save_media=false.")
end
