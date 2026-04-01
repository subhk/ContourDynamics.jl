# Beta-Plane Drift Example (QG) — PV Staircase Method
#
# A circular vortex patch drifts on a beta plane via the PV staircase
# approach: the background PV gradient βy is discretized into horizontal
# spanning contours in a periodic domain. The vortex patch deforms the
# staircase contours, producing beta drift and Rossby wave radiation
# without any explicit beta term in the equations.
#
# Reference: Dritschel (1988), "Contour surgery: A topological reconnection
# scheme for extended integrations using contour dynamics", JCP 77(1).

# To run on GPU, add `using CUDA` and pass `dev=GPU()`:
#   prob = ContourProblem(EulerKernel(), domain, contours; dev=GPU())
#   stepper = RK4Stepper(dt, total_nodes(prob); dev=GPU())

using ContourDynamics
using StaticArrays
using JLD2

T = Float64

# --- Physical parameters ---
beta = 1.0            # β = df/dy
Ld = 1.0              # Rossby deformation radius
R = 0.3               # vortex patch radius (< domain size)
pv_vortex = 2π        # positive PV → cyclone

# --- Periodic domain ---
L = 3.0               # half-width; domain is [-L, L] × [-L, L]
domain = PeriodicDomain(L)

# --- Build PV staircase (discretized βy background) ---
n_stairs = 12         # number of staircase levels
staircase = beta_staircase(T(beta), domain, n_stairs; nodes_per_contour=64)
println("Beta staircase: $(length(staircase)) spanning contours, Δq = $(round(staircase[1].pv; digits=4))")

# --- Circular vortex patch at origin ---
N_vortex = 64
nodes = [SVector{2,T}(R * cos(2π * k / N_vortex), R * sin(2π * k / N_vortex)) for k in 0:(N_vortex-1)]
vortex = PVContour(nodes, T(pv_vortex))

# Combine: staircase contours + vortex patch
all_contours = vcat(staircase, [vortex])
kernel = QGKernel(T(Ld))
prob = ContourProblem(kernel, domain, all_contours; dev=CPU())

println("Total contours: $(length(prob.contours)), total nodes: $(total_nodes(prob))")

# --- Time stepping ---
dt = 0.005
nsteps = 400
stepper = RK4Stepper(dt, total_nodes(prob); dev=CPU())
surgery_params = SurgeryParams(T(0.02), T(0.01), T(0.3), T(1e-6), nsteps + 1)

println("\nRunning $nsteps steps (dt=$dt), saving every t=0.5...")
recorder = jld2_recorder("beta_drift.jld2"; save_dt=0.5, dt=dt)

# Track the vortex patch centroid (last contour)
vortex_idx = length(prob.contours)
c0 = centroid(prob.contours[vortex_idx])

evolve!(prob, stepper, surgery_params;
        nsteps=nsteps, callbacks=[recorder])

# Find the vortex among final contours (non-spanning, largest area)
vortex_final = argmax(c -> is_spanning(c) ? zero(T) : abs(vortex_area(c)), prob.contours)
cf = centroid(prob.contours[vortex_final])
println("\nVortex drift: Δx=$(round(cf[1] - c0[1]; digits=4)), Δy=$(round(cf[2] - c0[2]; digits=4))")
println("(Cyclones drift north-westward on a beta plane)")

# --- Inspect saved data ---
snaps = load_simulation("beta_drift.jld2")
println("\nSaved $(length(snaps)) snapshots to beta_drift.jld2")
for s in snaps
    # Find largest non-spanning contour at each snapshot
    non_spanning = filter(c -> !is_spanning(c), s.contours)
    if !isempty(non_spanning)
        main_c = argmax(c -> abs(vortex_area(c)), non_spanning)
        ctr = centroid(main_c)
        println("  t=$(round(s.time; digits=2))  vortex centroid=($(round(ctr[1]; digits=4)), $(round(ctr[2]; digits=4)))")
    end
end
