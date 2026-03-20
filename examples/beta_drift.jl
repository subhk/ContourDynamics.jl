# Beta-Plane Drift Example (QG)
#
# A circular vortex patch on a quasi-geostrophic beta plane drifts
# westward (anticyclone) or north-westward (cyclone) due to the
# beta effect encoded in the QG kernel with finite deformation radius.

using ContourDynamics
using StaticArrays
using JLD2

T = Float64
N = 128
R = 0.5           # patch radius
Ld = 1.0          # Rossby deformation radius
pv = 2π           # positive PV → cyclone

# Circular patch centred at origin
nodes = [SVector{2,T}(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
contour = PVContour(nodes, pv)

kernel = QGKernel(Ld)
domain = UnboundedDomain()
prob = ContourProblem(kernel, domain, [contour])

dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))
nsteps = 500

println("QG vortex: R=$R, Ld=$Ld")
println("Running $nsteps steps (dt=$dt), saving every 100 iterations...")

# Save every 100 iterations
recorder = jld2_recorder("beta_drift.jld2"; save_every=100, dt=dt)

c0 = centroid(prob.contours[1])
evolve!(prob, stepper, SurgeryParams(0.01, 0.005, 0.2, 1e-6, nsteps + 1);
        nsteps=nsteps, callbacks=[recorder])

cf = centroid(prob.contours[1])
println("\nTotal drift: Δx=$(round(cf[1] - c0[1]; digits=4)), Δy=$(round(cf[2] - c0[2]; digits=4))")

# --- Inspect saved data ---
snaps = load_simulation("beta_drift.jld2")
println("\nSaved $(length(snaps)) snapshots to beta_drift.jld2")
for s in snaps
    c = s.contours[1]
    ctr = centroid(c)
    println("  t=$(round(s.time; digits=2))  centroid=($(round(ctr[1]; digits=4)), $(round(ctr[2]; digits=4)))")
end
