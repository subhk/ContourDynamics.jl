# Vortex Merger Example
#
# Two co-rotating circular vortex patches placed close enough to merge
# via contour surgery. This demonstrates the complete contour dynamics
# + surgery pipeline with JLD2 output.

# To run on GPU, add `using CUDA` and pass `dev=GPU()`:
#   prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours; dev=GPU())
#   stepper = RK4Stepper(dt, total_nodes(prob); dev=GPU())

using ContourDynamics
using StaticArrays
using JLD2

# --- Setup ---
T = Float64
N = 128          # nodes per contour
R = 0.5          # patch radius
sep = 1.8 * R    # centre-to-centre separation (< 3.3R triggers merger)
pv = 2π          # uniform PV jump

# Two circular patches offset along x
function circular_nodes(cx, cy, R, N)
    [SVector{2,T}(cx + R * cos(2π * k / N),
                  cy + R * sin(2π * k / N)) for k in 0:(N-1)]
end

c1 = PVContour(circular_nodes(-sep / 2, 0.0, R, N), pv)
c2 = PVContour(circular_nodes(+sep / 2, 0.0, R, N), pv)

kernel = EulerKernel()
domain = UnboundedDomain()
prob = ContourProblem(kernel, domain, [c1, c2])

# --- Time integration with surgery + file output ---
dt = 0.01
surgery_params = SurgeryParams(0.005, 0.02, 0.2, 1e-6, 5)
stepper = RK4Stepper(dt, total_nodes(prob))
nsteps = 500

println("Vortex merger: 2 patches, $(2N) total nodes")
println("Running $nsteps steps (dt=$dt), saving every 50 iterations...")

# Save contour state every 50 steps
recorder = jld2_recorder("vortex_merger.jld2"; save_every=50, dt=dt)

circ0 = circulation(prob)
evolve!(prob, stepper, surgery_params; nsteps=nsteps, callbacks=[recorder])
circ_final = circulation(prob)

println("\nDone. Circulation conserved: |ΔΓ/Γ₀| = $(abs(circ_final - circ0) / abs(circ0))")
println("Final state: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")

# --- Load and inspect saved data ---
snaps = load_simulation("vortex_merger.jld2")
println("\nSaved $(length(snaps)) snapshots to vortex_merger.jld2")
for s in snaps
    d = s.diagnostics
    println("  step=$(s.step)  t=$(round(s.time; digits=2))  " *
            "contours=$(length(s.contours))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Γ=$(round(d.circulation; digits=4))")
end
