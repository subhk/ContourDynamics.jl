# Two-Layer QG Example
#
# A circular vortex patch in the upper layer of a two-layer QG system.
# The inter-layer coupling introduces baroclinic effects through the
# eigenmode decomposition of the coupling matrix.

using ContourDynamics
using StaticArrays
using LinearAlgebra
using JLD2

T = Float64
N = 100
R = 0.5
pv = 2π

# --- Two-layer parameters ---
Ld = SVector{1,T}(1.0)                          # single deformation radius (N-1 = 1)
H = SVector{2,T}(0.5, 0.5)                      # equal layer depths
coupling = SMatrix{2,2,T}([-1.0, 1.0, 1.0, -1.0])  # standard 2-layer coupling

kernel = MultiLayerQGKernel(Ld, coupling, H)

println("Two-layer QG kernel:")
println("  Eigenvalues: $(kernel.eigenvalues)")
println("  Deformation radius: Ld = $(Ld[1])")

# Upper-layer vortex patch
nodes = [SVector{2,T}(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
c_upper = PVContour(nodes, pv)

domain = UnboundedDomain()

# Layer 1 has the vortex, layer 2 is empty
layer1 = ContourProblem(kernel, domain, [c_upper])
layer2 = ContourProblem(kernel, domain, PVContour{T}[])
prob = MultiLayerContourProblem((layer1, layer2))

dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))
nsteps = 200

println("\nRunning $nsteps steps (dt=$dt), saving every t=0.5...")

# Save based on physical time interval
recorder = jld2_recorder("two_layer_qg.jld2"; save_dt=0.5, dt=dt)

evolve!(prob, stepper, SurgeryParams(0.01, 0.005, 0.2, 1e-6, nsteps + 1);
        nsteps=nsteps, callbacks=[recorder])

println("\nDone. Final state:")
for (i, layer) in enumerate(prob.layers)
    println("  Layer $i: $(length(layer.contours)) contour(s), $(total_nodes(layer)) nodes")
end

# --- Inspect saved data ---
snaps = load_simulation("two_layer_qg.jld2")
println("\nSaved $(length(snaps)) snapshots to two_layer_qg.jld2")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  nodes=$(d.total_nodes)  " *
            "E=$(round(d.energy; digits=6))  Γ=$(round(d.circulation; digits=4))")
end
