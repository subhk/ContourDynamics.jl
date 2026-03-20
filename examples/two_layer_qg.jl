# Two-Layer QG Example
#
# A circular vortex patch in the upper layer of a two-layer QG system.
# The inter-layer coupling introduces baroclinic effects through the
# eigenmode decomposition of the coupling matrix.

using ContourDynamics
using StaticArrays
using LinearAlgebra

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

domain = UnboundedDomain{T}()

# Layer 1 has the vortex, layer 2 is empty
layer1 = ContourProblem(kernel, domain, [c_upper])
layer2 = ContourProblem(kernel, domain, PVContour{T}[])
prob = MultiLayerContourProblem((layer1, layer2))

dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))

nsteps = 200
println("\nRunning $nsteps steps (dt=$dt)...")

for step in 1:nsteps
    timestep!(prob, stepper)
    if step % 50 == 0
        c = centroid(prob.layers[1].contours[1])
        E = energy(prob)
        Γ = circulation(prob)
        println("  step $step: centroid=($(round(c[1]; digits=4)), $(round(c[2]; digits=4))), " *
                "E=$(round(E; digits=6)), Γ=$(round(Γ; digits=4))")
    end
end

println("\nDone. Final state:")
for (i, layer) in enumerate(prob.layers)
    println("  Layer $i: $(length(layer.contours)) contour(s), $(total_nodes(layer)) nodes")
end
