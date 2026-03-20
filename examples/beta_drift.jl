# Beta-Plane Drift Example (QG)
#
# A circular vortex patch on a quasi-geostrophic beta plane drifts
# westward (anticyclone) or north-westward (cyclone) due to the
# beta effect encoded in the QG kernel with finite deformation radius.

using ContourDynamics
using StaticArrays

T = Float64
N = 128
R = 0.5           # patch radius
Ld = 1.0          # Rossby deformation radius
pv = 2π           # positive PV → cyclone

# Circular patch centred at origin
nodes = [SVector{2,T}(R * cos(2π * k / N), R * sin(2π * k / N)) for k in 0:(N-1)]
contour = PVContour(nodes, pv)

kernel = QGKernel(Ld)
domain = UnboundedDomain{T}()
prob = ContourProblem(kernel, domain, [contour])

dt = 0.01
stepper = RK4Stepper(dt, total_nodes(prob))

nsteps = 500
println("QG vortex: R=$R, Ld=$Ld")
println("Running $nsteps steps (dt=$dt)...")

c0 = centroid(prob.contours[1])

for step in 1:nsteps
    timestep!(prob, stepper)
    if step % 100 == 0
        c = centroid(prob.contours[1])
        dx = c[1] - c0[1]
        dy = c[2] - c0[2]
        println("  step $step: centroid=($(round(c[1]; digits=4)), $(round(c[2]; digits=4))), " *
                "drift=($(round(dx; digits=4)), $(round(dy; digits=4)))")
    end
end

cf = centroid(prob.contours[1])
println("\nTotal drift: Δx=$(round(cf[1] - c0[1]; digits=4)), Δy=$(round(cf[2] - c0[2]; digits=4))")
