# Vortex Merger Example
#
# Two co-rotating circular vortex patches placed close enough to merge
# via contour surgery. This demonstrates the complete contour dynamics
# + surgery pipeline.

using ContourDynamics
using StaticArrays

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

kernel = EulerKernel{T}()
domain = UnboundedDomain{T}()
prob = ContourProblem(kernel, domain, [c1, c2])

# --- Time integration with surgery ---
dt = 0.01
surgery_params = SurgeryParams{T}()
stepper = RK4Stepper(dt, total_nodes(prob))

nsteps = 500
surgery_interval = 5

println("Vortex merger: 2 patches, $(2N) total nodes")
println("Running $nsteps steps (dt=$dt) with surgery every $surgery_interval steps...")

circ0 = circulation(prob)

for step in 1:nsteps
    timestep!(prob, stepper)
    if step % surgery_interval == 0
        surgery!(prob, surgery_params)
        resize_buffers!(stepper, prob)
    end
    if step % 100 == 0
        nc = length(prob.contours)
        nn = total_nodes(prob)
        circ = circulation(prob)
        E = energy(prob)
        println("  step $step: $nc contours, $nn nodes, Γ=$(round(circ; digits=4)), E=$(round(E; digits=6))")
    end
end

circ_final = circulation(prob)
println("\nDone. Circulation conserved: |ΔΓ/Γ₀| = $(abs(circ_final - circ0) / abs(circ0))")
println("Final state: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")
