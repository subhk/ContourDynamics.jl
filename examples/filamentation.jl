# Filamentation Example
#
# An elliptical vortex patch with aspect ratio > 1 evolves under the
# 2D Euler equations. At high aspect ratios, the vortex sheds thin
# filaments that are removed by contour surgery.

using ContourDynamics
using StaticArrays

T = Float64
N = 200         # nodes on the ellipse
a = 1.0         # semi-major axis
b = 0.3         # semi-minor axis (aspect ratio ~3.3)
pv = 2π

# Elliptical patch
nodes = [SVector{2,T}(a * cos(2π * k / N), b * sin(2π * k / N)) for k in 0:(N-1)]
contour = PVContour(nodes, pv)

kernel = EulerKernel{T}()
domain = UnboundedDomain{T}()
prob = ContourProblem(kernel, domain, [contour])

dt = 0.005
surgery_params = SurgeryParams{T}()
stepper = RK4Stepper(dt, total_nodes(prob))

nsteps = 1000
surgery_interval = 10

println("Filamentation: ellipse a=$a, b=$b (ratio=$(a/b))")
println("Running $nsteps steps (dt=$dt) with surgery every $surgery_interval steps...")

area0 = sum(vortex_area, prob.contours)
circ0 = circulation(prob)

for step in 1:nsteps
    timestep!(prob, stepper)
    if step % surgery_interval == 0
        surgery!(prob, surgery_params)
        resize_buffers!(stepper, prob)
    end
    if step % 200 == 0
        nc = length(prob.contours)
        nn = total_nodes(prob)
        area = sum(vortex_area, prob.contours)
        println("  step $step: $nc contours, $nn nodes, area=$(round(area; digits=6))")
    end
end

area_final = sum(vortex_area, prob.contours)
println("\nDone. Area change: |ΔA/A₀| = $(abs(area_final - area0) / abs(area0))")
println("Final state: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")
