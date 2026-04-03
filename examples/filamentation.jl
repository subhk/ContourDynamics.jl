# Filamentation Example
#
# An elliptical vortex patch with aspect ratio > 1 evolves under the
# 2D Euler equations. At high aspect ratios, the vortex sheds thin
# filaments that are removed by contour surgery.
#
# The Kirchhoff elliptic vortex is an exact rotating solution; Love (1893)
# showed it becomes unstable for aspect ratios above ~3, leading to
# filamentation. Contour surgery cleanly removes the shed filaments:
#
#   Love, A.E.H. (1893). "On the stability of certain vortex motions."
#   Proc. London Math. Soc. 25, 18–42. doi:10.1112/plms/s1-25.1.18
#
#   Dritschel, D.G. (1988). "Contour surgery: a topological reconnection
#   scheme for extended integrations using contour dynamics."
#   J. Comput. Phys. 77(1), 240–266. doi:10.1016/0021-9991(88)90165-9

# To run on GPU, add `using CUDA` and pass `dev=GPU()`:
#   prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours; dev=GPU())
#   stepper = RK4Stepper(dt, total_nodes(prob); dev=GPU())

using ContourDynamics
using StaticArrays
using JLD2

T = Float64
N = 200         # nodes on the ellipse
a = 1.0         # semi-major axis
b = 0.3         # semi-minor axis (aspect ratio ~3.3)
pv = 2π

# Elliptical patch
nodes = [SVector{2,T}(a * cos(2π * k / N), b * sin(2π * k / N)) for k in 0:(N-1)]
contour = PVContour(nodes, pv)

kernel = EulerKernel()
domain = UnboundedDomain()
prob = ContourProblem(kernel, domain, [contour]; dev=CPU())

dt = 0.005
surgery_params = SurgeryParams(0.005, 0.02, 0.2, 1e-6, 10)
stepper = RK4Stepper(dt, total_nodes(prob); dev=CPU())
nsteps = 1000

println("Filamentation: ellipse a=$a, b=$b (ratio=$(a/b))")
println("Running $nsteps steps (dt=$dt), saving every t=0.5...")

# Save based on physical time interval
recorder = jld2_recorder("filamentation.jld2"; save_dt=0.5, dt=dt)

area0 = sum(vortex_area, prob.contours)
evolve!(prob, stepper, surgery_params; nsteps=nsteps, callbacks=[recorder])

area_final = sum(vortex_area, prob.contours)
println("\nDone. Area change: |ΔA/A₀| = $(abs(area_final - area0) / abs(area0))")
println("Final state: $(length(prob.contours)) contour(s), $(total_nodes(prob)) nodes")

# --- Inspect saved data ---
snaps = load_simulation("filamentation.jld2")
println("\nSaved $(length(snaps)) snapshots to filamentation.jld2")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  contours=$(length(s.contours))  " *
            "nodes=$(d.total_nodes)  E=$(round(d.energy; digits=6))")
end
