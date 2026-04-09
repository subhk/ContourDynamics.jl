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

# To run on GPU, add `using CUDA` and pass `dev=:gpu`:
#   prob = Problem(; contours=[contour], dt=0.005, dev=:gpu)

using ContourDynamics
using JLD2

N = 200         # nodes on the ellipse
a = 1.0         # semi-major axis
b = 0.3         # semi-minor axis (aspect ratio ~3.3)
pv = 2π

contour = elliptical_patch(a, b, N, pv)

prob = Problem(; contours=[contour], dt=0.005,
                 surgery=SurgeryParams(0.005, 0.02, 0.2, 1e-6, 10))
display(prob); println()
nsteps = 1000

println("Running $nsteps steps, saving every t=0.5...")

# Save based on physical time interval
recorder = jld2_recorder("filamentation.jld2"; save_dt=0.5, dt=prob.stepper.dt)

area0 = sum(vortex_area, contours(prob))
evolve!(prob; nsteps=nsteps, callbacks=[recorder])

area_final = sum(vortex_area, contours(prob))
println("\nDone. Area change: |ΔA/A₀| = $(abs(area_final - area0) / abs(area0))")
println("Final state: $(length(contours(prob))) contour(s), $(total_nodes(prob)) nodes")

# --- Inspect saved data ---
snaps = load_simulation("filamentation.jld2")
println("\nSaved $(length(snaps)) snapshots to filamentation.jld2")
for s in snaps
    d = s.diagnostics
    println("  t=$(round(s.time; digits=2))  contours=$(length(s.contours))  " *
            "nodes=$(d.total_nodes)  E=$(round(d.energy; digits=6))")
end
