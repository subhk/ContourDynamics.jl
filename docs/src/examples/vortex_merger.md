# Vortex Merger

Two identical nested vortices are placed three radii apart, following the
symmetric vortex merger setup in Fig. 13 of Dritschel (1988). Each vortex uses
two contour levels to approximate ``\omega(r) = 2\pi(1-r^2)``.

What to look for:

- nested contours stretch and exchange filaments
- surgery reconnects nearby segments
- the final state has fewer contours than the initial state

The topological reconnection used for this partial-merger calculation is
described in [Dritschel (1988)](https://doi.org/10.1016/0021-9991(88)90165-9)
and [Dritschel (1989)](https://doi.org/10.1016/0167-7977(89)90004-X).
The full script in `examples/vortex_merger.jl` writes snapshots, a final figure,
an MP4 animation, and `vortex_merger_diagnostics.csv` with contour count, node
count, area, circulation drift, and energy drift for validation.

```@repl example_vortex_merger
using ContourDynamics

R = 1.0
sep = 3.0 * R
nlevels = 2
N = 32
pv_jump = 2π / nlevels
radii = [R * sqrt(j / nlevels) for j in 1:nlevels]

nested_vortex(cx) = [circular_patch(r, N, pv_jump; cx=cx) for r in radii]

initial_contours = PVContour{Float64}[]
append!(initial_contours, nested_vortex(-sep / 2))
append!(initial_contours, nested_vortex(+sep / 2))

prob = Problem(; contours=initial_contours, dt=0.005,
                 surgery=SurgeryParams(1e-4, 0.01, 0.15, 1e-8, 10))

circulation0 = circulation(prob)
evolve!(prob; nsteps=5)

final_contours = materialize_contours(prob)
println("Final: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(round(abs(circulation(prob) - circulation0) / abs(circulation0); digits=8))");
```

**References:**
- Dritschel, D.G. (1988). *Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
