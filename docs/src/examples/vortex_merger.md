# Vortex Merger

Two co-rotating circular vortex patches are placed close enough to merge. This
is one of the clearest examples of why contour surgery matters.

What to look for:

- the two contours exchange filaments
- surgery reconnects nearby segments
- the final state has fewer contours than the initial state

The critical merger distance and the topological reconnection that enables it are described in [Dritschel (1988)](https://doi.org/10.1016/0021-9991(88)90165-9) and [Dritschel (1989)](https://doi.org/10.1016/0167-7977(89)90004-X).

<!-- TODO: Replace with actual figure after running the example -->
::: info Figure placeholder
*Time evolution of two co-rotating vortex patches undergoing merger. Left: initial state with two circular patches separated by 1.8R. Right: merged state after filament exchange and surgery.*
:::

```julia
using ContourDynamics

N = 128          # nodes per contour
R = 0.5          # patch radius
sep = 1.8 * R    # centre-to-centre separation (< 3.3R triggers merger)
pv = 2π          # uniform PV jump

# Two circular patches offset along x
c1 = circular_patch(R, N, pv; cx=-sep / 2)
c2 = circular_patch(R, N, pv; cx=+sep / 2)

prob = Problem(; contours=[c1, c2], dt=0.01)

circulation0 = circulation(prob)
evolve!(prob; nsteps=500)

println("Final: $(length(contours(prob))) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(abs(circulation(prob) - circulation0) / abs(circulation0))")
```

**References:**
- Dritschel, D.G. (1988). *Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
