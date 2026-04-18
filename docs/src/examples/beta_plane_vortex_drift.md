# Beta-Plane Vortex Drift

A cyclonic vortex on a beta plane drifts north-westward due to the background
PV gradient. The gradient ``\beta y`` is represented by a PV staircase in a
periodic domain.

What to look for:

- the vortex is embedded in a set of spanning contours
- the vortex center moves over time
- the reported drift should be north-westward

<!-- TODO: Replace with actual figure after running the example -->
::: info Figure placeholder
*Trajectory of a cyclonic vortex patch on a beta plane (QG dynamics). The vortex drifts north-westward, consistent with the classical Rossby wave radiation mechanism. Background: PV staircase contours (horizontal lines).*
:::

```julia
using ContourDynamics

beta = 1.0            # planetary vorticity gradient
Ld = 1.0              # deformation radius
R = 0.3               # vortex radius
L = 3.0               # domain half-width

# PV staircase: 12 levels discretizing βy
staircase = beta_staircase(beta, PeriodicDomain(L), 12; nodes_per_contour=64)

# Cyclonic vortex at origin
vortex = circular_patch(R, 64, 2π)

prob = Problem(;
    contours = vcat(staircase, [vortex]),
    dt       = 0.005,
    kernel   = :qg,
    Ld       = Ld,
    domain   = :periodic,
    Lx       = L,
    Ly       = L,
)

c0 = centroid(vortex)
evolve!(prob; nsteps=400)

# Find vortex (largest non-spanning contour)
idx = argmax(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), contours(prob))
cf = centroid(contours(prob)[idx])
println("Vortex drift: dx=$(round(cf[1]-c0[1]; digits=4)), dy=$(round(cf[2]-c0[2]; digits=4))")
println("(Cyclones drift north-westward on a beta plane)")
```

**References:**
- Dritschel, D.G. (1988). *Contour surgery.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
