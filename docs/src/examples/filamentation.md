# Filamentation

An elongated elliptical vortex patch sheds thin filaments. This example shows
how surgery keeps the contour manageable once those filaments become too thin to
resolve well.

What to look for:

- thin filaments appear as the ellipse destabilizes
- small features are removed by surgery
- the core vortex remains well resolved

<!-- TODO: Replace with actual figure after running the example -->
::: info Figure placeholder
*Filamentation of an elliptical vortex patch (aspect ratio 3.3). The unstable ellipse sheds thin filaments that are removed by contour surgery, leaving a rounder core vortex.*
:::

```julia
using ContourDynamics

N = 200
a, b = 1.0, 0.3  # semi-axes (aspect ratio ≈ 3.3)
pv = 2π

c = elliptical_patch(a, b, N, pv)
prob = Problem(; contours=[c], dt=0.005)

A0 = vortex_area(contours(prob)[1])
evolve!(prob; nsteps=1000)

println("Final: $(length(contours(prob))) contour(s)")
println("Area of largest contour: $(maximum(c -> abs(vortex_area(c)), contours(prob)))")
println("Original area: $A0")
```

**References:**
- Love, A.E.H. (1893). *On the stability of certain vortex motions.* Proc. London Math. Soc. **25**, 18--42. [doi:10.1112/plms/s1-25.1.18](https://doi.org/10.1112/plms/s1-25.1.18)
- Dritschel, D.G. (1988). *Contour surgery.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
