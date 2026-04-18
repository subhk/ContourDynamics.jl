# SQG Elliptical Vortex

An elliptical surface buoyancy patch evolving under SQG dynamics. Compared with
Euler, SQG usually produces sharper fronts and more aggressive filamentation.

What to look for:

- stronger small-scale structure than in the Euler examples
- the role of the regularization parameter `delta`
- circulation staying nearly constant

SQG dynamics and their role in atmospheric front formation are described in [Held et al. (1995)](https://doi.org/10.1017/S0022112095000012) and [Constantin, Majda & Tabak (1994)](https://doi.org/10.1088/0951-7715/7/6/001). Filament cascades in contour SQG are studied by [Scott & Dritschel (2014)](https://doi.org/10.1103/PhysRevLett.112.144505).

<!-- TODO: Replace with actual figure after running the example -->
::: info Figure placeholder
*SQG evolution of an elliptical buoyancy patch. The fractional Laplacian produces sharper filaments and stronger fronts than the Euler kernel, with a self-similar cascade of instabilities.*
:::

```julia
using ContourDynamics

N = 200
a, b_ax = 1.0, 0.5   # semi-axes (aspect ratio 2)
pv = 2π
delta = 0.01         # regularization length ≈ segment spacing

c = elliptical_patch(a, b_ax, N, pv)
prob = Problem(; contours=[c], dt=0.002, kernel=:sqg, delta_sqg=delta)

circulation0 = circulation(prob)
evolve!(prob; nsteps=500)

println("Final: $(length(contours(prob))) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(abs(circulation(prob) - circulation0) / abs(circulation0))")
```

**References:**
- Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995). *Surface quasi-geostrophic dynamics.* J. Fluid Mech. **282**, 1--20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)
- Constantin, P., Majda, A.J. & Tabak, E. (1994). *Formation of strong fronts in the 2-D quasigeostrophic thermal active scalar.* Nonlinearity **7**(6), 1495--1533. [doi:10.1088/0951-7715/7/6/001](https://doi.org/10.1088/0951-7715/7/6/001)
- Scott, R.K. & Dritschel, D.G. (2014). *Numerical simulation of a self-similar cascade of filament instabilities in the surface quasigeostrophic system.* Phys. Rev. Lett. **112**, 144505. [doi:10.1103/PhysRevLett.112.144505](https://doi.org/10.1103/PhysRevLett.112.144505)
