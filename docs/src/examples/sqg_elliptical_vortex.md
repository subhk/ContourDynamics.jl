# SQG Elliptical Vortex

An elliptical surface buoyancy anomaly evolving under SQG dynamics. Compared
with Euler, SQG usually produces sharper fronts and more aggressive
filamentation.

What to look for:

- stronger small-scale structure than in the Euler examples
- the role of the regularization parameter `delta`
- circulation staying nearly constant

SQG dynamics and their role in atmospheric front formation are described in [Held et al. (1995)](https://doi.org/10.1017/S0022112095000012) and [Constantin, Majda & Tabak (1994)](https://doi.org/10.1088/0951-7715/7/6/001). Filament cascades in contour SQG are studied by [Scott & Dritschel (2014)](https://doi.org/10.1103/PhysRevLett.112.144505).

The setup below follows the elliptical-vortex example in Held et al. (1995),
where the smooth initial field is
``\Theta(x,y,0)=\exp(-x^2-(4y)^2)`` in a periodic square of side ``2\pi``. Since
ContourDynamics.jl evolves contours rather than grid values, the smooth Gaussian
is approximated by nested level-set contours. This is closer to Held's figure
than a single uniform patch, but it is still not the original 512×512 spectral
calculation with hyperviscosity. The full example script defaults to the
unbounded SQG contour solver for runtime; setting `SQG_ELLIPSE_PERIODIC=true`
uses the periodic box from the paper.

```@repl example_sqg_ellipse
using ContourDynamics

N = 48
aspect_ratio = 4.0
levels = collect(0.2:0.2:0.8)
delta = 0.01         # contour-kernel regularization length

function held_gaussian_contours(levels, nodes_per_outer_contour, aspect_ratio)
    outer_radius = sqrt(-log(first(levels)))
    previous_level = 0.0
    contours = PVContour{Float64}[]
    for level in levels
        radius = sqrt(-log(level))
        n = max(24, round(Int, nodes_per_outer_contour * sqrt(radius / outer_radius)))
        push!(contours, elliptical_patch(radius, radius / aspect_ratio,
                                         n, level - previous_level))
        previous_level = level
    end
    return contours
end

contours0 = held_gaussian_contours(levels, N, aspect_ratio)
remesh_only = SurgeryParams(1e-5, 0.01, 0.12, 1e-10, 5)
prob = Problem(; contours=contours0, dt=0.002, kernel=:sqg, delta_sqg=delta,
               surgery=remesh_only)

circulation0 = circulation(prob)
evolve!(prob; nsteps=5)

final_contours = materialize_contours(prob)
println("Final: $(length(final_contours)) contour(s), $(total_nodes(prob)) nodes")
println("Relative circulation change: $(round(abs(circulation(prob) - circulation0) / abs(circulation0); digits=8))");
```

**References:**
- Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995). *Surface quasi-geostrophic dynamics.* J. Fluid Mech. **282**, 1--20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)
- Constantin, P., Majda, A.J. & Tabak, E. (1994). *Formation of strong fronts in the 2-D quasigeostrophic thermal active scalar.* Nonlinearity **7**(6), 1495--1533. [doi:10.1088/0951-7715/7/6/001](https://doi.org/10.1088/0951-7715/7/6/001)
- Scott, R.K. & Dritschel, D.G. (2014). *Numerical simulation of a self-similar cascade of filament instabilities in the surface quasigeostrophic system.* Phys. Rev. Lett. **112**, 144505. [doi:10.1103/PhysRevLett.112.144505](https://doi.org/10.1103/PhysRevLett.112.144505)
