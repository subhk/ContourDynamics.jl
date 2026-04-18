# Two-Layer QG

A vortex patch in the upper layer of a two-layer quasi-geostrophic system with
baroclinic coupling. This is the simplest example that shows the multi-layer
API.

What to look for:

- how the coupling matrix is defined
- how a multi-layer problem is constructed
- energy and circulation staying nearly constant over the run

Multi-layer contour dynamics and the modal decomposition are described in [Dritschel (1989)](https://doi.org/10.1016/0167-7977(89)90004-X). Two-layer vortex dynamics, including upper-layer V-states and merger, are studied by [Polvani, Zabusky & Flierl (1989)](https://doi.org/10.1017/S0022112089002016).

<!-- TODO: Replace with actual figure after running the example -->
::: info Figure placeholder
*Evolution of a vortex patch in the upper layer of a two-layer QG system. The baroclinic coupling transfers energy between the barotropic and baroclinic modes, modifying the vortex shape compared to single-layer dynamics.*
:::

```julia
using ContourDynamics
using StaticArrays

R, pv = 0.5, 2π
N = 100

# Two-layer stretching operator
Ld = SVector(1.0)                               # baroclinic deformation radius
F = 1.0 / (2 * Ld[1]^2)
coupling = SMatrix{2,2}(-F, F, F, -F)          # symmetric with one zero mode

# Upper-layer vortex, quiescent lower layer
c_upper = circular_patch(R, N, pv)

prob = Problem(;
    kernel   = :multilayer_qg,
    dt       = 0.01,
    Ld       = Ld,
    coupling = coupling,
    layers   = ([c_upper], PVContour{Float64}[]),
)

energy0 = energy(prob)
circulation0 = circulation(prob)
evolve!(prob; nsteps=200)

println("Energy: $(energy(prob))  (change: $(abs(energy(prob)-energy0)/abs(energy0)))")
println("Circulation: $(circulation(prob))  (change: $(abs(circulation(prob)-circulation0)/abs(circulation0)))")
```
