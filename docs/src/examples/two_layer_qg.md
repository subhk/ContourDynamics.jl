# Two-Layer QG

A vortex patch in the upper layer of a two-layer quasi-geostrophic system with
baroclinic coupling. This is the simplest example that shows the multi-layer
API.

What to look for:

- how the coupling matrix is defined
- how a multi-layer problem is constructed
- energy and circulation staying nearly constant over the run

Multi-layer contour dynamics and the modal decomposition are described in
[Dritschel (1989)](https://doi.org/10.1016/0167-7977(89)90004-X). Two-layer
vortex dynamics, including upper-layer V-states and merger, are studied by
[Polvani, Zabusky & Flierl (1989)](https://doi.org/10.1017/S0022112089002016).
The example below uses the equal-depth nondimensional case from Polvani et al.:
unit upper-layer patch radius, unit PV jump, and `γ = R₁ / Ld_upper = 1`.
The full script in `examples/two_layer_qg.jl` writes snapshots and media output
under `examples/output/two_layer_qg/`.

```@repl example_two_layer_qg
using ContourDynamics, StaticArrays

nodes_per_quadrant = 75
N = 4 * nodes_per_quadrant
R, pv = 1.0, 1.0
depth_ratio = 1.0
gamma = 1.0

# Equal-depth two-layer Phillips-model stretching operator
F = gamma^2
Ld = SVector(1.0 / (gamma * sqrt(1 + depth_ratio)))    # modal baroclinic radius
coupling = SMatrix{2,2}(-F, F, F, -F)                   # symmetric with one zero mode

# Upper-layer vortex, quiescent lower layer
c_upper = circular_patch(R, N, pv)

prob = Problem(;
    kernel   = :multilayer_qg,
    dt       = 0.01,
    Ld       = Ld,
    coupling = coupling,
    layers   = ([c_upper], PVContour{Float64}[]),
    surgery  = :none,
)

energy0 = energy(prob)
circulation0 = circulation(prob)
evolve!(prob; nsteps=5)

println("Energy: $(round(energy(prob); digits=6))  (change: $(round(abs(energy(prob)-energy0)/abs(energy0); digits=8)))")
println("Circulation: $(round(circulation(prob); digits=6))  (change: $(round(abs(circulation(prob)-circulation0)/abs(circulation0); digits=8)))");
```
