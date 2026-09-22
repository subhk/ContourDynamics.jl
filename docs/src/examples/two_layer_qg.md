# Two-Layer QG

Two identical circular upper-layer vortex patches merge in a two-layer
quasi-geostrophic system. The physical setup is Figure 19 of Polvani, Zabusky &
Flierl (1989): unit radius and PV jump, initial centroid distance ``d_i=2.2``,
depth ratio ``\delta=0.2``, and ``\gamma=R/L_{\mathrm{upper}}=5``. The lower layer has uniform
geostrophic PV.

What to look for:

- how the coupling matrix is defined
- how a multi-layer problem is constructed
- energy and circulation staying nearly constant over the run

Polvani et al. write the Phillips stretching operator as

```math
\gamma^2\begin{bmatrix}-1 & 1 \\ \delta & -\delta\end{bmatrix}.
```

Here ``\gamma=R/L_{\mathrm{upper}}`` uses the **upper-layer deformation
radius** defined in section 2 of the paper, and ``\delta=H_1/H_2`` is its
layer-depth ratio. The displayed ``2\times2`` matrix is the nondimensional Phillips
stretching operator; matrix row and column indices identify the affected and
source layers, respectively.

The package's `Ld` argument instead contains the **baroclinic modal radius**:

```math
L_{\mathrm{modal}}
=\frac{L_{\mathrm{upper}}}{\sqrt{1+\delta}}
=\frac{R}{\gamma\sqrt{1+\delta}}.
```

The example uses ``R=1`` as its length unit, so the nonzero coupling eigenvalue
is ``-\gamma^2(1+\delta)`` and `Ld = SVector(L_modal)`.

Pass this physical stretching matrix and the layer thicknesses to the package.
The kernel forms its symmetric similar matrix internally, applies the weighted
PV transform, and reconstructs the physical-layer velocities. Here thicknesses
are expressed in units of ``H_2``, so ``H=(\delta,1)``. The short docs run below
uses fewer nodes and steps than the paper. The full script in
`examples/two_layer_qg.jl` writes snapshots and media under
`examples/output/two_layer_qg/`, but is likewise a literature-derived numerical
adaptation rather than an exact reproduction of the paper's time integration.

```@example example_two_layer_qg
using ContourDynamics, StaticArrays

N = 64
R, pv = 1.0, 1.0
depth_ratio = 0.2
gamma = 5.0
initial_distance = 2.2

# R = 1 is the length unit; distinguish the paper's radius from the modal one.
L_upper = R / gamma
L_modal = L_upper / sqrt(1 + depth_ratio)
Ld = SVector(L_modal)
# Physical two-layer Phillips stretching operator in these units
coupling = gamma^2 * SMatrix{2,2}(-1.0, depth_ratio, 1.0, -depth_ratio)
H = SVector(depth_ratio, 1.0)

# Figure 19: two upper-layer vortices and uniform lower-layer PV
c_left = circular_patch(R, N, pv; cx=-initial_distance / 2)
c_right = circular_patch(R, N, pv; cx=initial_distance / 2)

prob = Problem(;
    kernel   = :multilayer_qg,
    dt       = 0.01,
    Ld       = Ld,
    coupling = coupling,
    layer_thicknesses = H,
    layers   = ([c_left, c_right], PVContour{Float64}[]),
    surgery  = :none,
)

energy0 = energy(prob)
circulation0 = circulation(prob)
evolve!(prob; nsteps=5)

println("Energy: $(round(energy(prob); digits=6))  (change: $(round(abs(energy(prob)-energy0)/abs(energy0); digits=8)))")
println("Circulation: $(round(circulation(prob); digits=6))  (change: $(round(abs(circulation(prob)-circulation0)/abs(circulation0); digits=8)))");
```

**Reference:**
- Polvani, L.M., Zabusky, N.J. & Flierl, G.R. (1989). *Two-layer geostrophic vortex dynamics. Part 1. Upper-layer V-states and merger.* J. Fluid Mech. **205**, 215--242. [doi:10.1017/S0022112089002016](https://doi.org/10.1017/S0022112089002016)
