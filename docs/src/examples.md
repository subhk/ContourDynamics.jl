# Examples

This page collects short, runnable examples. Each one focuses on one idea and
keeps the setup small enough to read in one pass.

If you are new to the package, read these in roughly this order:

1. `Vortex Merger`
2. `Filamentation`
3. `Beta-Plane Vortex Drift`
4. `SQG Elliptical Vortex`
5. `Two-Layer QG`

Full scripts with JLD2 output are available in the [`examples/`](https://github.com/subhk/ContourDynamics.jl/tree/main/examples) directory.

::: tip GPU Acceleration
The vortex merger and filamentation examples support GPU acceleration.
Add `using CUDA` and pass `dev=:gpu` to `Problem`.
GPU velocity is currently available for the Euler and SQG kernels on unbounded domains.
:::

---

## Vortex Merger

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

---

## Filamentation

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

---

## Beta-Plane Vortex Drift

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

---

## SQG Elliptical Vortex

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
delta = 0.01          # regularization length ≈ segment spacing

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

---

## Two-Layer QG

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
    contours  = c_upper,
    dt        = 0.01,
    kernel    = :multilayer_qg,
    Ld        = Ld,
    coupling  = coupling,
    layers    = 2,
)

energy0 = energy(prob)
circulation0 = circulation(prob)
evolve!(prob; nsteps=200)

println("Energy: $(energy(prob))  (change: $(abs(energy(prob)-energy0)/abs(energy0)))")
println("Circulation: $(circulation(prob))  (change: $(abs(circulation(prob)-circulation0)/abs(circulation0)))")
```

**References:**
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Polvani, L.M., Zabusky, N.J. & Flierl, G.R. (1989). *Two-layer geostrophic vortex dynamics. Part 1. Upper-layer V-states and merger.* J. Fluid Mech. **205**, 215--242. [doi:10.1017/S0022112089002016](https://doi.org/10.1017/S0022112089002016)
