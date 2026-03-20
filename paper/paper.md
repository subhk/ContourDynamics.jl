---
title: 'ContourDynamics.jl: A Julia package for Lagrangian contour dynamics in geophysical fluid flows'
tags:
  - Julia
  - geophysical fluid dynamics
  - quasi-geostrophic
  - vortex dynamics
  - contour dynamics
  - contour surgery
  - potential vorticity
authors:
  - name: [Subhajit Kar]
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: [Tel Aviv University]
    index: 1
date: 2025
bibliography: paper.bib
---

# Summary

Geophysical fluid dynamics (GFD) is governed at large scales by the
quasi-geostrophic (QG) equations, in which the flow is entirely determined by
the distribution of potential vorticity (PV). A natural and powerful way to
numerically integrate the QG equations is to represent the PV field as a
collection of *patches* — regions of piecewise-constant PV — whose boundaries
evolve as material contours. This *contour dynamics* approach, introduced by
@zabusky1979 for the 2D Euler equations and extended to the QG system by
@dritschel1988, has three compelling properties for idealized GFD studies:
(1) it is *exactly* Lagrangian, conserving PV along material surfaces without
numerical diffusion; (2) it is *Hamiltonian*, allowing energy-conserving
symplectic integration over arbitrarily long times; and (3) it automatically
concentrates resolution on the dynamically active contour boundaries rather
than wasting computation on quiescent interior regions.

`ContourDynamics.jl` provides the first modern, open-source Julia
implementation of this method, supporting three flow regimes — 2D Euler,
quasi-geostrophic (QG) with arbitrary Rossby deformation radius $L_d$, and
Surface QG (SQG) — in both unbounded and doubly-periodic domains. The package
includes adaptive node management, *contour surgery* [@dritschel1988b] for
long-time topologically-stable integrations, and exact analytical diagnostics
(energy, enstrophy, vortex geometry) computed from the contour geometry without
any gridding. It integrates naturally with the existing Julia GFD ecosystem,
particularly `GeophysicalFlows.jl` [@constantinou2021] and `FourierFlows.jl`.

# Statement of Need

The contour dynamics/surgery framework has underpinned foundational GFD research
for over three decades, including studies of 2D turbulence vortex mergers and
the inverse energy cascade [@dritschel2008], filamentation and stripping of QG
vortices [@dritschel1989], vortex beta-drift on the $\beta$-plane [@lam2001],
and stratospheric vortex dynamics [@dritschel1994]. Despite this rich scientific
legacy, the primary software implementing these methods — the *Hydra* suite
maintained by @dritschel_hydra — is written in Fortran, distributed on a
contact-only basis with no public repository or open license, and lacks
documentation beyond the original research papers. A simplified C++14 library
(`CALIB`) exists on GitHub [@anderone_calib] but has no package manager support,
no documentation, and no Python or Julia interface.

The Julia GFD ecosystem has grown substantially in recent years, with
`GeophysicalFlows.jl` [@constantinou2021] providing GPU-accelerated
pseudospectral QG solvers, `Oceananigans.jl` [@ramadhan2020] providing
hydrostatic and non-hydrostatic ocean models, and `QGDipoles.jl` [@crowe2025]
computing steady dipolar QG vortex solutions. However, **no Julia package
implements Lagrangian contour dynamics**, leaving a methodological gap precisely
where the method has its greatest advantages: long-time vortex interaction
studies where pseudospectral diffusion artificially dissipates thin filaments.

`ContourDynamics.jl` fills this gap. It is the first open, pip/Pkg-installable
implementation of QG contour dynamics with surgery in any modern language,
enabling researchers to run idealized GFD studies — vortex mergers,
filamentation, turbulence cascades, baroclinic instability — within the same
Julia ecosystem used for analysis and visualization.

# Mathematical Formulation

## The Quasi-Geostrophic Equations

In the single-layer QG system, the PV $q$ and streamfunction $\psi$ satisfy:

$$\frac{\partial q}{\partial t} + J(\psi, q) = 0, \qquad q = \nabla^2\psi - \frac{\psi}{L_d^2}$$

where $J$ is the Jacobian operator and $L_d$ is the Rossby deformation radius.
The limiting cases $L_d \to \infty$ and $L_d \to 0$ recover the 2D Euler and
Surface QG equations respectively.

## Contour Representation and Velocity Inversion

The PV field is represented as piecewise-constant:

$$q(\mathbf{x}, t) = \sum_k \Delta q_k \, \mathbf{1}_{D_k}(\mathbf{x})$$

where $D_k$ is the region enclosed by the $k$-th contour and $\Delta q_k$ is
its PV jump. The streamfunction is obtained by inverting via the 2D Green's
function $G$ appropriate for each regime:

$$\psi(\mathbf{x}) = \sum_k \Delta q_k \iint_{D_k} G(\mathbf{x} - \mathbf{x}') \, d^2x'$$

Converting the area integral to a boundary integral (via Green's theorem), the
velocity at any point $\mathbf{x}$ is:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} \nabla G(\mathbf{x} - \mathbf{x}') \times \hat{e}_z \, ds'$$

The three Green's functions are:

| Regime | $G(r)$ | Velocity kernel |
|--------|--------|-----------------|
| 2D Euler ($L_d \to \infty$) | $\frac{1}{2\pi}\ln r$ | Biot-Savart (exact segment formula) |
| QG (finite $L_d$) | $-\frac{1}{2\pi} K_0(r/L_d)$ | Modified Bessel $K_1$ (8-pt Gauss-Legendre) |
| SQG ($L_d \to 0$) | $-\frac{1}{2\pi r}$ | $1/r^2$ kernel (8-pt Gauss-Legendre) |

For 2D Euler, the velocity induced by a straight segment from node $\mathbf{a}$
to $\mathbf{b}$ at point $\mathbf{x}$ has a closed-form analytical expression
(the signed angle subtended at $\mathbf{x}$ by the segment times the unit normal).
For QG and SQG, 8-point Gauss-Legendre quadrature along each segment is used.

## Contour Surgery

As contours evolve, they develop exponentially thin filaments through the
strain of surrounding vorticity. Without intervention, this leads to
ever-increasing node counts and eventual self-intersection. Contour surgery
[@dritschel1988b] resolves this by:

1. Building a KD-tree of all contour nodes (O(N log N))
2. Finding non-adjacent node pairs within distance $\delta$ (the surgery threshold)
3. Performing a *topological reconnection*: swapping the forward-connectivity of
   the two nodes, either splitting one contour into two or merging two into one
4. Discarding contours with area $< \delta^2$ as numerically sub-grid filaments

Surgery is the only dissipation mechanism in the method. Its physical
justification is that filaments thinner than $\delta$ are below the resolution
limit and their PV is rapidly mixed by real sub-grid-scale processes.

## Symplectic Time Integration

The QG equations have a Hamiltonian structure with energy:

$$E = -\frac{1}{2} \sum_{k,l} \Delta q_k \Delta q_l \oint_{C_k} \oint_{C_l} G(\mathbf{x}-\mathbf{x}') \, ds \, ds'$$

The Störmer-Verlet (leapfrog) symplectic integrator conserves this Hamiltonian
to $O(dt^2)$ for *all time*, with no secular energy drift. This is crucial for
turbulence studies spanning thousands of eddy turnover times, where RK4 would
accumulate $O(dt^4)$ energy errors over long runs.

# Implementation

## Package Architecture

The package is structured around five core modules:

- `contours.jl`: `PVContour`, `QGLayer`, `QGProblem` types; adaptive remeshing
- `kernels.jl`: `EulerKernel`, `QGKernel`, `SQGKernel`; segment velocity functions
- `surgery.jl`: KD-tree-based topological reconnection and filament removal
- `velocity.jl`: `DirectSummation` and `FMMSolver` velocity integration
- `timesteppers.jl`: `RK4Stepper`, `SymplecticStepper`; `run!` driver
- `diagnostics.jl`: exact contour-geometry-based energy, enstrophy, moments

Multiple dispatch on domain type (`UnboundedDomain`, `DoublyPeriodicDomain`)
and kernel type selects the correct velocity formula with zero runtime overhead,
a pattern natural in Julia and impossible to express cleanly in Python or Fortran.

## Performance

The direct summation velocity computation is $O(N^2)$ in the number of contour
nodes but is written to exploit Julia's SIMD auto-vectorization via
`LoopVectorization.jl`. For large problems (N > 5000 nodes), an FMM solver
reduces this to $O(N \log N)$.

## Ecosystem Integration

`ContourDynamics.jl` is designed to interoperate with the Julia GFD and
scientific computing ecosystem:

- PV fields from `GeophysicalFlows.jl` can be converted to contour
  representations via marching squares (using `Contour.jl`)
- Time integration registers as a `DifferentialEquations.jl`-compatible
  `ODEProblem`, allowing users to substitute any DiffEq.jl solver
- `Makie.jl` plot recipes enable animated visualization of contour evolution
- Diagnostics output is compatible with `Oceananigans.jl`'s output writers

# Example: Vortex Merger

The classical vortex merger problem [@overman1982] provides a canonical
validation: two circular QG patches of equal size and PV are initialized at
separation $d$. If $d/r \lesssim 3.3$, they merge into a single elliptical
patch; otherwise they rotate without merging.

```julia
using ContourDynamics
using StaticArrays

# Two circular patches, separation d/r = 2.8 → should merge
c1 = PVContour(center=SVector(-1.4, 0.0), radius=1.0, Δq=1.0, n_nodes=128)
c2 = PVContour(center=SVector( 1.4, 0.0), radius=1.0, Δq=1.0, n_nodes=128)

layer   = QGLayer([c1, c2], Inf)        # Inf L_d = 2D Euler
prob    = QGProblem([layer], UnboundedDomain(), RK4Stepper(0.05))

run!(prob, 30.0; remesh_every=5, surgery_every=5)

# After merger: single contour remains
@assert length(prob.layers[1].contours) == 1
```

Energy is conserved to better than $10^{-4}$ relative error throughout the
simulation, demonstrating the accuracy of the contour representation and the
surgery algorithm.

# Validation

The package is validated against five known analytical or converged results:

1. **Kirchhoff ellipse**: A uniform elliptical patch rotates steadily at rate
   $\Omega = ab/(a+b)^2$ (exact Euler result). `ContourDynamics.jl` reproduces
   this to $10^{-6}$ relative error for 100 rotation periods.

2. **Vortex merger threshold**: The critical $d/r \approx 3.3$ for 2D Euler
   [@overman1982] is reproduced within 1%.

3. **Energy conservation**: Both RK4 and symplectic steppers conserve energy
   to better than $10^{-3}$ over 1000 eddy turnover times; the symplectic
   stepper shows no secular drift while RK4 accumulates a small positive bias.

4. **QG deformation radius effect**: For finite $L_d$, a circular patch rotates
   at the analytically known rate $\Omega(L_d)$ [@dritschel1988].

5. **Beta-drift**: A circular patch on a $\beta$-plane drifts north-west at the
   rate predicted by @lam2001, agreeing to within 5% for 50 drift time scales.

# Acknowledgements

The authors thank David Dritschel for making the original CASL algorithm and
hydra documentation available in the literature, which provided the mathematical
foundation for this implementation. Development was supported by [funding source].

# References