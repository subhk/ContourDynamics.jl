---
title: 'ContourDynamics.jl: A Julia package for Lagrangian vortex dynamics via contour dynamics and surgery'
tags:
  - Julia
  - geophysical fluid dynamics
  - quasi-geostrophic
  - surface quasi-geostrophic
  - vortex dynamics
  - contour dynamics
  - contour surgery
  - potential vorticity
authors:
  - name: Subhajit Kar
    orcid: 0000-0001-9737-3345
    affiliation: 1
affiliations:
  - name: Tel Aviv University
    index: 1
date: 4 April 2026
bibliography: paper.bib
---

# Summary

The large-scale circulation of the atmosphere and ocean is populated by
long-lived, coherent vortices — Gulf Stream rings, polar stratospheric vortices,
Mediterranean eddies — that persist for months or years while interacting,
merging, and shedding thin filaments of fluid [@mcwilliams1984]. The theoretical
framework that explains why these structures are so robust centres on a quantity
called *potential vorticity* (PV): a scalar field, carried by each fluid parcel,
that combines information about the parcel's spin, the background planetary
rotation, and the density stratification of its environment
[@pedlosky1987; @vallis2017]. In the absence of friction and diabatic processes,
PV is materially conserved — it travels with the fluid rather than diffusing
through it. This conservation law is what gives geophysical vortices their
remarkable longevity.

In many idealised studies the PV distribution is taken to be
*piecewise-constant*: uniform patches separated by sharp boundaries, much like
coloured ink blots in a slowly stirred fluid. Under this simplification,
something elegant happens. Because the PV inside each patch never changes, the
entire dynamics reduce to tracking the *boundaries alone* — the contour lines
between patches. The interior flow at any instant is fully determined by the
shape of these boundaries through the Green's function of the PV inversion
operator. This is the idea behind *contour dynamics*, first introduced by
@zabusky1979: replace two-dimensional area integrals with one-dimensional
line integrals, and advect the boundary nodes with the velocity they induce.
The method is exact — it introduces no numerical diffusion and preserves PV
to machine precision.

`ContourDynamics.jl` is a Julia package that implements contour dynamics and
contour surgery [@dritschel1988] for four physical regimes — 2D Euler, surface
quasi-geostrophic (SQG), single-layer quasi-geostrophic (QG), and N-layer QG —
on both unbounded and doubly-periodic domains.

# Statement of Need

This combination of exactness and geometric elegance has made contour dynamics
a cornerstone of idealised geophysical fluid dynamics research for over three
decades. It has enabled landmark studies of the inverse energy cascade in 2D
turbulence [@dritschel2008], the filamentation and stripping of
quasi-geostrophic vortices [@dritschel1989], and the formation of sharp
temperature fronts in surface quasi-geostrophic flows
[@held1995; @scott2014]. The method excels precisely where grid-based solvers
struggle: long-time vortex interaction problems in which pseudospectral
diffusion artificially dissipates the fine-scale filamentary structures that
contour dynamics preserves exactly.

Despite this scientific impact, the method has remained difficult to access.
Widely used implementations are tied to legacy Fortran codes that are not
distributed as maintained open-source software, and in-house codes written by
individual research groups tend to be unpublished and limited to a single flow
regime. `ContourDynamics.jl` fills this gap with a performant, extensible Julia
implementation that makes contour dynamics reproducible and accessible to a new
generation of researchers.

# State of the Field

The primary reference implementations are Dritschel's original Fortran codes
[@dritschel1988; @dritschel1989; @dritschel1997], which remain the
methodological gold standard but are not publicly available as installable
packages. Grid-based alternatives such as GeophysicalFlows.jl (pseudospectral)
cover the same physical regimes but introduce numerical diffusion that limits
their fidelity for fine-scale structures. Point-vortex codes avoid grids but
sacrifice the exact PV conservation and topological surgery that contour
dynamics provides.

`ContourDynamics.jl` is, to our knowledge, the first publicly available,
open-source contour dynamics package that unifies multiple kernel types
(Euler, SQG, QG, N-layer QG), periodic domains with Ewald summation, and GPU
acceleration in a single, extensible codebase.

# Method

The central idea of contour dynamics is to replace the area integral of the
Green's function with a sum of line integrals along contour boundaries, via
Green's theorem:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} G(\mathbf{x} - \mathbf{x}') \, d\mathbf{x}'$$

Here $\Delta q_k$ is the PV jump across contour $C_k$ and $G$ is the Green's
function of the PV inversion operator. This formulation reduces the problem
from two dimensions to one. The package provides four Green's functions,
spanning a natural hierarchy of physical complexity:

| Kernel | Green's function | Physical regime |
|--------|-----------------|-----------------|
| 2D Euler | $G(r) = -\frac{1}{2\pi}\ln r$ | Barotropic vortex dynamics |
| SQG | $G(r) = -\frac{1}{2\pi r}$ | Surface temperature fronts |
| QG | $G(r) = -\frac{1}{2\pi} K_0(r/L_d)$ | Stratified vortices with deformation radius $L_d$ |
| N-layer QG | Eigenmode decomposition | Baroclinic vortex interactions |

Each kernel uses the most accurate available integration method: closed-form
antiderivatives for Euler and SQG, and singular subtraction with Gauss-Legendre
quadrature for QG. On doubly-periodic domains, Ewald summation [@dritschel1997]
decomposes the slowly-convergent lattice sum into rapidly-convergent real-space
and Fourier-space components.

As contours evolve, they develop exponentially thinning filaments that
eventually become unresolvable. Contour surgery [@dritschel1988] keeps the
calculation well-posed through four operations applied at regular intervals:
redistributing nodes at uniform arc-length spacing (remeshing), detecting
nearly-touching segments via spatial hashing, reconnecting them to change the
contour topology, and removing sub-grid filaments below an area threshold. This
cycle preserves topological consistency and integral invariants over arbitrarily
long integrations.

# Software Design

The package maps the physical hierarchy directly onto Julia's type system:
kernel types (`EulerKernel`, `QGKernel`, `SQGKernel`, `MultiLayerQGKernel`)
and domain types (`UnboundedDomain`, `PeriodicDomain`) select the correct
velocity formula at compile time via multiple dispatch. Node positions use
`SVector{2,T}` from StaticArrays.jl for efficient, allocation-free arithmetic.
Time integration uses a classical 4th-order Runge-Kutta scheme or a leapfrog
scheme with Robert-Asselin filtering.

For large problems, an adaptive treecode provides $O(N \log N)$ velocity
evaluation, replacing the $O(N^2)$ direct summation. GPU-accelerated evaluation
is available for the Euler kernel via KernelAbstractions.jl. All diagnostics —
energy, circulation, enstrophy, angular momentum, and ellipse geometry — are
computed analytically from the contour geometry via Green's theorem, without
gridding.

Optional package extensions integrate with DifferentialEquations.jl (ODE solver
bridge), Makie.jl (visualisation), RecordedArrays.jl (time-series recording),
and JLD2.jl (checkpointing), without adding heavy dependencies to the core
package.

# Validation

The package is tested against known analytical solutions and conservation laws:

- **Kirchhoff ellipse**: a uniformly rotating elliptical vortex patch, the
  classical benchmark for contour dynamics; the computed rotation period, aspect
  ratio, and circulation are preserved over full rotation cycles.
- **Integral invariants**: energy, area, circulation, and centroid position
  are conserved to $O(10^{-6})$ over $10^4$ RK4 time steps for Euler, QG,
  and periodic QG configurations.
- **Surgery correctness**: topological reconnection produces valid daughter
  contours with area and circulation preserved to within the surgery tolerance.
- **Treecode accuracy**: the $O(N\log N)$ treecode reproduces direct-summation
  velocities to relative error below $10^{-3}$.

The full test suite comprises over 370 assertions across 11 test files, run
on Julia 1.10--1.12 across Linux and macOS via continuous integration.

# Research Impact Statement

`ContourDynamics.jl` makes reproducible Lagrangian vortex dynamics experiments
accessible for the first time as open-source software. Example scripts
reproduce classical results from the literature — vortex merger, filamentation,
beta-plane drift, two-layer baroclinic dynamics — and serve as starting points
for new research. By unifying multiple kernel types, periodic domains, and GPU
acceleration in a tested, documented Julia package, it lowers the barrier to
entry for researchers investigating vortex dynamics across geophysical flow
regimes.

# Acknowledgements

The author thanks David Dritschel for making the original contour surgery
algorithm and its mathematical foundations available in the literature, which
this implementation closely follows.

# References
