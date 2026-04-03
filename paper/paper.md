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
date: 2026
bibliography: paper.bib
---

# Summary

In two-dimensional geophysical flows, the dynamics are often controlled by
the distribution of potential vorticity (PV). When that distribution is
*piecewise-constant* — regions of uniform PV separated by sharp boundaries —
the entire evolution can be tracked by following the boundaries alone. This is
the idea behind *contour dynamics* [@zabusky1979]: replace area integrals over
the PV field with line integrals along contour boundaries, and advect the
boundary nodes with the resulting velocity. Because the PV is carried exactly
with the flow, the method introduces no numerical diffusion — a decisive
advantage for studying thin filaments, vortex mergers, and long-time dynamics
that grid-based methods tend to smear out.

`ContourDynamics.jl` is a Julia implementation of contour dynamics and contour
surgery [@dritschel1988], covering four flow regimes — 2D Euler, surface
quasi-geostrophic (SQG), single-layer quasi-geostrophic (QG), and N-layer
QG — on both unbounded and doubly-periodic domains. The package provides
topologically stable long-time integration with automatic filament removal,
GPU-accelerated velocity evaluation, and analytical diagnostics (energy,
circulation, enstrophy, angular momentum, vortex geometry) computed directly
from the contour geometry.

# Statement of Need

Contour dynamics has been a cornerstone of idealized geophysical fluid dynamics
research for over three decades, enabling studies of vortex mergers and the
inverse energy cascade in 2D turbulence [@dritschel2008], filamentation and
stripping of QG vortices [@dritschel1989], and front formation in surface
quasi-geostrophic flows [@held1995; @scott2014]. Yet widely used
implementations remain tied to legacy Fortran codes that are difficult to
extend, reproduce, or integrate with modern scientific computing workflows.

The Julia ecosystem for geophysical fluid dynamics has matured substantially,
but open-source tooling for Lagrangian contour dynamics is absent. This leaves
a methodological gap precisely where the method excels: long-time vortex
interaction studies where pseudospectral diffusion artificially dissipates the
fine-scale structures that contour dynamics preserves exactly.
`ContourDynamics.jl` fills this gap with a performant, extensible
implementation that integrates with the broader Julia ecosystem.

# Method

## Velocity computation

The contour dynamics velocity at a point $\mathbf{x}$ is obtained by applying
Green's theorem to convert the area integral of the Green's function into a
sum of line integrals along contour boundaries:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} G(\mathbf{x} - \mathbf{x}') \, d\mathbf{x}'$$

where $\Delta q_k$ is the PV jump across contour $C_k$, $G$ is the Green's
function of the PV inversion operator, and $d\mathbf{x}' = (dx', dy')$ is the
vector line element. This formulation replaces the $\nabla^\perp G$ integrand
of the Biot-Savart law (which has a strong $1/r$ singularity) with $G$ itself
(a weaker, integrable singularity), enabling exact analytical evaluation.

The package supports four Green's functions, each corresponding to a different
physical regime:

| Kernel | Green's function | Inversion operator |
|--------|-----------------|---------------------|
| 2D Euler | $G(r) = -\frac{1}{2\pi}\ln r$ | $\nabla^2 \psi = q$ |
| SQG | $G(r) = -\frac{1}{2\pi r}$ | $(-\nabla^2)^{1/2}\psi = \theta$ |
| QG | $G(r) = -\frac{1}{2\pi} K_0(r/L_d)$ | $(\nabla^2 - L_d^{-2})\psi = q$ |
| N-layer QG | Eigenmode decomposition | $q_i = \sum_j C_{ij}\psi_j$ |

Each segment contribution is evaluated with the most accurate method available.
For 2D Euler, the contour integral admits a closed-form antiderivative with no
quadrature error. The SQG kernel $1/r$ is regularized to
$1/\sqrt{r^2 + \delta^2}$, yielding an exact $\operatorname{arcsinh}$
antiderivative [@held1995; @constantin1994]. For QG, singular subtraction
separates the $\ln r$ singularity (integrated analytically via the Euler
antiderivative) from the smooth remainder $K_0(r/L_d) + \ln r$, which is
integrated with 5-point Gauss-Legendre quadrature. The N-layer QG system is
reduced to independent vertical modes via eigendecomposition of the coupling
matrix, with each mode handled by the Euler or QG kernel according to its
modal deformation radius.

## Periodic domains

On doubly-periodic domains, the Green's function must account for all periodic
images. The package uses Ewald splitting [@dritschel1997] to decompose the
slowly-convergent lattice sum into a rapidly-convergent real-space sum
(Gaussian-damped) and a rapidly-convergent Fourier-space sum. The same
singular-subtraction strategy applies: the unbounded segment velocity handles
the central-image singularity exactly, and 3-point Gauss-Legendre quadrature
integrates the smooth periodic correction. Ewald coefficients are precomputed
and cached with thread-safe lazy initialization.

## Contour surgery

Long-time integrations produce exponentially thinning filaments that eventually
become unresolvable. Contour surgery [@dritschel1988] resolves this through a
four-step cycle applied at regular intervals: (1) remeshing to redistribute
nodes at uniform arc-length spacing, (2) proximity detection via a spatial hash
index, (3) topological reconnection — splitting a single contour or merging two
same-PV contours, and (4) removal of sub-grid filaments below an area
threshold. This maintains topological consistency over arbitrarily long
integrations.

# Software Design

The package uses Julia's type system to dispatch on kernel types (`EulerKernel`,
`SQGKernel`, `QGKernel`, `MultiLayerQGKernel`) and domain types
(`UnboundedDomain`, `PeriodicDomain`), selecting the correct velocity formula
at compile time with zero runtime overhead. Node positions are stored as
`SVector{2,T}` from StaticArrays.jl for cache-friendly, allocation-free
arithmetic. Time integration is provided by a 4th-order Runge-Kutta scheme and
a second-order leapfrog scheme with Robert-Asselin filtering.

For problems with many contour nodes, an adaptive treecode provides $O(N
\log N)$ velocity evaluation via first-order Taylor expansion with an
opening-angle criterion, replacing the default $O(N^2)$ direct summation.
GPU-accelerated evaluation is available for the Euler kernel via
KernelAbstractions.jl, with a CUDA extension that handles device transfer
transparently.

All diagnostics — area, centroid, circulation, enstrophy, energy, angular
momentum, and ellipse moments — are computed analytically from the contour
geometry via Green's theorem, without gridding. Energy is evaluated as a double
contour integral with singular subtraction for self-interaction terms.

Optional package extensions integrate with DifferentialEquations.jl (ODE
solver bridge), Makie.jl (animated contour evolution), RecordedArrays.jl
(time-series recording), and JLD2.jl (checkpoint save/load), without imposing
heavy dependencies on the core package.

# Validation

The package is validated against known analytical solutions and conservation
laws:

- **Kirchhoff ellipse**: an elliptical Euler vortex patch rotates at the exact
  rate $\Omega = ab\,q/(a+b)^2$; the computed rotation period, aspect ratio,
  and circulation are preserved over full rotation cycles.
- **Integral invariants**: energy, area, circulation, and centroid position of
  steady-state patches are conserved to $O(10^{-6})$ over $10^4$ RK4 steps for
  Euler, QG, and periodic QG kernels.
- **QG--Euler limit**: QG velocities with $L_d \gg R$ converge to Euler
  velocities, confirming correct singular subtraction.
- **Surgery correctness**: filament removal preserves area and circulation to
  within the surgery tolerance; reconnection produces topologically valid
  daughter contours with consistent winding.
- **Periodic convergence**: Ewald-accelerated periodic velocities agree with
  unbounded velocities for isolated vortices in large domains; energy is
  conserved in periodic multi-layer QG integrations.
- **Treecode accuracy**: the treecode reproduces direct $O(N^2)$ velocities to
  relative error below $10^{-3}$ on multi-contour configurations.

The full test suite comprises over 370 assertions across 11 test files.

# Acknowledgements

The author thanks David Dritschel for making the original contour surgery
algorithm and its mathematical foundations available in the literature, which
this implementation closely follows.

# References
