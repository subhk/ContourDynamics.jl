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
advantage for studying thin filaments, vortex mergers, and long-time
dynamics that grid-based methods tend to smear out.

`ContourDynamics.jl` is a Julia implementation of this method, covering four
flow regimes — 2D Euler, surface quasi-geostrophic (SQG), single-layer
quasi-geostrophic (QG), and N-layer QG — on both unbounded and doubly-periodic
domains. It includes the full *contour surgery* algorithm of @dritschel1988 for
topologically stable long-time integrations, and computes diagnostics (energy,
circulation, enstrophy, angular momentum, vortex geometry) directly from the
contour geometry without any gridding.

# Statement of Need

Contour dynamics has been a cornerstone of idealized geophysical fluid dynamics
research for over three decades, enabling studies of vortex mergers and the
inverse energy cascade in 2D turbulence [@dritschel2008], filamentation and
stripping of QG vortices [@dritschel1989], and front formation in surface
quasi-geostrophic flows [@held1995; @scott2014]. Yet widely used
implementations remain tied to legacy Fortran codes that are difficult to
extend, reproduce, or integrate with modern scientific computing workflows.

The Julia GFD ecosystem has matured substantially, but open-source tooling for
Lagrangian contour dynamics is absent. This leaves a methodological gap
precisely where the method excels: long-time vortex interaction studies where
pseudospectral diffusion artificially dissipates the fine-scale structures that
contour dynamics preserves exactly.

`ContourDynamics.jl` fills this gap with a performant, extensible
implementation that integrates with the broader Julia ecosystem through
optional package extensions for DifferentialEquations.jl, Makie.jl,
GeophysicalFlows.jl, RecordedArrays.jl, and JLD2.jl.

# Physics Kernels

The velocity at a contour node $\mathbf{x}$ is obtained by summing boundary
integrals over all PV contours:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} \nabla^\perp G(\mathbf{x} - \mathbf{x}') \, ds'$$

where $\Delta q_k$ is the PV jump across contour $C_k$ and $G$ is the
Green's function of the PV inversion operator. The package supports four
Green's functions:

| Kernel | Green's function | Inversion operator |
|--------|-----------------|---------------------|
| 2D Euler | $G(r) = -\frac{1}{2\pi}\ln r$ | $\nabla^2 \psi = q$ |
| SQG | $G(r) = -\frac{1}{2\pi r}$ | $(-\nabla^2)^{1/2}\psi = \theta$ |
| QG | $G(r) = -\frac{1}{2\pi} K_0(r/L_d)$ | $(\nabla^2 - L_d^{-2})\psi = q$ |
| N-layer QG | Eigenmode decomposition | $q_i = \sum_j C_{ij}\psi_j$ |

Each segment integral is evaluated with the most accurate method available.
For 2D Euler, the integral has a *closed-form antiderivative* — no
quadrature error. The SQG kernel $1/r$ is regularized to $1/\sqrt{r^2+\delta^2}$
to smooth the boundary singularity, yielding an exact $\operatorname{arcsinh}$
antiderivative [@held1995; @constantin1994]. For QG, singular subtraction
separates the $\ln r$ singularity (integrated analytically) from the smooth
remainder $K_0(r/L_d) + \ln(r/L_d)$ (5-point Gauss-Legendre quadrature). The
N-layer QG system uses eigendecomposition of the coupling matrix to reduce to
independent vertical modes, each handled by one of the above kernels.

# Periodic Domains and Ewald Summation

On doubly-periodic domains, the Green's function must account for all periodic
images. Direct summation converges slowly, so the package uses *Ewald
splitting* [@dritschel1997] to decompose the periodic sum into a
rapidly-convergent real-space sum (Gaussian-damped) and a rapidly-convergent
Fourier-space sum. The same singular-subtraction strategy applies: the
unbounded segment velocity (exact antiderivative) handles the central-image
singularity, and 3-point Gauss-Legendre quadrature integrates the smooth
periodic correction.

Ewald caches are precomputed, keyed by domain size and kernel type, and stored
with lazy thread-safe access so that repeated velocity evaluations incur no
redundant setup cost.

# Contour Surgery

Long-time integrations produce exponentially thinning filaments that
eventually become unresolvable. Contour surgery [@dritschel1988] resolves this
by periodically: (1) redistributing nodes along each contour to maintain
uniform resolution (remeshing), (2) detecting non-adjacent segments that
approach within a surgery distance $\delta$ via a spatial hash index,
(3) performing topological reconnection — splitting a single contour into two,
or merging two same-PV contours into one, and (4) removing sub-grid filaments
whose area falls below a threshold. This allows arbitrarily long integrations
without loss of topological consistency.

# Implementation

The package uses trait-based dispatch on kernel types (`EulerKernel`,
`SQGKernel`, `QGKernel`, `MultiLayerQGKernel`) and domain types
(`UnboundedDomain`, `PeriodicDomain`) to select the correct velocity formula
with zero runtime overhead. Node positions are stored as `SVector{2,T}` from
StaticArrays.jl for cache-friendly, allocation-free arithmetic. Time
integration is provided by a 4th-order Runge-Kutta scheme (recommended for
most applications) and a symplectic leapfrog scheme with Robert-Asselin
filtering.

All diagnostics — area, centroid, circulation, enstrophy, angular momentum,
and ellipse moments — are computed analytically from the contour geometry via
Green's theorem. Energy is evaluated as a double contour integral using
Gauss-Legendre quadrature over segment pairs.

Package extensions provide optional integration with DifferentialEquations.jl
(bridging to any ODE solver), Makie.jl (animated contour evolution),
GeophysicalFlows.jl (grid-contour conversion), RecordedArrays.jl (time-series
recording), and JLD2.jl (checkpoint save/load) — without imposing heavy
dependencies on the core package.

# Validation

The package is validated against known analytical results:

1. **Kirchhoff ellipse**: an elliptical Euler vortex patch rotates at the
   exact rate $\Omega = ab\,q/(a+b)^2$. The computed rotation rate, aspect
   ratio, and circulation are preserved over one full rotation period.

2. **Conservation laws**: energy, area, circulation, and centroid position of
   a circular steady-state patch are conserved to $O(10^{-6})$ over $10^4$
   RK4 time steps.

3. **QG–Euler limit**: the QG kernel with $L_d \gg R$ reproduces Euler
   velocities to within 10\%, confirming correct singular subtraction.

# Acknowledgements

The author thanks David Dritschel for making the original contour surgery
algorithm and its mathematical foundations available in the literature, which
this implementation closely follows.

# References
