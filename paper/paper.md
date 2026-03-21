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

Geophysical fluid dynamics (GFD) is governed at large scales by the
quasi-geostrophic (QG) equations, in which the flow is entirely determined by
the distribution of potential vorticity (PV). A natural and powerful way to
numerically integrate the QG equations is to represent the PV field as a
collection of *patches* — regions of piecewise-constant PV — whose boundaries
evolve as material contours. This *contour dynamics* approach, introduced by
@zabusky1979 for the 2D Euler equations and extended to the QG system by
@dritschel1988, has three compelling properties for idealized GFD studies:
(1) it is *exactly* Lagrangian, conserving PV along material surfaces without
numerical diffusion; (2) the underlying equations are *Hamiltonian*, so that
symplectic time-steppers (e.g. leapfrog) conserve energy to high accuracy
between surgery events, while even non-symplectic schemes like RK4 benefit
from the absence of numerical diffusion; and (3) it automatically
concentrates resolution on the dynamically active contour boundaries rather
than wasting computation on quiescent interior regions.

`ContourDynamics.jl` provides the first modern, open-source Julia
implementation of this method, supporting three flow regimes — 2D Euler,
single-layer quasi-geostrophic (QG) with arbitrary Rossby deformation radius $L_d$,
and N-layer QG — in both unbounded and doubly-periodic domains. The package
includes adaptive node management, *contour surgery* [@dritschel1988] for
long-time topologically-stable integrations, and grid-free diagnostics
(area, circulation, enstrophy, and vortex geometry computed exactly from
contour nodes; energy via Gauss-Legendre quadrature of contour integrals).

# Statement of Need

The contour dynamics/surgery framework has underpinned foundational GFD research
for over three decades, including studies of 2D turbulence vortex mergers and
the inverse energy cascade [@dritschel2008], filamentation and stripping of QG
vortices [@dritschel1989], and stratospheric vortex dynamics. Despite this rich
scientific legacy, the primary software implementing these methods is written in
Fortran and distributed on a contact-only basis with no public repository or
open license.

The Julia GFD ecosystem has grown substantially in recent years, with
`GeophysicalFlows.jl` providing GPU-accelerated pseudospectral QG solvers.
However, **no Julia package implements Lagrangian contour dynamics**, leaving a
methodological gap precisely where the method has its greatest advantages:
long-time vortex interaction studies where pseudospectral diffusion artificially
dissipates thin filaments.

`ContourDynamics.jl` fills this gap with a performant, extensible Julia
implementation that integrates with the broader Julia GFD ecosystem.

# Mathematical Formulation

## The Quasi-Geostrophic Equations

In the single-layer QG system, the PV $q$ and streamfunction $\psi$ satisfy:

$$\frac{\partial q}{\partial t} + J(\psi, q) = 0, \qquad q = \nabla^2\psi - \frac{\psi}{L_d^2}$$

where $J$ is the Jacobian operator and $L_d$ is the Rossby deformation radius.
The limiting case $L_d \to \infty$ recovers the 2D Euler equations.

## Contour Representation and Velocity Inversion

The PV field is represented as piecewise-constant:

$$q(\mathbf{x}, t) = \sum_k \Delta q_k \, \mathbf{1}_{D_k}(\mathbf{x})$$

where $D_k$ is the region enclosed by the $k$-th contour and $\Delta q_k$ is
its PV jump. Converting the area integral to a boundary integral via Green's
theorem, the velocity at any point $\mathbf{x}$ is:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} \nabla^\perp G(\mathbf{x} - \mathbf{x}') \, ds'$$

The Green's functions are:

| Regime | $G(r)$ |
|--------|--------|
| 2D Euler ($L_d \to \infty$) | $\frac{1}{2\pi}\ln r$ |
| QG (finite $L_d$) | $-\frac{1}{2\pi} K_0(r/L_d)$ |

For 2D Euler, the velocity induced by a straight segment has a closed-form
analytical expression. For QG, singular subtraction separates the log singularity
(integrated analytically) from the smooth remainder (5-point Gauss-Legendre
quadrature).

## Contour Surgery

Contour surgery [@dritschel1988] resolves the exponential filamentation of
contours by: (1) building a spatial index of all nodes, (2) finding non-adjacent
node pairs within the surgery distance $\delta$, (3) performing topological
reconnection (splitting or merging contours), and (4) removing sub-grid filaments.

# Implementation

## Package Architecture

The package uses trait-based dispatch on kernel types (`EulerKernel`, `QGKernel`,
`MultiLayerQGKernel`) to select the correct velocity formula with zero runtime
overhead. Core types include:

- `PVContour{T}`: parameterized contour with `Vector{SVector{2,T}}` nodes
- `ContourProblem{K,D,T}`: kernel + domain + contours
- `MultiLayerContourProblem{N,K,D,T}`: N-layer coupled problem with `NTuple` layers

The N-layer QG implementation uses eigendecomposition of the coupling matrix to
decompose into independent vertical modes, each using a QG-like kernel with an
effective deformation radius.

## Ecosystem Integration

Package extensions provide optional integration with DifferentialEquations.jl,
Makie.jl, and GeophysicalFlows.jl without imposing heavy dependencies on
the core package.

# Validation

The package is validated against known analytical results:

1. **Kirchhoff ellipse**: rotation at the exact rate $\Omega = ab/(a+b)^2 q$,
   reproduced to $10^{-4}$ relative error over one full period.

2. **Conservation**: energy, area, circulation, and centroid of a circular
   steady-state patch are conserved to $10^{-6}$ over $10^4$ RK4 timesteps.

3. **QG-Euler limit**: QG kernel with $L_d \gg R$ reproduces Euler velocities
   to within 10%.

# Acknowledgements

The authors thank David Dritschel for making the original contour surgery
algorithm available in the literature, which provided the mathematical
foundation for this implementation.

# References
