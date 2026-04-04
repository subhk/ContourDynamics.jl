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

Many geophysical flows — ocean eddies, atmospheric vortices, weather fronts —
are governed by the distribution of a quantity called *potential vorticity* (PV),
which measures the local spin of a fluid parcel adjusted for stratification and
rotation. In idealized models, these flows often consist of patches of uniform
PV separated by sharp boundaries. When PV is piecewise-constant in this way, the
entire flow evolution can be determined by tracking only the patch boundaries,
rather than solving for the flow on a grid that fills the entire domain. This
boundary-tracking approach is called *contour dynamics* [@zabusky1979].

The key advantage of contour dynamics is that PV is transported exactly with the
flow, introducing no numerical diffusion. This makes the method ideally suited
for studying thin filaments, vortex mergers, and long-time dynamics —
phenomena that grid-based methods tend to smear out over time.

`ContourDynamics.jl` is a Julia package that implements contour dynamics and
contour surgery [@dritschel1988] for four physical regimes: 2D Euler, surface
quasi-geostrophic (SQG), single-layer quasi-geostrophic (QG), and N-layer QG.
It supports both unbounded and doubly-periodic domains, provides automatic
filament removal for long-time stability, and computes analytical diagnostics
(energy, circulation, enstrophy, angular momentum) directly from the contour
geometry.

# Statement of Need

Contour dynamics has been a foundational tool in idealized geophysical fluid
dynamics for over three decades, enabling studies of vortex mergers and the
inverse energy cascade in 2D turbulence [@dritschel2008], filamentation and
stripping of QG vortices [@dritschel1989], and front formation in surface
quasi-geostrophic flows [@held1995; @scott2014].

Yet widely used implementations remain tied to legacy Fortran codes that are
difficult to extend, reproduce, or integrate with modern scientific computing
workflows. The Julia ecosystem for geophysical fluid dynamics has matured
substantially — packages like GeophysicalFlows.jl provide pseudospectral solvers
— but open-source tooling for Lagrangian contour dynamics is absent. This leaves
a methodological gap precisely where the method excels: long-time vortex
interaction studies where pseudospectral diffusion artificially dissipates the
fine-scale structures that contour dynamics preserves exactly.
`ContourDynamics.jl` fills this gap.

# State of the Field

The primary existing implementations of contour dynamics are Dritschel's
original Fortran codes, which have been used extensively in the research
literature [@dritschel1988; @dritschel1989; @dritschel1997] but are not
publicly distributed as maintained open-source packages. Individual research
groups have developed in-house implementations, but these are typically
unpublished, single-purpose, and limited to one kernel type.

Grid-based alternatives such as GeophysicalFlows.jl (pseudospectral) handle
the same physical regimes but introduce numerical diffusion that limits their
accuracy for fine-scale structures. Vortex methods (e.g., point-vortex codes)
avoid grids but sacrifice the exact PV conservation and topological surgery that
contour dynamics provides.

`ContourDynamics.jl` is, to the best of our knowledge, the first publicly
available, open-source contour dynamics package that supports multiple kernel
types (Euler, SQG, QG, N-layer QG), periodic domains with Ewald summation, and
GPU acceleration — in a single, extensible codebase.

# Method

The velocity at a point $\mathbf{x}$ is obtained by converting the area
integral of the Green's function into line integrals along contour boundaries:

$$\mathbf{u}(\mathbf{x}) = \sum_k \Delta q_k \oint_{C_k} G(\mathbf{x} - \mathbf{x}') \, d\mathbf{x}'$$

where $\Delta q_k$ is the PV jump across contour $C_k$ and $G$ is the Green's
function of the PV inversion operator. The package supports four Green's
functions:

| Kernel | Green's function | Physical regime |
|--------|-----------------|-----------------|
| 2D Euler | $G(r) = -\frac{1}{2\pi}\ln r$ | Barotropic vortex dynamics |
| SQG | $G(r) = -\frac{1}{2\pi r}$ | Surface temperature fronts |
| QG | $G(r) = -\frac{1}{2\pi} K_0(r/L_d)$ | Stratified vortices with deformation radius $L_d$ |
| N-layer QG | Eigenmode decomposition | Baroclinic vortex interactions |

Each kernel uses the most accurate available integration method: closed-form
antiderivatives for Euler and SQG, and singular subtraction with Gauss-Legendre
quadrature for QG. On periodic domains, Ewald splitting [@dritschel1997]
accelerates the lattice sum by decomposing it into rapidly-convergent real-space
and Fourier-space components.

Long-time integrations produce exponentially thinning filaments. Contour surgery
[@dritschel1988] handles this through a four-step cycle: (1) remeshing to
redistribute nodes at uniform arc-length spacing, (2) proximity detection via
spatial hashing, (3) topological reconnection of nearby contour segments, and
(4) removal of sub-grid filaments. This maintains topological consistency over
arbitrarily long simulations.

# Software Design

The package uses Julia's multiple dispatch to select the correct velocity
formula based on kernel and domain types at compile time, with zero runtime
overhead. Node positions are stored as `SVector{2,T}` from StaticArrays.jl for
cache-friendly arithmetic. Time integration uses 4th-order Runge-Kutta or
leapfrog with Robert-Asselin filtering.

For large problems, an adaptive treecode provides $O(N \log N)$ velocity
evaluation, replacing the default $O(N^2)$ direct summation. GPU-accelerated
evaluation is available for the Euler kernel via KernelAbstractions.jl.

All diagnostics — area, centroid, circulation, enstrophy, energy, angular
momentum, and ellipse moments — are computed analytically from the contour
geometry via Green's theorem, without gridding.

Optional package extensions integrate with DifferentialEquations.jl (ODE solver
bridge), Makie.jl (visualization), RecordedArrays.jl (time-series recording),
and JLD2.jl (checkpointing), without imposing heavy dependencies on the core
package.

# Validation

The package is validated against analytical solutions and conservation laws:

- **Kirchhoff ellipse**: an elliptical vortex patch rotates at the exact
  theoretical rate; aspect ratio and circulation are preserved over full cycles.
- **Integral invariants**: energy, area, circulation, and centroid position
  are conserved to $O(10^{-6})$ over $10^4$ time steps for Euler, QG, and
  periodic QG.
- **Surgery correctness**: reconnection produces topologically valid daughter
  contours; area and circulation are preserved to within the surgery tolerance.
- **Treecode accuracy**: reproduces direct $O(N^2)$ velocities to relative
  error below $10^{-3}$.

The full test suite comprises over 370 assertions across 11 test files, run on
Julia 1.10--1.12 across Linux and macOS via continuous integration.

# Research Impact Statement

`ContourDynamics.jl` enables reproducible computational experiments in
Lagrangian vortex dynamics that were previously confined to unpublished,
single-use Fortran codes. The package has been developed over multiple months
with continuous integration, comprehensive test coverage, and documentation.
Example scripts reproduce classical results from the literature (vortex
merger, filamentation, beta-drift, two-layer baroclinic dynamics), providing
ready-made starting points for new research. The combination of multiple kernel
types, periodic domains, and GPU support in a single open-source Julia package
lowers the barrier to entry for researchers investigating vortex dynamics across
geophysical flow regimes.

# AI Usage Disclosure

Generative AI tools (Anthropic Claude) were used during the development of this
software to assist with code implementation, documentation drafting, and test
development. All AI-generated outputs were reviewed, edited, and validated by
the human author. Core algorithmic and design decisions were made by the author
based on the published literature.

# Acknowledgements

The author thanks David Dritschel for making the original contour surgery
algorithm and its mathematical foundations available in the literature, which
this implementation closely follows.

# References
