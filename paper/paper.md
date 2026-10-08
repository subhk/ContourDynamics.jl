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
  - name: University of Maryland, College Park, Maryland, USA
    index: 1
date: 25 July 2026
bibliography: paper.bib
---

# Summary

`ContourDynamics.jl` is a Julia package for simulating how vortices, swirling
regions of fluid, move and interact. It represents each vortex patch by its
boundary and advances those boundaries in time, so stretching, merging, and
splitting are followed directly. The package supports two-dimensional Euler,
single-layer and multilayer quasi-geostrophic (QG), beta-plane QG, and surface
quasi-geostrophic (SQG) flows, in unbounded and doubly periodic domains, with
Dritschel-style contour surgery to remesh, reconnect, and remove unresolved
filaments. The same problem description runs on threaded CPUs or NVIDIA GPUs,
and optional extensions connect it to time integrators, plotting, recording,
and checkpointing in the Julia ecosystem.

Coherent vortices are central to geophysical turbulence
[@mcwilliams1984; @dritschel2008]. In inviscid, adiabatic models their evolution
is governed by an advected scalar: vorticity, potential vorticity, or surface
buoyancy [@pedlosky1987; @vallis2017; @held1995]. When that scalar is piecewise
constant, contour dynamics replaces the area integrals of velocity inversion by
boundary integrals [@zabusky1979], which keeps sharp jumps exact without a fixed
grid.

# Statement of Need

Vortex merger, filamentation, frontogenesis, and layer interactions are
studied across two-dimensional Euler, QG, multilayer QG, and SQG models
[@dritschel1989; @held1995; @scott2014]. Comparing such processes across models
requires a solver whose initial conditions, diagnostics, and treatment of
unresolved scales stay the same while the inversion kernel changes. Contour
dynamics with surgery is well suited to this, but its established
implementations are Fortran research codes such as Dritschel's Hydra suite
[@hydra2023] or unpublished in-house codes, often written for one model and
without the packaging, documentation, and tests that make a method easy to
adopt or verify. Julia users have grid-based geophysical solvers
[@geophysicalflows2021] but no registered, maintained contour-dynamics package.

`ContourDynamics.jl` fills that gap for researchers and students working on
idealized inviscid patch dynamics. It provides a registered, documented, and
tested implementation in which the physical kernel, the domain, and the
execution device are interchangeable within one problem description, with a
reproducible verification script and optional connections to the Julia
analysis ecosystem.

# State of the Field

Contour dynamics and surgery build on established algorithms
[@zabusky1979; @dritschel1988; @dritschel1989]. Contour-advective semi-Lagrangian
(CASL) methods combine contour transport with grid-based inversion
[@dritschel1997]; Hydra [@hydra2023] implements this approach for many fluid
systems and the open-source [CALIB library](https://github.com/AnderOne/CALIB)
for a single-layer model, and an open Fortran implementation of
[contour surgery in multiply connected domains](https://github.com/rhodrin/contour-surgery-mc)
accompanies a study of bounded Euler flow. GeophysicalFlows.jl supplies Fourier
pseudospectral solvers for periodic geophysical flows on CPUs and GPUs
[@geophysicalflows2021]. In grid-based solvers the treatment of small scales
depends on resolution, viscosity, and filtering choices.

The contribution of `ContourDynamics.jl` is the integration of direct boundary
inversion and topology-changing contour management across the supported models
and domains in Julia. Direct inversion avoids contour-to-grid conversion and
also supports unbounded domains. This choice requires contour geometry, surgery,
and execution machinery distinct from a Fourier-grid solver, motivating a
separate package with optional ecosystem integrations. Direct all-pairs
interactions have quadratic cost in contour-node count at fixed inversion
settings; the implementation makes no general speed or accuracy superiority
claim over grid-based or hybrid methods.

# Method

Contour dynamics expresses the velocity at any point as a sum of boundary
integrals of the model's inversion kernel over the patch contours, weighted by
each contour's scalar jump. Velocity evaluation combines analytic segment
integrals with numerical quadrature on cubic arcs. Doubly periodic interactions
use truncated Ewald sums [@ewald1921], multilayer QG separates the coupled
inversion into vertical modes weighted by layer thickness, and the SQG kernel
requires a softening length that is distinct from the surgery cutoff. On the
beta plane, the background PV gradient is a staircase of spanning contours; the
inversion acts on their departure from a frozen straight reference, whose
velocity is added analytically. Nodes move as material points under classical
fourth-order Runge–Kutta stepping, with periodic wrapping after each step.

Surgery follows Dritschel's contour surgery [@dritschel1988]:
curvature-dependent remeshing redistributes nodes, compatible nearby contour
segments reconnect, and cleanup removes unresolved fragments. Closed-contour
remeshing corrects polygon area, but reconnection and removal can change
integral quantities, so results require checks against resolution, timestep,
and the surgery, softening, and Ewald settings. Kernel derivations, sign
conventions, quadrature rules, periodic zero modes, and remeshing details are
given in the
[web theory documentation](https://subhk.github.io/ContourDynamics.jl/dev/theory/).

# Software Design

Kernel and domain types select specialized numerical paths through Julia's
multiple dispatch. CPU and CUDA paths share scalar segment formulas, reducing
duplication when correcting or extending the numerical methods;
KernelAbstractions.jl supports device execution and CUDA.jl is loaded through
an optional extension. Surgery updates topology and resizes the time-stepping
buffers between steps, and multilayer surgery acts independently within each
layer, with coupling entering only through inversion. The Makie, JLD2,
RecordedArrays, and OrdinaryDiffEq integrations are likewise optional
extensions, so the core solver carries no dependency on them.

Geometric diagnostics use polygon formulas. Energy approximates the
model-specific Hamiltonian using $3\times3$ Gauss–Legendre quadrature over pairs
of straight segments, whereas velocity generally uses cubic arcs, so it is not
an exact invariant of the discrete time-stepping scheme. The contour-wise
enstrophy diagnostic omits cross-terms for arbitrary nested scalar jumps, as
documented in the diagnostic definitions. A verification script,
`paper/validate.jl`, compares circular-patch velocity and Hamiltonian values
for each kernel against independent references and checks convergence under
refinement, with results recorded in `paper/README.md`; the test suite covers
periodic inversion, layer coupling, surgery, CPU/GPU agreement, and
allocations.

# Research Impact Statement

The package ships literature-based initial conditions as runnable examples:
the perturbed-ellipse filamentation and nested-vortex merger cases of
@dritschel1988, the elliptical SQG vortex of @held1995, an upper-layer merger
with unequal layer depths following @polvani1989, and a beta-plane vortex
following @lam2001. The beta-plane example represents the background PV
gradient by material staircase contours and uses direct contour inversion,
which differs from the CASL inversion of the reference study. These examples
document physical configurations and numerical choices; quantitative
reproduction of published evolution requires separate convergence studies.

The intended impact is to make contour dynamics with surgery usable without
rewriting a solver: a student or researcher can set up a patch problem, switch
the inversion kernel, domain, or device, and compare results with the same
diagnostics and surgery settings. The verification script and test suite give
a documented baseline for such comparisons and for extending the method.

<!-- Author input needed before submission: add specific research use, including
an ongoing project, publication/preprint, external adoption, or a documented
research workflow. Do not claim adoption from the examples alone. -->

# References
