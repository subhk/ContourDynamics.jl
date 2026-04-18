# Theory & Method

This section describes the mathematical foundations of the contour dynamics method as implemented in ContourDynamics.jl.

The goal here is to explain the method without assuming too much notation up
front. Each page keeps the equations, but also introduces the symbols in plain
language. For example:

- ``\psi`` is the streamfunction
- ``\mathbf{u}`` is the velocity field
- ``q`` is potential vorticity (or vorticity in the Euler case)
- ``G`` is the Green's function for the inversion operator
- ``\mathbf{x}`` is the point where you want the velocity
- ``\mathbf{x}'`` is an integration point on a contour or in an area integral
- ``C`` is a contour boundary

If you mainly want to know how the package computes things, start with
[Contour Dynamics](/theory/contour_dynamics). If you want the numerical details
for periodic domains, surgery, or multi-layer QG, use the more specific pages
in the sidebar.

Use the left sidebar to jump directly to the topic you want:

- [Contour Dynamics](/theory/contour_dynamics)
- [Ewald Summation](/theory/ewald_summation)
- [Contour Surgery](/theory/contour_surgery)
- [Multi-Layer QG](/theory/multilayer_qg)
- [Time Integration](/theory/time_integration)
- [References](/theory/references)

## Topic Guide

### Contour Dynamics

Covers the core contour-integral formulation, segment discretization, and how Euler, QG, and SQG kernels are evaluated.

Open: [Contour Dynamics](/theory/contour_dynamics)

### Ewald Summation

Explains periodic Green's functions, Ewald splitting, and the periodic decompositions used for Euler, QG, and SQG.

Open: [Ewald Summation](/theory/ewald_summation)

### Contour Surgery

Explains remeshing, reconnection, filament removal, and the meaning of the surgery parameters.

Open: [Contour Surgery](/theory/contour_surgery)

### Multi-Layer QG

Summarizes the modal decomposition used to convert layer coupling into independent barotropic and baroclinic modes.

Open: [Multi-Layer QG](/theory/multilayer_qg)

### Time Integration

Describes the RK4 and leapfrog schemes used to advance contour nodes.

Open: [Time Integration](/theory/time_integration)

### References

Lists the main contour-dynamics, contour-surgery, and SQG references behind the implementation.

Open: [References](/theory/references)
