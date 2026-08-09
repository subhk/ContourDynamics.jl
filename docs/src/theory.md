# Theory & Method

This section describes the mathematical foundations of the contour dynamics method as implemented in ContourDynamics.jl.

This section introduces the method with the equations used by the
implementation and defines the main symbols as they appear. For example:

- ``\psi`` is the streamfunction
- ``\mathbf{u}`` is the velocity field
- ``q`` is potential vorticity (or vorticity in the Euler case)
- ``G`` is the Green's function for the inversion operator
- ``\mathbf{x}`` is the point where the velocity is evaluated
- ``\mathbf{x}'`` is an integration point on a contour or in an area integral
- ``C`` is a contour boundary

The complete cross-page reference is [Notation and Parameter
Glossary](theory/notation.md). Every theory page also defines symbols beside
the equation in which they first appear.

For the core formulation, start with
[Contour Dynamics](theory/contour_dynamics.md). Numerical details for periodic
domains, surgery, and multi-layer QG are covered in the topic pages listed in
the sidebar.

The left sidebar provides direct access to each topic:

- [Contour Dynamics](theory/contour_dynamics.md)
- [Notation and Parameter Glossary](theory/notation.md)
- [Ewald Summation](theory/ewald_summation.md)
- [Contour Surgery](theory/contour_surgery.md)
- [Multi-Layer QG](theory/multilayer_qg.md)
- [Time Integration](theory/time_integration.md)
- [References](theory/references.md)

## Topic Guide

### Notation and Parameters

Maps every recurring equation symbol to its physical meaning and corresponding
Julia field or keyword, including kernel, Ewald, surgery, time-integration,
multi-layer, and beta-plane parameters.

Open: [Notation and Parameter Glossary](theory/notation.md)

### Contour Dynamics

Covers the core contour-integral formulation, segment discretization, and how Euler, QG, and SQG kernels are evaluated.

Open: [Contour Dynamics](theory/contour_dynamics.md)

### Ewald Summation

Explains periodic Green's functions, Ewald splitting, and the periodic decompositions used for Euler, QG, and SQG.

Open: [Ewald Summation](theory/ewald_summation.md)

### Contour Surgery

Explains remeshing, reconnection, filament removal, and the meaning of the surgery parameters.

Open: [Contour Surgery](theory/contour_surgery.md)

### Multi-Layer QG

Summarizes the modal decomposition used to convert layer coupling into independent barotropic and baroclinic modes.

Open: [Multi-Layer QG](theory/multilayer_qg.md)

### Time Integration

Describes the RK4 scheme used to advance contour nodes.

Open: [Time Integration](theory/time_integration.md)

### References

Lists the main contour-dynamics, contour-surgery, and SQG references behind the implementation.

Open: [References](theory/references.md)
