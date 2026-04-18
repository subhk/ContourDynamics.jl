# Theory & Method

This section describes the mathematical foundations of the contour dynamics method as implemented in ContourDynamics.jl.

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
