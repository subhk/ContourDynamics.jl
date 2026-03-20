# API Reference

## Types

```@docs
PVContour
EulerKernel
QGKernel
MultiLayerQGKernel
UnboundedDomain
PeriodicDomain
ContourProblem
MultiLayerContourProblem
SurgeryParams
RK4Stepper
LeapfrogStepper
```

## Velocity Computation

```@docs
velocity!
velocity
segment_velocity
```

## Time Integration

```@docs
timestep!
evolve!
resize_buffers!
```

## Surgery

```@docs
surgery!
remesh
```

## Diagnostics

```@docs
vortex_area
centroid
ellipse_moments
energy
enstrophy
circulation
angular_momentum
```

## Utilities

```@docs
nnodes
nlayers
total_nodes
arc_lengths
```
