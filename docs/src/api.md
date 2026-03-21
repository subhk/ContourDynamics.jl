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
find_close_segments
build_spatial_index
reconnect!
remove_filaments!
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

## Contour Helpers

```@docs
nnodes
nlayers
total_nodes
arc_lengths
next_node
beta_staircase
is_spanning
```

## Ewald Summation

```@docs
EwaldCache
build_ewald_cache
clear_ewald_cache!
```

## Internals

```@docs
ContourDynamics._collect_all_nodes
ContourDynamics._scatter_nodes!
ContourDynamics._scatter_shifted!
ContourDynamics._expint_e1
ContourDynamics._segment_min_dist2
```
