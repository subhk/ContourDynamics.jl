# API Reference

## Types

### Kernels

```@docs
EulerKernel
QGKernel
SQGKernel
MultiLayerQGKernel
```

### Contours and Domains

```@docs
PVContour
UnboundedDomain
PeriodicDomain
```

### Problem Structs

```@docs
ContourProblem
MultiLayerContourProblem
SurgeryParams
```

### Time Steppers

```@docs
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
ContourDynamics.find_close_segments
ContourDynamics.build_spatial_index
ContourDynamics.reconnect!
ContourDynamics.remove_filaments!
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
setup_ewald_cache!
clear_ewald_cache!
```

## Periodic Domains

```@docs
wrap_nodes!
```

## Device Types

```@docs
CPU
GPU
device_array
```

## Internals

```@docs
ContourDynamics._collect_all_nodes!
ContourDynamics._scatter_nodes!
ContourDynamics._scatter_shifted!
ContourDynamics._expint_e1
ContourDynamics._segment_min_dist2
ContourDynamics._best_stitch_nodes
ContourDynamics._check_spanning_proximity
```
