# API Reference

This page lists the public API by topic.

If you are new to the package, the most common entry points are:

- `Problem` for the high-level convenience interface
- `ContourProblem` and `MultiLayerContourProblem` for lower-level setup
- `evolve!` to run a simulation
- `energy`, `circulation`, and `vortex_area` for diagnostics

If you want a worked example before diving into the full reference, start with
the [Euler tutorial](/tutorial_euler).

## Accelerator Status

Current large-problem behavior is:

- single-layer unbounded CPU: direct for small problems, treecode for large problems
- single-layer unbounded CPU with experimental proxy FMM enabled: optional experimental FMM path
- single-layer periodic CPU: treecode for large problems in normal use; the current periodic `_fmm_velocity!` method is still a direct fallback
- multi-layer CPU: treecode for large problems in normal use; the current multi-layer `_fmm_velocity!` method is still a direct fallback
- GPU: supported for the Euler and SQG kernels on unbounded domains

So, at the moment, the production accelerator story is treecode plus the direct
GPU unbounded-kernel path. The proxy FMM remains experimental and is not the active
production path for periodic or multi-layer problems.

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
Problem
```

### Accessors

```@docs
contours
kernel
domain
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

The surgery API handles remeshing, reconnection, and filament removal. Most
users only need `SurgeryParams` and `surgery!`; the lower-level functions are
mainly useful if you want to customize the surgery pipeline.

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

## Shape Helpers

```@docs
circular_patch
elliptical_patch
rankine_vortex
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

These functions are documented for developers and advanced users. They are not
part of the stable high-level API.

```@docs
ContourDynamics._collect_all_nodes!
ContourDynamics._scatter_nodes!
ContourDynamics._scatter_shifted!
ContourDynamics._expint_e1
ContourDynamics._segment_min_dist2
ContourDynamics._best_stitch_nodes
ContourDynamics._check_spanning_proximity
```
