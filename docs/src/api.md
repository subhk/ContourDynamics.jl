# API Reference

This section lists the public API by topic.

If you are new to the package, the most common entry points are:

- `Problem` for the high-level convenience interface
- `ContourProblem` and `MultiLayerContourProblem` for lower-level setup
- `evolve!` to run a simulation
- `energy`, `circulation`, and `vortex_area` for diagnostics

If you want a worked example before diving into the full reference, start with
the [Euler tutorial](/tutorial_euler).

Use the left sidebar to jump directly to the topic you want:

- [Types](/api/types)
- [Velocity & Acceleration](/api/velocity)
- [Time Integration](/api/time_integration)
- [Surgery](/api/surgery)
- [Diagnostics](/api/diagnostics)
- [Helpers](/api/helpers)
- [Periodic & Ewald](/api/periodic_ewald)
- [Devices](/api/devices)
- [Internals](/api/internals)

## Topic Guide

### Types

Core public structs including kernels, contours, domains, problems, and steppers.

Open: [Types](/api/types)

### Velocity & Acceleration

Pointwise and batched velocity APIs, plus the current status of treecode, FMM, and GPU support.

Open: [Velocity & Acceleration](/api/velocity)

### Time Integration

Timestep and evolution entry points.

Open: [Time Integration](/api/time_integration)

### Surgery

Remeshing, reconnection, and filament-removal functions.

Open: [Surgery](/api/surgery)

### Diagnostics

Energy, circulation, enstrophy, geometry, and related contour diagnostics.

Open: [Diagnostics](/api/diagnostics)

### Helpers

Contour utilities and shape constructors.

Open: [Helpers](/api/helpers)

### Periodic & Ewald

Periodic-domain helpers and Ewald cache/setup routines.

Open: [Periodic & Ewald](/api/periodic_ewald)

### Devices

CPU/GPU device types and array-constructor helpers.

Open: [Devices](/api/devices)

### Internals

Lower-level functions documented for advanced users and developers.

Open: [Internals](/api/internals)
