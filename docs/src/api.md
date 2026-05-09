# API Reference

This section lists the public API by topic.

Common entry points:

- `Problem` for the high-level convenience interface
- `ContourProblem` and `MultiLayerContourProblem` for lower-level setup
- `evolve!` to run a simulation
- `energy`, `circulation`, and `vortex_area` for diagnostics

For an end-to-end example before using the full reference, see the
[Euler tutorial](tutorial_euler.md).

The left sidebar provides direct access to each topic:

- [Types](api/types.md)
- [Velocity & Acceleration](api/velocity.md)
- [Time Integration](api/time_integration.md)
- [Surgery](api/surgery.md)
- [Diagnostics](api/diagnostics.md)
- [Helpers](api/helpers.md)
- [Periodic & Ewald](api/periodic_ewald.md)
- [Devices](api/devices.md)
- [Extensions](api/extensions.md)
- [Internals](api/internals.md)

## Topic Guide

### Types

Core public structs including kernels, contours, domains, problems, and steppers.

Open: [Types](api/types.md)

### Velocity & Acceleration

Pointwise and batched velocity APIs, plus the current CPU and GPU velocity support.

Open: [Velocity & Acceleration](api/velocity.md)

### Time Integration

Timestep and evolution entry points.

Open: [Time Integration](api/time_integration.md)

### Surgery

Remeshing, reconnection, and filament-removal functions.

Open: [Surgery](api/surgery.md)

### Diagnostics

Energy, circulation, enstrophy, geometry, and related contour diagnostics.

Open: [Diagnostics](api/diagnostics.md)

### Helpers

Contour utilities and shape constructors.

Open: [Helpers](api/helpers.md)

### Periodic & Ewald

Periodic-domain helpers and Ewald cache/setup routines.

Open: [Periodic & Ewald](api/periodic_ewald.md)

### Devices

CPU/GPU device types and array-constructor helpers.

Open: [Devices](api/devices.md)

### Extensions

Optional integrations for DifferentialEquations.jl, RecordedArrays.jl, JLD2.jl,
and Makie.

Open: [Extensions](api/extensions.md)

### Internals

Lower-level functions documented for advanced users and developers.

Open: [Internals](api/internals.md)
