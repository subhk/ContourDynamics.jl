# API Reference: Velocity & Acceleration

## Velocity Computation

```@docs
velocity!
velocity
segment_velocity
```

## Accelerator Status

Current large-problem behavior is:

- single-layer CPU: direct reference evaluation
- periodic single-layer CPU: direct reference evaluation with Ewald corrections
- multi-layer CPU: direct modal decomposition
- beta-plane QG: direct CPU evaluation with analytic beta-plane correction
- GPU velocity: direct KernelAbstractions evaluation for single-layer Euler, QG, and SQG on unbounded or periodic domains; multi-layer QG velocity through modal single-layer KA calls

The production velocity path is intentionally direct at the moment. This keeps
the contour surgery and curved-segment geometry coupled to one reference
implementation instead of maintaining a separate approximate accelerator.
Curved-segment geometry is used for the single-layer Euler, QG, and SQG kernels
on both unbounded and periodic domains; exactly straight segments still use the
analytic straight-segment formulas.

Multi-layer `GPU()` support is currently velocity-only: timestepping, periodic
wrapping, and surgery for multi-layer GPU problems throw instead of silently
falling back to CPU work. `BetaPlaneQGKernel` currently requires `dev=CPU()`.
