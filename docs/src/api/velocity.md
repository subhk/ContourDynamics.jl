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
- GPU: direct KernelAbstractions evaluation for single-layer Euler, QG, and SQG on unbounded or periodic domains, plus direct multi-layer QG on unbounded or periodic domains

The production velocity path is intentionally direct at the moment. This keeps
the contour surgery and curved-segment geometry coupled to one reference
implementation instead of maintaining a separate approximate accelerator.
Curved-segment geometry is used for the single-layer Euler, QG, and SQG kernels
on both unbounded and periodic domains; exactly straight segments still use the
analytic straight-segment formulas.
