# API Reference: Velocity & Acceleration

## Velocity Computation

```@docs
velocity!
velocity
segment_velocity
```

## Accelerator Status

Current large-problem behavior is:

- single-layer CPU: direct for small problems, proxy FMM for large problems
- single-layer CPU with `_FMM_ACCELERATION_ENABLED = false`: direct for small problems, treecode for large problems
- periodic single-layer CPU: the same large-problem proxy-FMM policy as the unbounded case
- multi-layer CPU: the same large-problem proxy-FMM policy as the single-layer case
- GPU: supported for single-layer Euler, QG, and SQG on unbounded or periodic domains, plus direct multi-layer QG on unbounded or periodic domains; large GPU-tagged problems use a hybrid path today, with treecode/FMM dispatch still CPU-led and the per-leaf direct and linearized treecode worklists able to reuse KA in batched leaf-sized evaluations

So, at the moment, the production accelerator story is proxy FMM for large CPU
problems, plus the direct GPU paths and GPU-assisted treecode leaf evaluations.
