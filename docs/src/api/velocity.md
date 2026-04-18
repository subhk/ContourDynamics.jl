# API Reference: Velocity & Acceleration

## Velocity Computation

```@docs
velocity!
velocity
segment_velocity
```

## Accelerator Status

Current large-problem behavior is:

- single-layer unbounded CPU: direct for small problems, treecode for large problems
- single-layer unbounded CPU with experimental proxy FMM enabled: optional experimental FMM path
- single-layer periodic CPU: treecode for large problems in normal use; the current periodic `_fmm_velocity!` method is still a direct fallback
- multi-layer CPU: treecode for large problems in normal use; the current multi-layer `_fmm_velocity!` method is still a direct fallback
- GPU: supported for single-layer Euler, QG, and SQG on unbounded or periodic domains, plus direct multi-layer QG on unbounded or periodic domains; large GPU-tagged problems use a hybrid path today, with treecode/FMM dispatch still CPU-led and the per-leaf direct and linearized treecode worklists able to reuse KA in batched leaf-sized evaluations

So, at the moment, the production accelerator story is treecode plus the direct
GPU unbounded-kernel path. The proxy FMM remains experimental and is not the active
production path for periodic or multi-layer problems.
