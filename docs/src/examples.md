# Examples

This section collects short, runnable examples. Each one focuses on one idea
and keeps the setup small enough to read in one pass.

If you are new to the package, read these in roughly this order:

1. [Vortex Merger](examples/vortex_merger.md)
2. [Filamentation](examples/filamentation.md)
3. [Beta-Plane Vortex Drift](examples/beta_plane_vortex_drift.md)
4. [SQG Elliptical Vortex](examples/sqg_elliptical_vortex.md)
5. [Two-Layer QG](examples/two_layer_qg.md)

Longer runnable scripts are available in the [`examples/`](https://github.com/subhk/ContourDynamics.jl/tree/main/examples) directory. The pages in this section are shorter walkthrough versions of those ideas.

::: tip GPU Acceleration
All examples in this section can use GPU velocity evaluation on supported
hardware. Add `using CUDA` and pass `dev=GPU()` to `Problem`.
GPU velocity is currently available for single-layer Euler, QG, and SQG on unbounded or periodic domains, plus direct multi-layer QG on unbounded or periodic domains.
:::

Each example page includes:

- a short description of what the example shows
- a small runnable code block
- a few cues for what to look for in the output
- references to the classical literature where relevant

The repository currently includes full scripts for:

- `vortex_merger.jl`
- `filamentation.jl`
- `beta_drift.jl`
- `sqg_elliptical_vortex.jl`
- `two_layer_qg.jl`

Use the `Examples` section in the left-hand navigation to jump directly to a
specific case.
