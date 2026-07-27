# Examples

This section collects short, runnable examples. Each page isolates one workflow
or one physical effect in a compact setup.

Suggested reading order:

1. [Vortex Merger](examples/vortex_merger.md)
2. [Filamentation](examples/filamentation.md)
3. [Beta-Plane Vortex Drift](examples/beta_plane_vortex_drift.md)
4. [SQG Elliptical Vortex](examples/sqg_elliptical_vortex.md)
5. [Two-Layer QG](examples/two_layer_qg.md)

Longer runnable scripts are available in the [`examples/`](https://github.com/subhk/ContourDynamics.jl/tree/main/examples) directory. Those scripts save JLD2 snapshots, final PNG/SVG figures, and MP4 animations under `examples/output/`. The pages in this section provide shorter versions of the same setups, with smaller contour discretizations so the docs build stays fast.

::: tip GPU Acceleration
Every example here can run on GPU: add `using CUDA` and pass `dev=GPU()` to
`Problem`. Single-layer Euler, QG, SQG, and beta-plane QG, as well as
multi-layer QG, all evaluate velocity, timestep, and run surgery against
device-resident contour state.
:::

Each example page includes:

- a short statement of the physical setup
- a runnable code block
- expected qualitative behavior
- references to the classical literature where relevant

The repository currently includes full scripts for:

- `vortex_merger.jl`
- `filamentation.jl`
- `beta_drift.jl`
- `sqg_elliptical_vortex.jl`
- `two_layer_qg.jl`

Run a full example from the repository root with:

```sh
julia --project=. examples/vortex_merger.jl
```

The `Examples` section in the left-hand navigation provides direct access to
each case.
