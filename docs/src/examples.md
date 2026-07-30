# Examples

This section collects short, runnable adaptations of published benchmark cases.
Every physical example has a specific literature source; package-specific
discretization choices are identified separately and are not presented as values
from the source paper.

| Script | Literature case | Package adaptation |
|:--|:--|:--|
| `vortex_merger.jl` | [Dritschel (1988), Figure 13](https://doi.org/10.1016/0021-9991(88)90165-9) | Shorter run and package remeshing/surgery |
| `filamentation.jl` | [Dritschel (1988), equation (13), Figure 5, Table IV case 1](https://doi.org/10.1016/0021-9991(88)90165-9) | Package remeshing parameters and diagnostics |
| `beta_drift.jl` | [Lam & Dritschel (2001), Table 1 case D and Figure 5](https://doi.org/10.1017/S0022112001003974) | Direct contour dynamics; reduced-resolution `demo` preset |
| `sqg_elliptical_vortex.jl` | [Held et al. (1995), equation (19) and Figure 2](https://doi.org/10.1017/S0022112095000012) | Smooth field represented by contour levels; unbounded default |
| `two_layer_qg.jl` | [Polvani, Zabusky & Flierl (1989), Figure 19](https://doi.org/10.1017/S0022112089002016) | Symmetric modal transform, automatic surgery, and shorter run |

`visualization.jl` and `visualization_geometry.jl` are shared output helpers,
not standalone physical examples.

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
- the exact equation, table, or figure that defines the physical case
- an explicit note about package-specific numerical adaptations

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
