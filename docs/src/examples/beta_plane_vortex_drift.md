# Beta-Plane Vortex Drift

This example uses contour dynamics on a beta plane. The live contours encode
full PV: a straight beta staircase plus a circular vortex anomaly. The
[`BetaPlaneQGKernel`](@ref) keeps a frozen copy of the initial straight
staircase, subtracts it from the contour sum, and adds the analytic velocity of
the finite-step residual relative to the continuous beta plane:

```math
q_\mathrm{regular}
  = q_\mathrm{full\ contours}
  - q_\mathrm{reference\ staircase}
  + (q_\mathrm{reference\ staircase} - \beta y).
```

That is the contour-only form of the Lam-Dritschel decomposed inversion
``(\nabla^2 - R_d^{-2})\psi_r = q_r - \beta y``. The beta contours remain
material and can deform; no contour-to-grid velocity solve is introduced.

The physical initial condition is equation (2.4) and case D in Table 1 of Lam &
Dritschel (2001), plotted in their Figure 5. Their equation (2.5) sets
``\beta=1`` and ``R_d=1``; case D has ``R=1`` and ``\omega_0=5``; and section
3.1 uses domain half-width ``l=5\pi``. The paper uses ``n_\beta=50``, a CASL
grid with ``\bar n_h=512`` and ``m_g=2``, surgery scale
``\delta=10^{-3}``, and a time window ending at ``t=28``.

The default `demo` preset preserves the literature model and vortex parameters
but reduces contour resolution and duration. The `paper` preset restores
``n_\beta=50`` and ``t=28``. Neither preset is an exact reproduction because
this package evaluates velocity directly on contours instead of using the
paper's contour-advective semi-Lagrangian grid inversion.

Run the default demonstration with:

```bash
julia -t 5 examples/beta_drift.jl
```

To use the paper time window and the paper beta-contour count, set
`preset = "paper"` near the top of `examples/beta_drift.jl`, then run:

```bash
julia -t 5 examples/beta_drift.jl
```

Useful in-file controls:

- `n_beta` sets the number of beta-staircase contours
- `nodes_per_beta_contour` sets nodes per spanning beta contour
- `vortex_nodes` sets the vortex contour resolution
- `dt`, `t_final`, and `nsteps` control time
- `save_dt` controls snapshot and animation cadence
- `dry_run = true` checks parameters without evolving or writing media

What to look for:

- a straight reference beta staircase induces only the small finite-step
  `q_r - beta*y` sawtooth velocity
- the vortex deforms the material beta contours
- the largest closed contour reports the vortex centroid displacement over time

**Reference:**
- Lam, J.S.-L. & Dritschel, D.G. (2001). *On the beta-drift of an initially circular vortex patch.* J. Fluid Mech. **436**, 107--129. [doi:10.1017/S0022112001003974](https://doi.org/10.1017/S0022112001003974)
