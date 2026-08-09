# Beta-Plane Vortex Drift

This example uses contour dynamics on a beta plane. The live contours encode
full PV: a straight beta staircase plus a circular vortex anomaly. The
[`BetaPlaneQGKernel`](@ref) keeps a frozen copy of the initial straight
staircase, subtracts it from the contour sum, and adds the analytic velocity of
the finite-step residual relative to the continuous beta plane. In the notation
of Lam & Dritschel (2001), the source passed to the Helmholtz inversion is

```math
(\nabla^2-R_d^{-2})\psi
  = (q_s + q_r - q_\mathrm{ref})
  + (q_\mathrm{ref} - \beta y)
  = q_s + q_r - \beta y.
```

Here ``q_s`` is the singular vortex-patch PV, ``q_r`` is the material regular
PV represented by the evolving beta contours, and ``q_\mathrm{ref}`` is the PV
of the frozen initial straight staircase. Thus the right-hand side is the total
relative PV ``q-\beta y``, where ``q=q_s+q_r``.

This is the contour-only sum of the paper's two inversions:
``(\nabla^2-R_d^{-2})\psi_s=q_s`` and
``(\nabla^2-R_d^{-2})\psi_r=q_r-\beta y`` (or `q_r - beta*y` in ASCII
notation). The beta contours remain material and can deform; no contour-to-grid
velocity solve is introduced. Here ``\nabla^2`` is the horizontal Laplacian and
``R_d`` is the deformation radius (called `Ld` by the package).

The physical initial condition is equation (2.4) and case D in Table 1 of Lam &
Dritschel (2001), plotted in their Figure 5. Their equation (2.5) sets
``\beta=1`` and ``R_d=1``; case D has ``R=1`` and ``\omega_0=5``; and section
3.1 uses domain half-width ``l=5\pi``. The paper uses ``n_\beta=50``, a CASL
grid with ``\bar n_h=512`` and ``m_g=2``, surgery scale
``\delta=10^{-3}``, and a time window ending at ``t=28``.

The example uses only the paper's case-D physical parameters: ``n_\beta=50``,
``t=28``, the Figure 5 output times ``t=7,14,21,28``, and the Figure 5 view
``[-l,l] \times [-0.25l,0.75l]``. It checks the contour levels and PV jumps
against equations (3.2)--(3.4) before evolving.

This is a reproduction of the paper's mathematical initial-value problem and
case-D parameters, not its numerical method. The paper uses a 512-by-512 CASL
grid, a twice-finer contour-to-grid conversion, and CASL surgery at
``\delta=10^{-3}``. This example instead evaluates velocity by direct contour
dynamics with the analytic beta-plane correction described above. CASL surgery
is therefore reported as paper metadata but is not silently replaced by
incomparable direct-CD surgery settings. A quantitative reproduction requires a
separate direct-CD convergence study.

Run the example with:

```bash
julia -t 5 examples/beta_drift.jl
```

The direct calculation is expensive. To validate the complete paper setup
without evolving or loading the visualization dependencies, run:

```bash
BETA_DRIFT_DRY_RUN=true julia examples/beta_drift.jl
```

Useful in-file controls:

- `n_beta` sets the number of beta-staircase contours
- `nodes_per_beta_contour` sets nodes per spanning beta contour
- `vortex_nodes` sets the vortex contour resolution
- `dt`, `t_final`, and `nsteps` control time
- `save_dt` controls snapshot and animation cadence
- `BETA_DRIFT_DRY_RUN=true` checks parameters without evolving or writing media

What to look for:

- a straight reference beta staircase induces only the small finite-step
  `q_r - beta*y` sawtooth velocity
- the vortex deforms the material beta contours
- the largest closed contour reports the vortex centroid displacement over time

**Reference:**
- Lam, J.S.-L. & Dritschel, D.G. (2001). *On the beta-drift of an initially circular vortex patch.* J. Fluid Mech. **436**, 107--129. [doi:10.1017/S0022112001003974](https://doi.org/10.1017/S0022112001003974)
