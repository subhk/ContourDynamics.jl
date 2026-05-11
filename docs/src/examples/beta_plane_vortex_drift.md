# Beta-Plane Vortex Drift

This example uses contour dynamics on a beta plane. The live contours encode
full PV: a straight beta staircase plus a circular vortex anomaly. The
[`BetaPlaneQGKernel`](@ref) keeps a frozen copy of the initial straight
staircase and subtracts it from the QG inversion, so the velocity is induced by

```math
q_\mathrm{regular}
  = q_\mathrm{full\ contours}
  - q_\mathrm{reference\ staircase}.
```

That reference straight beta staircase subtraction is the important beta-plane correction.
The beta contours remain material and can deform, but the undeformed planetary
PV gradient is not inverted as active anomaly PV.

The parameters follow vortex D from Lam & Dritschel (2001): their equation
(2.5) sets ``\beta = 1`` and ``R_d = 1``, section 3.1 uses domain half-width
``l = 5\pi``, and Table 1 gives vortex radius ``R = 1``, PV anomaly
``\omega_0 = 5``, and ``n_\beta = 50`` beta contours. The paper used CASL with
Table 1 parameters ``\bar n_h = 512``, ``m_g = 2``, and
``\delta = 10^{-3}``, and ran to ``t = 28``. This package example is direct
contour dynamics with reference-staircase subtraction, not the full CASL
algorithm.

Run the default demonstration with:

```bash
julia -t 5 examples/beta_drift.jl
```

To use the paper time window and the paper beta-contour count:

```bash
BETA_DRIFT_PRESET=paper julia -t 5 examples/beta_drift.jl
```

Useful controls:

- `BETA_DRIFT_NBETA` sets the number of beta-staircase contours
- `BETA_DRIFT_BETA_NODES` sets nodes per spanning beta contour
- `BETA_DRIFT_VORTEX_NODES` sets the vortex contour resolution
- `BETA_DRIFT_DT`, `BETA_DRIFT_T_FINAL`, and `BETA_DRIFT_NSTEPS` control time
- `BETA_DRIFT_SAVE_DT` controls snapshot and animation cadence
- `BETA_DRIFT_DRY_RUN=true` checks parameters without evolving or writing media

What to look for:

- a straight reference beta staircase alone induces no velocity
- the vortex deforms the material beta contours
- the largest closed contour reports the vortex centroid displacement over time

**Reference:**
- Lam, J.S.-L. & Dritschel, D.G. (2001). *On the beta-drift of an initially circular vortex patch.* J. Fluid Mech. **436**, 107--129. [doi:10.1017/S0022112001003974](https://doi.org/10.1017/S0022112001003974)
