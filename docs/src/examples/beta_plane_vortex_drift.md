# Beta-Plane Vortex Drift

A cyclonic vortex on a beta plane drifts north-westward due to the background
PV gradient. The gradient ``\beta y`` is represented by a PV staircase in a
periodic domain.

The parameters below follow vortex D from Lam & Dritschel (2001): their
equation (2.5) sets ``\beta = 1`` and ``R_d = 1``, section 3.1 uses domain
half-width ``l = 5\pi``, and Table 1 gives vortex radius ``R = 1``, PV anomaly
``\omega_0 = 5``, and ``n_\beta = 50`` β-contours. The β-contour positions
follow their equations (3.2)--(3.4). The original calculation used CASL with
common Table 1 parameters ``\bar n_h = 512``, ``m_g = 2``, and
``\delta = 10^{-3}``, and ran to ``t = 28``. Those grid parameters are not
direct-contour node counts; the example script reports them as CASL metadata
and keeps the direct contour node counts separately configurable.

To run the full paper time window with the literature parameters that map
directly to this solver, use:

```bash
BETA_DRIFT_PRESET=paper julia -t 5 examples/beta_drift.jl
```

For direct-contour convergence checks, vary ``BETA_DRIFT_BETA_NODES`` and
``BETA_DRIFT_VORTEX_NODES``. Do not interpret ``\bar n_h = 512`` as 512 nodes
per β-contour.

What to look for:

- the vortex is embedded in a set of spanning contours
- the vortex center moves over time
- the reported drift should be north-westward

```@repl example_beta_plane
using ContourDynamics
using StaticArrays

beta = 1.0            # planetary vorticity gradient
Ld = 1.0              # deformation radius Rd
R = 1.0               # vortex radius
omega0 = 5.0          # uniform positive PV anomaly
L = 5π                # domain half-width
n_beta = 50

# Rounded Table 1 resolution ratios for vortex D
@assert isapprox(L / (n_beta * R), 0.31; atol=0.005)
@assert isapprox(L / (n_beta * abs(omega0)), 6.3e-2; atol=5e-4)

# PV staircase discretizing βy, Lam & Dritschel (2001), Eqs. (3.2)--(3.4)
function lam_dritschel_beta_staircase(beta, L, n_beta; nodes_per_contour)
    d_beta = 2L / n_beta
    dq = beta * d_beta
    wrap = SVector(2L, 0.0)
    return [begin
        y = (k - 0.5) * d_beta - L
        nodes = [SVector(-L + 2L * (i - 1) / nodes_per_contour, y)
                 for i in 1:nodes_per_contour]
        PVContour(nodes, dq, wrap)
    end for k in 1:n_beta]
end

staircase = lam_dritschel_beta_staircase(beta, L, n_beta; nodes_per_contour=3)

# Cyclonic vortex at origin
vortex = circular_patch(R, 24, omega0)

prob = Problem(;
    contours = vcat(staircase, [vortex]),
    dt       = 0.005,
    kernel   = :qg,
    Ld       = Ld,
    domain   = :periodic,
    Lx       = L,
    Ly       = L,
)

c0 = centroid(vortex)
evolve!(prob; nsteps=1)

# Find vortex (largest non-spanning contour)
final_contours = materialize_contours(prob)
vortex_out = argmax(c -> is_spanning(c) ? 0.0 : abs(vortex_area(c)), final_contours)
cf = centroid(vortex_out)
println("Vortex drift: dx=$(round(cf[1]-c0[1]; digits=4)), dy=$(round(cf[2]-c0[2]; digits=4))")
println("Largest non-spanning area: $(round(abs(vortex_area(vortex_out)); digits=6))");
```

**References:**
- Dritschel, D.G. (1988). *Contour surgery.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Lam, J.S.-L. & Dritschel, D.G. (2001). *On the beta-drift of an initially circular vortex patch.* J. Fluid Mech. **436**, 107--129. [doi:10.1017/S0022112001003974](https://doi.org/10.1017/S0022112001003974)
