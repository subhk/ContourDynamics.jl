# Theory & Method

This page describes the mathematical foundations of the contour dynamics method as implemented in ContourDynamics.jl.

## Contour Dynamics

### Vortex Patches and Green's Functions

Consider a 2D inviscid flow with piecewise-constant potential vorticity (PV). The streamfunction ``\psi`` satisfies:

```math
\mathcal{L}\psi = q(\mathbf{x})
```

where ``\mathcal{L}`` is the PV inversion operator (``\nabla^2`` for Euler, ``\nabla^2 - L_d^{-2}`` for QG). The solution is:

```math
\psi(\mathbf{x}) = \int\!\!\int G(|\mathbf{x} - \mathbf{x}'|) \, q(\mathbf{x}') \, dA'
```

For a single vortex patch with uniform PV jump ``q`` bounded by contour ``C``, the velocity ``\mathbf{u} = (-\psi_y, \psi_x)`` can be converted from an area integral to a **contour integral** via Green's theorem:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

This is the contour dynamics equation: the velocity at any point depends only on the **boundary** of the PV patch, not its interior.

### Segment Integration

The contour is discretized into piecewise-linear segments connecting nodes ``\{\mathbf{x}_j\}``. Each segment from ``\mathbf{a}`` to ``\mathbf{b}`` contributes:

```math
\mathbf{v}_{\text{seg}}(\mathbf{x}) = -\frac{1}{4\pi}(\mathbf{b}-\mathbf{a}) \int_0^1 \log|\mathbf{x} - \mathbf{a} - t(\mathbf{b}-\mathbf{a})|^2 \, dt
```

#### Euler Kernel

For the Euler kernel, this integral has a **closed-form antiderivative**. Projecting onto the segment's tangent and normal directions:

```math
F(u) = u\log(u^2 + h^2) - 2u + 2|h|\arctan(u/|h|)
```

where ``u`` is the tangential coordinate and ``h`` is the normal distance from ``\mathbf{x}`` to the segment line. The segment velocity is:

```math
\mathbf{v}_{\text{seg}} = -\frac{1}{4\pi}\hat{\mathbf{t}} \left[F(u_a) - F(u_b)\right]
```

This is **exact** — no quadrature error.

#### QG Kernel

For the QG kernel with ``G(r) = -\frac{1}{2\pi}K_0(r/L_d)``, we use **singular subtraction**:

```math
K_0(r/L_d) = -\log(r) + \underbrace{\left[K_0(r/L_d) + \log(r)\right]}_{\text{smooth at } r=0}
```

The logarithmic singularity is handled by the exact Euler antiderivative. The smooth remainder ``K_0(r/L_d) + \log(r) \to \log(2L_d) - \gamma`` as ``r \to 0`` is integrated with **5-point Gauss-Legendre quadrature**.

#### SQG Kernel

Surface quasi-geostrophic (SQG) dynamics replaces the Laplacian PV inversion with a **fractional Laplacian**:

```math
(-\nabla^2)^{1/2}\psi = \theta
```

where ``\theta`` is the surface buoyancy. The Green's function is ``G(r) = -1/(2\pi r)``, and the contour integral becomes:

```math
\mathbf{u}(\mathbf{x}) = -\frac{1}{2\pi}\oint_C \frac{d\mathbf{x}'}{|\mathbf{x}-\mathbf{x}'|}
```

The segment integral has a closed-form antiderivative:

```math
F(u) = \log\!\left(u + \sqrt{u^2 + h_{\text{eff}}^2}\right) = \operatorname{arcsinh}\!\left(\frac{u}{\sqrt{h_{\text{eff}}^2}}\right) + \text{const}
```

Unlike the Euler and QG kernels, the SQG velocity is **singular at the contour boundary** — the tangential component diverges logarithmically. A regularization ``\delta > 0`` is required, replacing ``1/r`` with ``1/\sqrt{r^2 + \delta^2}`` so that ``h_{\text{eff}}^2 = h^2 + \delta^2``. The segment velocity remains exact (no quadrature):

```math
\mathbf{v}_{\text{seg}} = -\frac{1}{2\pi}\hat{\mathbf{t}} \left[F(u_a) - F(u_b)\right]
```

## Ewald Summation

### Periodic Green's Functions

On a doubly-periodic domain ``[-L_x, L_x) \times [-L_y, L_y)``, the Green's function includes contributions from all periodic images. Direct summation converges slowly, so we use **Ewald splitting** to decompose:

```math
G_{\text{per}}(\mathbf{r}) = G_{\text{real}}(\mathbf{r}) + G_{\text{Fourier}}(\mathbf{r})
```

#### Real-Space Sum

```math
G_{\text{real}}(\mathbf{r}) = \frac{1}{4\pi} \sum_{\mathbf{n}} E_1(\alpha^2|\mathbf{r} - \mathbf{L}_\mathbf{n}|^2)
```

where ``E_1`` is the exponential integral, ``\alpha = \sqrt{\pi}/\min(L_x, L_y)`` is the splitting parameter, and the sum runs over periodic images ``\mathbf{L}_\mathbf{n} = (2nL_x, 2mL_y)``. The Gaussian damping ensures rapid convergence (typically 2 images suffice).

#### Fourier-Space Sum

```math
G_{\text{Fourier}}(\mathbf{r}) = \frac{1}{A} \sum_{\mathbf{k} \neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|^2} \cos(\mathbf{k} \cdot \mathbf{r})
```

where ``A = 4L_xL_y`` is the domain area. The Gaussian factor ``e^{-k^2/(4\alpha^2)}`` ensures rapid convergence in Fourier space.

#### Singular Subtraction for Periodic Velocity

The periodic segment velocity uses the same singular-subtraction approach: the log singularity from the central image is handled by the exact unbounded Euler formula, and the smooth correction ``G_{\text{per}} - G_\infty`` is integrated with 3-point Gauss-Legendre quadrature.

### QG Periodic Decomposition

For the QG kernel on a periodic domain, we decompose:

```math
G_{\text{QG,per}} = G_{\text{Euler,per}} + \underbrace{\frac{1}{A}\sum_{\mathbf{k}\neq 0} \frac{\kappa^2}{|\mathbf{k}|^2(|\mathbf{k}|^2 + \kappa^2)}\cos(\mathbf{k}\cdot\mathbf{r})}_{\text{smooth QG correction}}
```

where ``\kappa = 1/L_d``. The Euler periodic part uses the full Ewald machinery, and the QG correction is a smooth, rapidly convergent (``\sim 1/k^4``) Fourier series.

### SQG Periodic Decomposition

For the SQG kernel ``G(r) = -1/(2\pi r)`` on a periodic domain, the Ewald splitting decomposes the periodic sum of ``1/r`` into:

```math
\sum_{\mathbf{n}} \frac{1}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} = \sum_{\mathbf{n}} \frac{\operatorname{erfc}(\alpha|\mathbf{r} - \mathbf{L}_\mathbf{n}|)}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} + \frac{2\pi}{A}\sum_{\mathbf{k}\neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|}\cos(\mathbf{k}\cdot\mathbf{r})
```

The Fourier coefficients decay as ``1/|\mathbf{k}|`` (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature.

The periodic segment velocity uses singular subtraction: the regularized unbounded SQG velocity (exact ``\operatorname{arcsinh}`` antiderivative with ``\delta > 0``) handles the ``1/r`` singularity, and the smooth correction is integrated with 3-point Gauss-Legendre quadrature. The central-image correction ``\operatorname{erfc}(\alpha r)/r - 1/\sqrt{r^2+\delta^2}`` is finite at all quadrature points since the evaluation distance ``r > 0``.

## Contour Surgery

The Dritschel surgery algorithm (Dritschel, 1988) handles the topological changes that arise during long-time evolution.

### Node Redistribution (Remeshing)

After each surgery pass, nodes are redistributed along each contour to maintain segment lengths between ``\mu`` (minimum) and ``\Delta_{\max}`` (maximum):

1. Compute cumulative arc lengths along the contour
2. Walk the perimeter, placing new nodes at arc-length intervals
3. Short segments (``< \mu``) are merged; long segments (``> \Delta_{\max}``) are subdivided

### Reconnection

When two contour segments approach within distance ``\delta``:

- **Same contour**: the contour is **split** (pinched) into two daughter contours
- **Different contours with same PV**: the contours are **merged** (stitched together)

Reconnection uses a **spatial index** (hash-map binned by ``\delta``-sized grid) for ``O(N \log C)`` candidate detection, where ``N`` is the total node count and ``C`` is the number of contours.

### Filament Removal

After reconnection, contours with ``|A| < A_{\min}`` (where ``A`` is the signed area) are removed. Spanning contours (which encode the periodic domain topology) are always preserved.

### Surgery Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `delta` | ``\delta`` | Proximity threshold for detecting close segments |
| `mu` | ``\mu`` | Minimum segment length after remeshing |
| `Delta_max` | ``\Delta_{\max}`` | Maximum segment length after remeshing |
| `area_min` | ``A_{\min}`` | Minimum contour area; smaller contours are removed |
| `n_surgery` | — | Number of time steps between surgery passes |

Typical choices: ``\delta \approx \mu``, ``\Delta_{\max} \approx 10\text{–}40\mu``, ``A_{\min} \approx \delta^2``.

## Multi-Layer QG

### Modal Decomposition

For an ``N``-layer QG system, the PV in each layer is coupled to the streamfunction via the coupling matrix ``\mathbf{C}``:

```math
q_i = \sum_j C_{ij} \psi_j
```

The coupling matrix is diagonalized: ``\mathbf{C} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}``. Each eigenmode with eigenvalue ``\lambda_m`` evolves independently:

- If ``|\lambda_m| \approx 0``: **barotropic mode** — uses the Euler kernel
- Otherwise: ``L_d^{(\text{mode})} = 1/\sqrt{|\lambda_m|}`` — uses a QG kernel

The velocity in physical layers is recovered by projecting back through the eigenvector matrix.

## Time Integration

### RK4

The classical 4th-order Runge-Kutta scheme advances all node positions simultaneously:

```math
\mathbf{x}^{n+1} = \mathbf{x}^n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
```

This is the recommended integrator for most applications.

### Leapfrog with Robert-Asselin Filter

The leapfrog scheme is 2nd-order centred:

```math
\mathbf{x}^{n+1} = \mathbf{x}^{n-1} + 2\Delta t \, \mathbf{u}(\mathbf{x}^n)
```

A **Robert-Asselin filter** (``\nu = 0.05`` by default) damps the computational mode. The first step is bootstrapped with a 2nd-order midpoint (RK2) method.

## References

### Contour Dynamics and Surgery

- Dritschel, D. G. (1988). Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics. *J. Comput. Phys.*, 77(1), 240-266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D. G. (1989). Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows. *Comput. Phys. Rep.*, 10(3), 77-146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Dritschel, D. G. & Ambaum, M. H. P. (1997). A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields. *Q. J. R. Meteorol. Soc.*, 123(540), 1097-1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)

### Surface Quasi-Geostrophic Dynamics

- Held, I. M., Pierrehumbert, R. T., Garner, S. T. & Swanson, K. L. (1995). Surface quasi-geostrophic dynamics. *J. Fluid Mech.*, 282, 1-20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)
- Constantin, P., Majda, A. J. & Tabak, E. (1994). Formation of strong fronts in the 2-D quasigeostrophic thermal active scalar. *Nonlinearity*, 7(6), 1495-1533. [doi:10.1088/0951-7715/7/6/001](https://doi.org/10.1088/0951-7715/7/6/001)
- Scott, R. K. & Dritschel, D. G. (2014). Numerical simulation of a self-similar cascade of filament instabilities in the surface quasigeostrophic system. *Phys. Rev. Lett.*, 112, 144505. [doi:10.1103/PhysRevLett.112.144505](https://doi.org/10.1103/PhysRevLett.112.144505)
- Rodrigo, J. L. (2005). On the evolution of sharp fronts for the quasi-geostrophic equation. *Comm. Pure Appl. Math.*, 58(6), 821-866. [doi:10.1002/cpa.20059](https://doi.org/10.1002/cpa.20059)
