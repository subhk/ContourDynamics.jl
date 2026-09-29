# Ewald Summation

## Periodic Green's Functions

On a doubly-periodic domain ``[-L_x, L_x) \times [-L_y, L_y)``, the contour
kernel includes contributions from all periodic images. Ewald splitting
separates the singular short-range and smooth long-range contributions. For
Euler, the implementation uses the **mean-free** contour kernel:

```math
G_{\text{Euler,per}}(\mathbf{r}) = G_{\text{real}}(\mathbf{r}) + G_{\text{Fourier}}(\mathbf{r}) - \frac{1}{4\alpha^2 A}.
```

The two sums below are the Euler decomposition. Their real-space part has
cell average ``1/(4\alpha^2 A)``; the displayed subtraction removes it, matching
`_periodic_euler_zero_mode_scalar`. This constant does not affect velocity from
a closed contour because ``\oint_C d\mathbf x'=0``. All kernels on this page
use the contour-integral sign convention, ``G=-G_\psi``.

The basic problem is this:

- in a periodic domain, each contour interacts not only with the copy in the
  main domain, but also with infinitely many translated copies
- summing those copies directly is too slow and converges poorly
- Ewald summation rewrites the same periodic Green's function as two rapidly
  convergent pieces

In this page:

- ``\mathbf{r}`` is the displacement from the source point to the target point
- ``|\mathbf{r}|`` is its Euclidean length
- ``G_{\text{per}}`` is the periodic Green's function
- ``G_{\text{real}}`` is the short-range part, summed over nearby image copies
- ``G_{\text{Fourier}}`` is the smooth long-range part, summed in Fourier space
- ``L_x`` and ``L_y`` are the half-widths of the periodic domain
- ``A = 4L_xL_y`` is the full domain area

The reason this helps is that the singular, short-range part is easy to handle
in physical space, while the smooth long-range part is easy to handle in
Fourier space.

### Real-Space Sum

```math
G_{\text{real}}(\mathbf{r}) = \frac{1}{4\pi} \sum_{\mathbf{n}} E_1(\alpha^2|\mathbf{r} - \mathbf{L}_\mathbf{n}|^2)
```

Here:

- ``E_1(z)=\int_z^\infty e^{-t}/t\,dt`` is the exponential integral
- ``\alpha = \sqrt{\pi}/\sqrt{L_xL_y}`` is the splitting parameter used by the implementation
- ``\mathbf{n}=(n,m)\in\mathbb{Z}^2`` is a two-dimensional image index
- ``\mathbf{L}_\mathbf{n} = (2nL_x, 2mL_y)`` is the corresponding lattice shift
- ``\sum_{\mathbf n}`` is the image sum, truncated in code by `n_images`

This real-space sum contains the short-range part of the interaction. Because of
the Gaussian damping introduced by Ewald splitting, contributions from distant
images decay quickly, so only a small number of nearby images are needed in
practice.

### Fourier-Space Sum

```math
G_{\text{Fourier}}(\mathbf{r}) = \frac{1}{A} \sum_{\mathbf{k} \neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|^2} \cos(\mathbf{k} \cdot \mathbf{r})
```

Here:

- ``\mathbf{k}`` is a Fourier wavevector on the periodic domain
- specifically, ``\mathbf{k}=(\pi p/L_x,\pi s/L_y)`` for integer mode indices ``p`` and ``s``
- ``\mathbf{k}\cdot\mathbf{r}`` is the usual Fourier phase
- ``|\mathbf{k}|^2=k_x^2+k_y^2`` is the squared wavenumber
- the term ``\mathbf{k} \neq 0`` excludes the zero mode
- the Gaussian factor ``e^{-|\mathbf{k}|^2/(4\alpha^2)}`` makes the Fourier sum converge rapidly
- the sum is truncated in code by `n_fourier`

This Fourier-space sum represents the smooth long-range part of the periodic
interaction. It is the part that would be awkward to compute accurately by
adding many distant image copies directly.

Every Fourier coefficient used by the package depends on ``|\mathbf{k}|`` only,
so it is even in ``k_x`` and in ``k_y``. The ``\sin(k_xr_x)\sin(k_yr_y)`` halves
of ``\cos(\mathbf{k}\cdot\mathbf{r})`` then cancel between the modes
``(\pm p,\pm s)``, and the code evaluates each such sum as

```math
\sum_{p,s\ge 0} w_{ps}\cos(p\theta_x)\cos(s\theta_y),
\qquad \theta_x=\pi r_x/L_x,\quad \theta_y=\pi r_y/L_y,
```

where ``w_{ps}`` collects the coefficients of the modes ``(\pm p,\pm s)``. The
cosines of multiples of ``\theta`` follow from the Chebyshev recurrence
``\cos((p+1)\theta)=2\cos\theta\cos(p\theta)-\cos((p-1)\theta)``, so an
evaluation needs two cosines rather than two trigonometric calls per mode.
`EwaldCache` derives these folded tables from its coefficient tables and
rejects tables without this symmetry.

### Singular Subtraction for Periodic Velocity

Every periodic segment velocity is built by singular subtraction: a
singularity-handling **base velocity** plus a **smooth correction** integrated
by Gauss–Legendre quadrature. In the code this split is explicit — the base
term is `_periodic_base_velocity` and the per-quadrature-point correction is
`_periodic_green_correction`, dispatched on the kernel:

- **Euler and SQG**: the base is the *unbounded* segment contribution, and the
  correction is ``G_{\text{per}} - G_\infty``, where ``G_\infty`` is the
  corresponding unbounded-space Green's function.
- **QG**: the base is the *periodic Euler* (Ewald) velocity itself, and the
  correction is the smooth QG–Euler difference described in the next section,
  itself Ewald split (its Fourier coefficients are precomputed in
  `EwaldCache.corr_coeffs`). For deformation radii short compared with the
  domain, the unbounded QG kernel is instead summed directly over the few
  periodic images that matter.

The unbounded Euler and regularized SQG contributions are analytic for straight
segments. For cubic arcs, the base contribution also uses quadrature, as
described in [Contour Surgery](contour_surgery.md#Curved-Segment-Velocity).

Either way, only a smooth function is left for numerical quadrature. This is
important because quadrature is most reliable on smooth integrands, not on
functions with logarithmic or stronger singular behavior.

## QG Periodic Decomposition

For the QG kernel on a periodic domain, we decompose:

```math
G_{\text{QG,per}} = G_{\text{Euler,per}} + \frac{1}{A\kappa^2} - \underbrace{\frac{1}{A}\sum_{\mathbf{k}\neq 0} \frac{\kappa^2}{|\mathbf{k}|^2(|\mathbf{k}|^2 + \kappa^2)}\cos(\mathbf{k}\cdot\mathbf{r})}_{\text{smooth QG correction}}
```

Here ``G_{\text{QG,per}}`` and ``G_{\text{Euler,per}}`` are the periodic QG
and Euler Green's functions, ``\kappa=1/L_d`` is inverse deformation radius,
and ``A``, ``\mathbf{k}``, and ``\mathbf{r}`` retain their definitions above.
The constant ``1/(A\kappa^2)`` is the QG zero Fourier mode. The velocity
implementation omits this constant: it cancels around closed contours and
between each live beta-staircase contour and its matching reference contour.
The full periodic QG energy retains the corresponding zero-mode contribution,
as described in [Diagnostics](../api/diagnostics.md).
The key idea is that the QG periodic kernel can be written as:

- an Euler-like periodic part, which already has a validated Ewald treatment
- a smooth correction ``\kappa^2\hat G``, with
  ``\hat G(\mathbf r) = -A^{-1}\sum_{\mathbf k\neq 0}\cos(\mathbf k\cdot\mathbf r)/(|\mathbf k|^2(|\mathbf k|^2+\kappa^2))``

The correction's coefficients decay only like ``|\mathbf{k}|^{-2}`` until
``|\mathbf k|`` exceeds ``\kappa``, so a truncated series converges slowly once
``L_d`` is comparable to the Fourier cutoff scale. The correction is therefore
Ewald split as well. Writing
``1/(k^2(k^2+\kappa^2)) = \int_0^\infty s\,\varphi_1(\kappa^2 s)\,e^{-k^2 s}\,ds``
with ``\varphi_1(x) = (1-e^{-x})/x`` and splitting at ``s_0 = 1/(4\alpha^2)`` gives

```math
\hat G(\mathbf r) = \frac{s_0}{4\pi}\sum_{\mathbf n} P\!\left(\frac{|\mathbf r-\mathbf L_{\mathbf n}|^2}{4 s_0}, x\right)
- \frac{1}{A}\sum_{\mathbf k\neq 0}\frac{e^{-k^2 s_0}\,(1+k^2 s_0\varphi_1(x))}{k^2(k^2+\kappa^2)}\cos(\mathbf k\cdot\mathbf r)
+ \frac{s_0^2\,\psi_2(x)}{A},
```

where ``x=\kappa^2 s_0``, ``\psi_2(x)=(x-1+e^{-x})/x^2``, and
``P(u,x)=\sum_{n\ge 1}(-1)^n x^{n-1}E_{n+1}(u)/n!`` in terms of generalized
exponential integrals. Both sums now decay like Gaussians, so the Euler
truncation (`n_fourier`, `n_images`) also converges the correction. The series
for ``P`` is used while ``x\le 4``; beyond that, ``\kappa`` is so large that
``K_0`` decays within a few periods, and the velocity sums the unbounded QG
segment contribution over images directly (keeping the zero-mean convention by
removing ``1/(A\kappa^2)``).

## SQG Periodic Decomposition

For the unregularized SQG kernel ``G(r) = 1/(2\pi r)``, the periodic
``1/r`` sum has the following Ewald representation, modulo a spatial constant:

```math
\sum_{\mathbf{n}} \frac{1}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} = \sum_{\mathbf{n}} \frac{\operatorname{erfc}(\alpha|\mathbf{r} - \mathbf{L}_\mathbf{n}|)}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} + \frac{2\pi}{A}\sum_{\mathbf{k}\neq 0} \frac{\operatorname{erfc}(|\mathbf{k}|/(2\alpha))}{|\mathbf{k}|}\cos(\mathbf{k}\cdot\mathbf{r})
```

The image index ``\mathbf n``, lattice shift ``\mathbf L_{\mathbf n}``,
wavevector ``\mathbf k``, splitting parameter ``\alpha``, and area ``A`` are
defined above. The complementary error function is
``\operatorname{erfc}(z)=1-\operatorname{erf}(z)``. Both sums omit terms only
through the configured finite `n_images` and `n_fourier` truncations; the
displayed equation is the infinite-truncation form.

The two-dimensional lattice sum of ``1/r`` and the ``k=0`` inverse of the
fractional Laplacian are defined only up to a spatial constant. Accordingly,
the displayed identity is understood modulo that constant. Closed-contour
velocity is insensitive to it. The real- and Fourier-space terms, together
with the mean-free convention, are given by
[Holzmann & Bernu (2005), Eqs. (6)–(7)](https://doi.org/10.1016/j.jcp.2004.11.037).
Their mean-free ``1/r`` potential subtracts ``2\sqrt{\pi}/(\alpha A)`` from
the two sums above; multiplying by ``1/(2\pi)`` gives the SQG contour-kernel
normalization.

The package applies the analogous split to the softened kernel
``1/r_\delta`` itself, the potential of a unit charge at height ``\delta``
above the plane (the quasi-2-D Ewald sum): real-space terms
``\operatorname{erfc}(\alpha r_\delta)/r_\delta`` and Fourier coefficients

```math
\frac{\pi}{A|\mathbf k|}\left[e^{|\mathbf k|\delta}\operatorname{erfc}\!\left(\frac{|\mathbf k|}{2\alpha}+\alpha\delta\right)
+ e^{-|\mathbf k|\delta}\operatorname{erfc}\!\left(\frac{|\mathbf k|}{2\alpha}-\alpha\delta\right)\right],
```

both of which decay like Gaussians, so the softening needs no separate image
sum. The mean-free convention removes the constant

```math
C_0=\frac{1}{A}\left(\frac{e^{-\alpha^2\delta^2}}{\alpha\sqrt{\pi}}-\delta\,\operatorname{erfc}(\alpha\delta)\right)
```

(in the ``1/(2\pi)`` normalization) from the velocity kernel. Closed contours
are insensitive to it, but spanning contours with ``\sum \mathrm{pv}\cdot\mathrm{wrap}\neq 0``
would otherwise drift with a uniform velocity that depends on ``\alpha``.

The Fourier coefficients contain an ``\operatorname{erfc}(|\mathbf{k}|/(2\alpha))`` damping factor and a leading ``1/|\mathbf{k}|`` behavior (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature. In practical terms, this means SQG is less smooth than Euler in Fourier space and therefore needs a bit more care numerically.

The periodic segment velocity again uses singular subtraction:

- the regularized unbounded SQG contribution handles the near-singular part,
  analytically for straight segments and by quadrature for cubic arcs
- the periodic correction is smooth enough to integrate with 5-point Gauss-Legendre quadrature

Regularization is applied to every periodic image. For the central image, the
regularized unbounded contribution is supplied by the base velocity and the Ewald
correction is ``-\operatorname{erf}(\alpha r_\delta)/r_\delta``, which remains
bounded at coincidence. Each non-central real-space image adds
``\operatorname{erfc}(\alpha r_\delta)/r_\delta`` with
``r_\delta=\sqrt{r^2+\delta^2}``.

Thus the combined real-space and Fourier sums represent the periodic softened
kernel ``1/r_\delta`` with the configured truncations. Its nonzero Fourier modes
carry the softening factor ``e^{-\delta|\mathbf k|}`` derived in
[Contour Dynamics](contour_dynamics.md#SQG-Kernel); finite ``\delta`` changes
the inversion from the unregularized SQG model.

## Periodic Energy Potentials

The energy diagnostic integrates a contour potential ``\Phi`` with
``\nabla^2\Phi`` proportional to the Green's function. Its Fourier
coefficients decay two powers of ``|\mathbf k|`` faster than the Green's
function's, but a plain truncated series still misses the high-``k`` energy of
contours smaller than the cutoff scale. The periodic potentials are therefore
Ewald split too, and are zero-mean:

- **Euler and QG**: ``\Phi = 4\pi\hat G`` with ``\hat G`` as above (``\kappa=0``
  for Euler, where ``P(u,0)=-E_2(u)``); QG with short ``L_d`` combines the direct
  QG image sum with the periodic Euler Ewald sum.
- **SQG**: the real-space part of each image is neutralized, subtracting a
  Gaussian charge of the same total (carried in Fourier space instead), and the
  ``-\delta^2/(2r)`` tail of the softening is moved to Fourier space through
  ``\operatorname{erf}(\alpha r)/r``. What remains decays fast enough that a
  finite image block around the minimum image is exact and periodic.

A finite image block of a potential that grows like ``\log r``, which is what
the unneutralized real-space SQG terms do, is not periodic: its value jumps
where the minimum image changes, at half-period separations.

Here ``r=|\mathbf r|``, ``r_\delta`` is the regularized distance, and
``\delta`` is `SQGKernel.δ` (the `δ_sqg` constructor keyword), not the
independent contour-surgery threshold.

## References and Further Reading

- Holzmann, M. & Bernu, B. (2005). *Optimized periodic 1/r Coulomb potential in two dimensions.* J. Comput. Phys. **206**(1), 111–121. [doi:10.1016/j.jcp.2004.11.037](https://doi.org/10.1016/j.jcp.2004.11.037)
- Dritschel, D.G. & Ambaum, M.H.P. (1997). *A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields.* Q. J. R. Meteorol. Soc. **123**(540), 1097--1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)

For the full list used across the theory pages, see [References](references.md).
