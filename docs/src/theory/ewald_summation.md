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
  correction is the smooth QG–Euler Fourier series described in the next
  section (its coefficients are precomputed in `EwaldCache.corr_coeffs`).

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
- a smooth correction, which is easier to evaluate as a Fourier series

That correction decays like ``|\mathbf{k}|^{-4}``, so it converges much faster than the
raw periodic Green's function would.

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
normalization. For the softened kernel used by the package,
the Ewald representative carries the constant coefficient

```math
C_0=\frac{1}{A}\left(\frac{1}{\alpha\sqrt{\pi}}-\delta\right).
```

The periodic energy diagnostic subtracts ``C_0\Gamma^2/2`` so that its
Hamiltonian corresponds to the mean-free, nonzero-``k`` inversion.

The Fourier coefficients contain an ``\operatorname{erfc}(|\mathbf{k}|/(2\alpha))`` damping factor and a leading ``1/|\mathbf{k}|`` behavior (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature. In practical terms, this means SQG is less smooth than Euler in Fourier space and therefore needs a bit more care numerically.

The periodic segment velocity again uses singular subtraction:

- the regularized unbounded SQG contribution handles the near-singular part,
  analytically for straight segments and by quadrature for cubic arcs
- the periodic correction is smooth enough to integrate with 5-point Gauss-Legendre quadrature

Regularization is applied to every periodic image. For the central image, the
regularized unbounded contribution is supplied by the base velocity and the Ewald
correction is ``-\operatorname{erf}(\alpha r)/r``. This correction remains
bounded at coincidence, where its limit is ``-2\alpha/\sqrt{\pi}``. Each
non-central real-space image adds

```math
\frac{\operatorname{erfc}(\alpha r)}{r}
+ \left(\frac{1}{r_\delta}-\frac{1}{r}\right),
\qquad r_\delta=\sqrt{r^2+\delta^2}.
```

Thus the combined real-space and Fourier sums approximate the periodic
softened kernel ``1/r_\delta`` with the configured truncations. Its nonzero
Fourier modes include the softening factor ``e^{-\delta|\mathbf k|}`` derived
in [Contour Dynamics](contour_dynamics.md#SQG-Kernel); finite ``\delta`` changes
the inversion from the unregularized SQG model.

Here ``r=|\mathbf r|``, ``r_\delta`` is the regularized distance, and
``\delta`` is `SQGKernel.δ` (the `δ_sqg` constructor keyword), not the
independent contour-surgery threshold.

## References and Further Reading

- Holzmann, M. & Bernu, B. (2005). *Optimized periodic 1/r Coulomb potential in two dimensions.* J. Comput. Phys. **206**(1), 111–121. [doi:10.1016/j.jcp.2004.11.037](https://doi.org/10.1016/j.jcp.2004.11.037)
- Dritschel, D.G. & Ambaum, M.H.P. (1997). *A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields.* Q. J. R. Meteorol. Soc. **123**(540), 1097--1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)

For the full list used across the theory pages, see [References](references.md).
