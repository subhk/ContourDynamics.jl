# Ewald Summation

## Periodic Green's Functions

On a doubly-periodic domain ``[-L_x, L_x) \times [-L_y, L_y)``, the Green's function includes contributions from all periodic images. Direct summation converges slowly, so we use **Ewald splitting** to decompose:

```math
G_{\text{per}}(\mathbf{r}) = G_{\text{real}}(\mathbf{r}) + G_{\text{Fourier}}(\mathbf{r})
```

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

The periodic segment velocity uses the same singular-subtraction approach as the
unbounded formulation:

- the singular part is handled analytically using the exact unbounded segment formula
- only the smooth correction ``G_{\text{per}} - G_\infty`` is left for numerical quadrature

Here ``G_\infty`` means the corresponding unbounded-space Green's function. This
is important because quadrature is most reliable on smooth integrands, not on
functions with logarithmic or stronger singular behavior.

## QG Periodic Decomposition

For the QG kernel on a periodic domain, we decompose:

```math
G_{\text{QG,per}} = G_{\text{Euler,per}} - \underbrace{\frac{1}{A}\sum_{\mathbf{k}\neq 0} \frac{\kappa^2}{|\mathbf{k}|^2(|\mathbf{k}|^2 + \kappa^2)}\cos(\mathbf{k}\cdot\mathbf{r})}_{\text{smooth QG correction}}
```

Here ``G_{\text{QG,per}}`` and ``G_{\text{Euler,per}}`` are the periodic QG
and Euler Green's functions, ``\kappa=1/L_d`` is inverse deformation radius,
and ``A``, ``\mathbf{k}``, and ``\mathbf{r}`` retain their definitions above.
The key idea is that the QG periodic kernel can be written as:

- an Euler-like periodic part, which already has a validated Ewald treatment
- a smooth correction, which is easier to evaluate as a Fourier series

That correction decays like ``|\mathbf{k}|^{-4}``, so it converges much faster than the
raw periodic Green's function would.

## SQG Periodic Decomposition

For the SQG kernel ``G(r) = 1/(2\pi r)`` on a periodic domain, the Ewald splitting decomposes the periodic sum of ``1/r`` into:

```math
\sum_{\mathbf{n}} \frac{1}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} = \sum_{\mathbf{n}} \frac{\operatorname{erfc}(\alpha|\mathbf{r} - \mathbf{L}_\mathbf{n}|)}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} + \frac{2\pi}{A}\sum_{\mathbf{k}\neq 0} \frac{\operatorname{erfc}(|\mathbf{k}|/(2\alpha))}{|\mathbf{k}|}\cos(\mathbf{k}\cdot\mathbf{r})
```

The image index ``\mathbf n``, lattice shift ``\mathbf L_{\mathbf n}``,
wavevector ``\mathbf k``, splitting parameter ``\alpha``, and area ``A`` are
defined above. The complementary error function is
``\operatorname{erfc}(z)=1-\operatorname{erf}(z)``. Both sums omit terms only
through the configured finite `n_images` and `n_fourier` truncations; the
displayed equation is the infinite-sum identity.

The Fourier coefficients contain an ``\operatorname{erfc}(|\mathbf{k}|/(2\alpha))`` damping factor and a leading ``1/|\mathbf{k}|`` behavior (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature. In practical terms, this means SQG is less smooth than Euler in Fourier space and therefore needs a bit more care numerically.

The periodic segment velocity again uses singular subtraction:

- the regularized unbounded SQG segment velocity handles the near-singular part analytically
- the periodic correction is smooth enough to integrate with 5-point Gauss-Legendre quadrature

Regularization is applied to every periodic image. For the central image, the
exact regularized unbounded contribution is added analytically and the Ewald
correction is ``-\operatorname{erf}(\alpha r)/r``. This correction remains
bounded at coincidence, where its limit is ``-2\alpha/\sqrt{\pi}``. Each
non-central real-space image adds

```math
\frac{\operatorname{erfc}(\alpha r)}{r}
+ \left(\frac{1}{r_\delta}-\frac{1}{r}\right),
\qquad r_\delta=\sqrt{r^2+\delta^2}.
```

Thus the combined real-space and Fourier sums represent the periodic sum of
the documented softened kernel ``1/r_\delta``, rather than making the answer
depend on where the Ewald split is introduced.

Here ``r=|\mathbf r|``, ``r_\delta`` is the regularized distance, and
``\delta`` is `SQGKernel.delta` (the `delta_sqg` constructor keyword), not the
independent contour-surgery threshold.

## References and Further Reading

- Dritschel, D.G. & Ambaum, M.H.P. (1997). *A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields.* Q. J. R. Meteorol. Soc. **123**(540), 1097--1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)

For the full list used across the theory pages, see [References](references.md).
