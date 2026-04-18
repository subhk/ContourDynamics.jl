# Ewald Summation

## Periodic Green's Functions

On a doubly-periodic domain ``[-L_x, L_x) \times [-L_y, L_y)``, the Green's function includes contributions from all periodic images. Direct summation converges slowly, so we use **Ewald splitting** to decompose:

```math
G_{\text{per}}(\mathbf{r}) = G_{\text{real}}(\mathbf{r}) + G_{\text{Fourier}}(\mathbf{r})
```

The basic problem is this:

- in a periodic domain, each contour interacts not only with the copy you see in
  the main domain, but also with infinitely many translated copies
- summing those copies directly is too slow and converges poorly
- Ewald summation rewrites the same periodic Green's function as two rapidly
  convergent pieces

In this page:

- ``\mathbf{r}`` is the displacement from the source point to the target point
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

- ``E_1`` is the exponential integral
- ``\alpha = \sqrt{\pi}/\sqrt{L_xL_y}`` is the splitting parameter used by the implementation
- ``\mathbf{L}_\mathbf{n} = (2nL_x, 2mL_y)`` is the lattice shift to a periodic image
- the integers ``n`` and ``m`` label image copies of the domain

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
- ``\mathbf{k}\cdot\mathbf{r}`` is the usual Fourier phase
- the term ``\mathbf{k} \neq 0`` excludes the zero mode
- the Gaussian factor ``e^{-|\mathbf{k}|^2/(4\alpha^2)}`` makes the Fourier sum converge rapidly

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

Here ``\kappa = 1/L_d`` is the inverse deformation radius. The key idea is that
the QG periodic kernel can be written as:

- an Euler-like periodic part, which already has a validated Ewald treatment
- a smooth correction, which is easier to evaluate as a Fourier series

That correction decays like ``1/k^4``, so it converges much faster than the
raw periodic Green's function would.

## SQG Periodic Decomposition

For the SQG kernel ``G(r) = -1/(2\pi r)`` on a periodic domain, the Ewald splitting decomposes the periodic sum of ``1/r`` into:

```math
\sum_{\mathbf{n}} \frac{1}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} = \sum_{\mathbf{n}} \frac{\operatorname{erfc}(\alpha|\mathbf{r} - \mathbf{L}_\mathbf{n}|)}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} + \frac{2\pi}{A}\sum_{\mathbf{k}\neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|}\cos(\mathbf{k}\cdot\mathbf{r})
```

The Fourier coefficients decay as ``1/|\mathbf{k}|`` (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature. In practical terms, this means SQG is less smooth than Euler in Fourier space and therefore needs a bit more care numerically.

The periodic segment velocity again uses singular subtraction:

- the regularized unbounded SQG segment velocity handles the near-singular part analytically
- the periodic correction is smooth enough to integrate with 3-point Gauss-Legendre quadrature

The central-image correction ``\operatorname{erfc}(\alpha r)/r - 1/\sqrt{r^2+\delta^2}`` stays finite at quadrature points because the evaluation distance ``r > 0`` along the numerical integration rule.

## References and Further Reading

- Dritschel, D.G. & Ambaum, M.H.P. (1997). *A contour-advective semi-Lagrangian numerical algorithm for simulating fine-scale conservative dynamical fields.* Q. J. R. Meteorol. Soc. **123**(540), 1097--1130. [doi:10.1002/qj.49712354015](https://doi.org/10.1002/qj.49712354015)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)

For the full list used across the theory pages, see [References](/theory/references).
