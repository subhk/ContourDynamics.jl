# Ewald Summation

## Periodic Green's Functions

On a doubly-periodic domain ``[-L_x, L_x) \times [-L_y, L_y)``, the Green's function includes contributions from all periodic images. Direct summation converges slowly, so we use **Ewald splitting** to decompose:

```math
G_{\text{per}}(\mathbf{r}) = G_{\text{real}}(\mathbf{r}) + G_{\text{Fourier}}(\mathbf{r})
```

### Real-Space Sum

```math
G_{\text{real}}(\mathbf{r}) = \frac{1}{4\pi} \sum_{\mathbf{n}} E_1(\alpha^2|\mathbf{r} - \mathbf{L}_\mathbf{n}|^2)
```

where ``E_1`` is the exponential integral, ``\alpha = \sqrt{\pi}/\sqrt{L_xL_y}`` is the splitting parameter used by the implementation, and the sum runs over periodic images ``\mathbf{L}_\mathbf{n} = (2nL_x, 2mL_y)``. The Gaussian damping ensures rapid convergence (typically 2 images suffice).

### Fourier-Space Sum

```math
G_{\text{Fourier}}(\mathbf{r}) = \frac{1}{A} \sum_{\mathbf{k} \neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|^2} \cos(\mathbf{k} \cdot \mathbf{r})
```

where ``A = 4L_xL_y`` is the domain area. The Gaussian factor ``e^{-k^2/(4\alpha^2)}`` ensures rapid convergence in Fourier space.

### Singular Subtraction for Periodic Velocity

The periodic segment velocity uses the same singular-subtraction approach: the log singularity from the central image is handled by the exact unbounded Euler formula, and the smooth correction ``G_{\text{per}} - G_\infty`` is integrated with 3-point Gauss-Legendre quadrature.

## QG Periodic Decomposition

For the QG kernel on a periodic domain, we decompose:

```math
G_{\text{QG,per}} = G_{\text{Euler,per}} - \underbrace{\frac{1}{A}\sum_{\mathbf{k}\neq 0} \frac{\kappa^2}{|\mathbf{k}|^2(|\mathbf{k}|^2 + \kappa^2)}\cos(\mathbf{k}\cdot\mathbf{r})}_{\text{smooth QG correction}}
```

where ``\kappa = 1/L_d``. The Euler periodic part uses the full Ewald machinery, and the QG correction is a smooth, rapidly convergent (``\sim 1/k^4``) Fourier series.

## SQG Periodic Decomposition

For the SQG kernel ``G(r) = -1/(2\pi r)`` on a periodic domain, the Ewald splitting decomposes the periodic sum of ``1/r`` into:

```math
\sum_{\mathbf{n}} \frac{1}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} = \sum_{\mathbf{n}} \frac{\operatorname{erfc}(\alpha|\mathbf{r} - \mathbf{L}_\mathbf{n}|)}{|\mathbf{r} - \mathbf{L}_\mathbf{n}|} + \frac{2\pi}{A}\sum_{\mathbf{k}\neq 0} \frac{e^{-|\mathbf{k}|^2/(4\alpha^2)}}{|\mathbf{k}|}\cos(\mathbf{k}\cdot\mathbf{r})
```

The Fourier coefficients decay as ``1/|\mathbf{k}|`` (compared to ``1/k^2`` for Euler), reflecting the fractional Laplacian's half-order nature.

The periodic segment velocity uses singular subtraction: the regularized unbounded SQG velocity (exact ``\operatorname{arcsinh}`` antiderivative with ``\delta > 0``) handles the ``1/r`` singularity, and the smooth correction is integrated with 3-point Gauss-Legendre quadrature. The central-image correction ``\operatorname{erfc}(\alpha r)/r - 1/\sqrt{r^2+\delta^2}`` is finite at all quadrature points since the evaluation distance ``r > 0``.
