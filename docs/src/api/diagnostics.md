# API Reference: Diagnostics

`energy(prob)` is available for single-layer Euler, QG, and SQG problems on
unbounded and periodic domains, and for multi-layer QG problems.

For each closed contour ``i``, let ``A_i`` be its signed area and ``q_i`` its
stored PV jump. The scalar diagnostics use

```math
\Gamma=\sum_i q_iA_i,
\qquad
Z=\frac12\sum_i q_i^2A_i,
\qquad
I=\sum_i q_i\int_{A_i}|\mathbf{x}|^2\,dA.
```

Here ``\Gamma`` is circulation, ``Z`` is the package's contour-wise enstrophy,
``I`` is angular momentum, ``\mathbf{x}=(x,y)`` is position relative to the
origin, and ``dA`` is an area element inside contour ``i``. Counterclockwise
boundaries have positive ``A_i`` and clockwise inner boundaries have negative
``A_i``. Spanning contours are excluded because they do not enclose a finite
area. For arbitrary nested multi-jump contours, `enstrophy` omits cross-terms
from squaring the reconstructed piecewise PV field; see its docstring below.

`energy` evaluates the kernel- and domain-specific symmetric double contour
integral. Its normalization, Green's function, and regularization therefore
follow the selected Euler, QG, SQG, or multi-layer kernel rather than a single
universal scalar formula. See [Contour Dynamics](../theory/contour_dynamics.md)
and the [notation glossary](../theory/notation.md).

For single-layer QG, with ``\kappa=1/L_d`` and
``q=(\nabla^2-\kappa^2)\psi``, the reported positive Hamiltonian is

```math
H=-\frac12\int q\psi\,dA.
```

On a periodic domain of area ``A``, this includes the spatially constant
Helmholtz mode ``\Gamma^2/(2A\kappa^2)``. Multi-layer QG energy applies the
same formula independently to each orthonormal vertical mode, omitting only
the non-invertible constant barotropic Euler mode.

For SQG, the package uses the lower-boundary convention
``\theta=-(-\nabla^2)^{1/2}\psi`` and the softened contour kernel
``1/\sqrt{r^2+\delta^2}``. The reported positive Hamiltonian is

```math
H=-\frac12\int \theta\psi\,dA
 =\frac{1}{4\pi}\iint
 \frac{\theta(\mathbf{x})\theta(\mathbf{x}')}
 {\sqrt{|\mathbf{x}-\mathbf{x}'|^2+\delta^2}}\,dA\,dA'.
```

On a periodic domain the spatially constant fractional-Laplacian mode is
excluded; equivalently, the inversion acts on the mean-free part of
``\theta``.

```@docs
vortex_area
centroid
ellipse_moments
energy
enstrophy
circulation
angular_momentum
```
