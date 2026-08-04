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

```@docs
vortex_area
centroid
ellipse_moments
energy
enstrophy
circulation
angular_momentum
```
