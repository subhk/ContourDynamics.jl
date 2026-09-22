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

Both CPU and GPU energy paths join the stored nodes with **straight segments**
and apply **3×3 Gauss–Legendre quadrature** to each segment pair. Velocity
evaluation generally uses cubic Dritschel arcs. Energy therefore approximates
the continuum Hamiltonian on the polygonal geometry; it is not an exact
invariant of the discrete velocity and RK4 update. When assessing conservation,
check convergence with node resolution, timestep, and periodic truncation, and
account for changes introduced by remeshing and reconnection.

For single-layer QG, with ``\kappa=1/L_d`` and
``q=(\nabla^2-\kappa^2)\psi``, the reported positive Hamiltonian is

```math
H=-\frac12\int q\psi\,dA.
```

On a periodic domain of area ``A``, this includes the spatially constant
Helmholtz mode ``\Gamma^2/(2A\kappa^2)``. Multi-layer QG energy applies the
same formula independently to each orthonormal vertical mode, omitting only
the non-invertible constant barotropic Euler mode.

For SQG, the unregularized lower-boundary convention is
``\theta=-(-\nabla^2)^{1/2}\psi``. The implementation uses a softened kernel
with ``\delta>0``, giving a regularized streamfunction ``\psi_\delta`` whose
nonzero Fourier modes satisfy

```math
\widehat{\psi_\delta}(\mathbf{k})
=-\frac{e^{-\delta|\mathbf{k}|}}{|\mathbf{k}|}\widehat\theta(\mathbf{k}),
\qquad \mathbf{k}\ne0.
```

Thus finite ``\delta`` changes the inversion as described in
[Contour Dynamics](../theory/contour_dynamics.md). In unbounded space, the
regularized Hamiltonian approximated by `energy` is

```math
H_\delta=-\frac12\int \theta\psi_\delta\,dA
 =\frac{1}{4\pi}\iint
 \frac{\theta(\mathbf{x})\theta(\mathbf{x}')}
 {\sqrt{|\mathbf{x}-\mathbf{x}'|^2+\delta^2}}\,dA\,dA'.
```

On a periodic domain the kernel is periodized and the spatially constant mode
is excluded. The inversion acts on the mean-free part of ``\theta`` and retains
the same ``e^{-\delta|\mathbf{k}|}`` softening factor for nonzero modes, subject
to the configured Ewald truncation.

```@docs
vortex_area
centroid
ellipse_moments
energy
enstrophy
circulation
angular_momentum
```
