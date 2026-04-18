# Contour Dynamics

## Vortex Patches and Green's Functions

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

## Segment Integration

The contour is discretized into piecewise-linear segments connecting nodes ``\{\mathbf{x}_j\}``. Each segment from ``\mathbf{a}`` to ``\mathbf{b}`` contributes:

```math
\mathbf{v}_{\text{seg}}(\mathbf{x}) = -\frac{1}{4\pi}(\mathbf{b}-\mathbf{a}) \int_0^1 \log|\mathbf{x} - \mathbf{a} - t(\mathbf{b}-\mathbf{a})|^2 \, dt
```

### Euler Kernel

For the Euler kernel, this integral has a **closed-form antiderivative**. Projecting onto the segment's tangent and normal directions:

```math
F(u) = u\log(u^2 + h^2) - 2u + 2|h|\arctan(u/|h|)
```

where ``u`` is the tangential coordinate and ``h`` is the normal distance from ``\mathbf{x}`` to the segment line. The segment velocity is:

```math
\mathbf{v}_{\text{seg}} = -\frac{1}{4\pi}\hat{\mathbf{t}} \left[F(u_a) - F(u_b)\right]
```

This is **exact** — no quadrature error.

### QG Kernel

For the QG scalar kernel ``G(r) = \frac{1}{2\pi}K_0(r/L_d)`` used in the contour
integral, we use **singular subtraction**:

```math
K_0(r/L_d) = -\log(r) + \underbrace{\left[K_0(r/L_d) + \log(r)\right]}_{\text{smooth at } r=0}
```

The logarithmic singularity is handled by the exact Euler antiderivative. The smooth remainder ``K_0(r/L_d) + \log(r) \to \log(2L_d) - \gamma`` as ``r \to 0`` is integrated with **5-point Gauss-Legendre quadrature**.

### SQG Kernel

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
