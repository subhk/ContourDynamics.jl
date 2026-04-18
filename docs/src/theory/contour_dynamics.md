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

Here:

- ``\psi`` is the streamfunction
- ``q(\mathbf{x}')`` is the PV field at source point ``\mathbf{x}'``
- ``\mathbf{x}`` is the point where we want to evaluate the solution
- ``G`` is the Green's function, meaning the response at ``\mathbf{x}`` to a unit source placed at ``\mathbf{x}'``
- ``dA'`` means we integrate over patch area

In plain terms, ``G`` tells us how strongly one part of the patch influences another part.

For a single vortex patch with uniform PV jump ``q`` bounded by contour ``C``, the velocity ``\mathbf{u} = (-\psi_y, \psi_x)`` can be converted from an area integral to a **contour integral** via Green's theorem:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

Here:

- ``\mathbf{u}(\mathbf{x})`` is the velocity at the target point ``\mathbf{x}``
- ``C`` is the patch boundary
- ``\mathbf{x}'`` is now a point moving along that boundary
- ``d\mathbf{x}'`` is a short tangent vector along the contour
- the contour integral ``\oint_C`` means “walk once around the closed boundary”

This is the contour dynamics equation: the velocity at any point depends only on the **boundary** of the PV patch, not its interior.

## Segment Integration

The contour is discretized into piecewise-linear segments connecting nodes ``\{\mathbf{x}_j\}``. Each segment from ``\mathbf{a}`` to ``\mathbf{b}`` contributes:

```math
\mathbf{v}_{\text{seg}}(\mathbf{x}) = -\frac{1}{4\pi}(\mathbf{b}-\mathbf{a}) \int_0^1 \log|\mathbf{x} - \mathbf{a} - t(\mathbf{b}-\mathbf{a})|^2 \, dt
```

In this formula:

- ``\mathbf{a}`` and ``\mathbf{b}`` are the endpoints of one straight contour segment
- ``\mathbf{b}-\mathbf{a}`` is the segment direction
- ``t \in [0,1]`` moves from one endpoint to the other
- ``\mathbf{v}_{\text{seg}}`` is the velocity contribution from that one segment

The full contour velocity is the sum over all segments from all contours.

### Euler Kernel

For the Euler kernel, this integral has a **closed-form antiderivative**. Projecting onto the segment's tangent and normal directions:

```math
F(u) = u\log(u^2 + h^2) - 2u + 2|h|\arctan(u/|h|)
```

where:

- ``u`` is distance measured along the segment direction
- ``h`` is distance measured perpendicular to the segment
- ``F(u)`` is the antiderivative used to evaluate the line integral exactly

The segment velocity is:

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

Here:

- ``K_0`` is the modified Bessel function of the second kind
- ``r`` is the distance from the target point to the integration point
- ``L_d`` is the deformation radius
- ``\gamma`` is the Euler-Mascheroni constant

The practical idea is simple: split the QG kernel into one part that looks like
Euler near the singularity and one part that is smooth enough for standard
quadrature.

### SQG Kernel

Surface quasi-geostrophic (SQG) dynamics replaces the Laplacian PV inversion with a **fractional Laplacian**:

```math
(-\nabla^2)^{1/2}\psi = \theta
```

where ``\theta`` is the surface buoyancy. The Green's function is ``G(r) = -1/(2\pi r)``, and the contour integral becomes:

```math
\mathbf{u}(\mathbf{x}) = -\frac{1}{2\pi}\oint_C \frac{d\mathbf{x}'}{|\mathbf{x}-\mathbf{x}'|}
```

Here ``\theta`` plays the role of the active scalar, and the kernel is more
singular than in Euler. That is why SQG tends to generate sharper fronts and
stronger filamentation.

The segment integral has a closed-form antiderivative:

```math
F(u) = \log\!\left(u + \sqrt{u^2 + h_{\text{eff}}^2}\right) = \operatorname{arcsinh}\!\left(\frac{u}{\sqrt{h_{\text{eff}}^2}}\right) + \text{const}
```

Unlike the Euler and QG kernels, the SQG velocity is **singular at the contour boundary** — the tangential component diverges logarithmically. A regularization ``\delta > 0`` is required, replacing ``1/r`` with ``1/\sqrt{r^2 + \delta^2}`` so that ``h_{\text{eff}}^2 = h^2 + \delta^2``.

Here:

- ``\delta`` is the regularization length used by the implementation
- ``h_{\text{eff}}`` is the regularized normal distance

The segment velocity remains exact (no quadrature):

```math
\mathbf{v}_{\text{seg}} = -\frac{1}{2\pi}\hat{\mathbf{t}} \left[F(u_a) - F(u_b)\right]
```

## References and Further Reading

- Zabusky, N.J., Hughes, M.H. & Roberts, K.V. (1979). *Contour dynamics for the Euler equations in two dimensions.* J. Comput. Phys. **30**(1), 96--106. [doi:10.1016/0021-9991(79)90089-5](https://doi.org/10.1016/0021-9991(79)90089-5)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995). *Surface quasi-geostrophic dynamics.* J. Fluid Mech. **282**, 1--20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)

For a broader bibliography, see [References](/theory/references).
