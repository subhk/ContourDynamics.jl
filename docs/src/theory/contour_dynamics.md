# Contour Dynamics

## Vortex Patches and Green's Functions

Consider a 2D inviscid flow with piecewise-constant potential vorticity (PV). The streamfunction ``\psi`` satisfies:

```math
\mathcal{L}\psi = q(\mathbf{x})
```

where ``\mathbf{x}=(x,y)`` is position, ``\psi(\mathbf{x})`` is the
streamfunction, ``q(\mathbf{x})`` is PV (vorticity for Euler), and
``\mathcal{L}`` is the PV inversion operator (``\nabla^2`` for Euler,
``\nabla^2-L_d^{-2}`` for single-layer QG). Here ``\nabla^2`` is the horizontal
Laplacian and ``L_d`` is the deformation radius. The solution is:

```math
\psi(\mathbf{x}) = \int\!\!\int G_\psi(|\mathbf{x} - \mathbf{x}'|) \, q(\mathbf{x}') \, dA'
```

Here:

- ``\psi`` is the streamfunction
- ``q(\mathbf{x}')`` is the PV field at source point ``\mathbf{x}'``
- ``\mathbf{x}`` is the point where we want to evaluate the solution
- ``G_\psi`` is the streamfunction Green's function, meaning the response at ``\mathbf{x}`` to a unit source placed at ``\mathbf{x}'``
- ``r=|\mathbf{x}-\mathbf{x}'|`` is source-target distance
- ``dA'`` is the area element at the source point, and the double integral covers the PV-supporting area

The scalar kernel multiplying the oriented tangent in a contour integral has
the opposite sign, ``G=-G_\psi``, after integration by parts. For Euler,
``G_\psi=\log r/(2\pi)``; for QG, ``G_\psi=-K_0(r/L_d)/(2\pi)``.
For the patch formulas below, take counterclockwise contours with jumps defined
as the inside value minus the outside value.

For a single vortex patch with uniform PV jump ``q`` bounded by contour ``C``, the velocity ``\mathbf{u} = (-\psi_y, \psi_x)`` can be converted from an area integral to a **contour integral** via Green's theorem:

```math
\mathbf{u}(\mathbf{x}) = -\frac{q}{4\pi} \oint_C \log|\mathbf{x} - \mathbf{x}'|^2 \, d\mathbf{x}'
```

Here:

- ``\mathbf{u}(\mathbf{x})`` is the velocity at the target point ``\mathbf{x}``
- ``C`` is the patch boundary
- ``\mathbf{x}'`` is now a point moving along that boundary
- ``d\mathbf{x}'`` is a short tangent vector along the contour
- ``q`` is the jump in PV across ``C``; multiple contours contribute by summing their individual jumps
- the contour integral ``\oint_C`` means “walk once around the closed boundary”

This is the contour dynamics equation: the velocity at any point depends only on the **boundary** of the PV patch, not its interior.

## Segment Integration

The contour is stored as nodes ``\{\mathbf{x}_j\}``. Exactly straight pieces use
the segment formulas below; when endpoint curvatures are nonzero, the velocity
paths evaluate the same contour integral on the cubic Dritschel arc described
in the contour-surgery theory page. A straight segment from ``\mathbf{a}`` to
``\mathbf{b}`` contributes:

```math
\mathbf{v}_{\text{seg}}(\mathbf{x}) = -\frac{1}{4\pi}(\mathbf{b}-\mathbf{a}) \int_0^1 \log|\mathbf{x} - \mathbf{a} - t(\mathbf{b}-\mathbf{a})|^2 \, dt
```

In this formula:

- ``\mathbf{a}`` and ``\mathbf{b}`` are the endpoints of one straight contour segment
- ``\mathbf{b}-\mathbf{a}`` is the segment direction
- ``\mathbf{x}`` is the target position and ``\mathbf{x}'(t)=\mathbf{a}+t(\mathbf{b}-\mathbf{a})`` is the source position
- ``t \in [0,1]`` moves from one endpoint to the other
- ``\mathbf{v}_{\text{seg}}`` is the unit-PV-jump velocity contribution from that one segment; the contour's stored PV jump supplies the multiplier in the full sum

The full contour velocity is the sum over all segment or cubic-arc
contributions from all contours.

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

Here ``\hat{\mathbf{t}}=(\mathbf{b}-\mathbf{a})/|\mathbf{b}-\mathbf{a}|`` is
the unit tangent, while ``u_a=(\mathbf x-\mathbf a)\cdot\hat{\mathbf t}`` and
``u_b=(\mathbf x-\mathbf b)\cdot\hat{\mathbf t}=u_a-|\mathbf b-\mathbf a|``
are the signed target coordinates measured from the two endpoints. The
normal coordinate ``h`` is constant along a straight segment.

This straight-segment fallback is exact and introduces no quadrature error.

### QG Kernel

For the QG scalar kernel ``G(r) = \frac{1}{2\pi}K_0(r/L_d)`` used in the contour
integral, we use **singular subtraction**:

```math
K_0(r/L_d) = -\log(r) + \underbrace{\left[K_0(r/L_d) + \log(r)\right]}_{\text{smooth at } r=0}
```

The logarithmic singularity is handled by the exact Euler antiderivative. The smooth remainder ``K_0(r/L_d) + \log(r) \to \log(2L_d) - \gamma_E`` as ``r \to 0`` is integrated with **5-point Gauss-Legendre quadrature**.

Here:

- ``K_0`` is the modified Bessel function of the second kind
- ``r`` is the distance from the target point to the integration point
- ``L_d`` is the deformation radius
- ``\gamma_E`` is the Euler-Mascheroni constant (the subscript distinguishes it from unrelated vortex-scale parameters)

The QG kernel is therefore split into an Euler-like singular part and a smooth
remainder suitable for standard quadrature.

### SQG Kernel

Unregularized surface quasi-geostrophic (SQG) dynamics replaces the Laplacian PV
inversion with a **fractional Laplacian**:

```math
-(-\nabla^2)^{1/2}\psi = \theta
```

where ``\theta`` is the normalized surface scalar in the package's
lower-boundary convention, with units of velocity. It represents surface
buoyancy after the physical normalization needed to obtain this inversion.
The streamfunction Green's function is
``G_\psi(r)=-1/(2\pi r)``, and integration by parts gives the contour integral:

```math
\mathbf{u}(\mathbf{x}) = \frac{\Delta\theta}{2\pi}\oint_C \frac{d\mathbf{x}'}{|\mathbf{x}-\mathbf{x}'|}
```

Here ``(-\nabla^2)^{1/2}`` is the half-order fractional Laplacian,
``\theta`` plays the role of the active scalar, ``r=|\mathbf{x}-\mathbf{x}'|``,
``C`` is its patch boundary, ``\Delta\theta`` is the scalar jump carried by the
contour, and ``d\mathbf{x}'`` is the oriented tangent line
element. The kernel is more singular than in Euler, so SQG tends to generate
sharper fronts and stronger filamentation.

This sign convention corresponds to a lower boundary with fluid above it and
keeps positive stored jumps counter-clockwise, consistently with the Euler and
QG kernels. Reversing the physical boundary orientation reverses the meaning
of the stored buoyancy sign.

Unlike the Euler and QG kernels, the unregularized SQG velocity is **singular at
the contour boundary**: the tangential component diverges logarithmically. The
implementation requires a regularization length ``\delta>0`` and replaces
``1/r`` with ``1/\sqrt{r^2+\delta^2}``.

This softening changes the inversion. Fourier-transforming the implemented
streamfunction kernel ``-1/(2\pi\sqrt{r^2+\delta^2})`` gives

```math
\widehat{\psi_\delta}(\mathbf{k})
=-\frac{e^{-\delta|\mathbf{k}|}}{|\mathbf{k}|}\widehat\theta(\mathbf{k}),
\qquad \mathbf{k}\ne0.
```

Here hats denote horizontal Fourier transforms, ``\mathbf k`` is the
wavevector, and ``\psi_\delta`` is the regularized streamfunction used to
advect the contours. The unregularized relation in Held et al. (1995), Eq. (13),
is recovered as ``\delta\to0`` at fixed nonzero wavenumber. At finite ``\delta``,
the extra exponential factor suppresses small-scale velocity; it is a model
regularization as well as a way to evaluate boundary velocities. The associated
Hamiltonian is described in [Diagnostics](../api/diagnostics.md).

For a straight segment, let ``h_{\mathrm{eff}}^2=h^2+\delta^2``. The regularized
segment integral has a closed-form antiderivative:

```math
F(u) = \log\!\left(u + \sqrt{u^2 + h_{\text{eff}}^2}\right) = \operatorname{arcsinh}\!\left(\frac{u}{\sqrt{h_{\text{eff}}^2}}\right) + \text{const}
```

Here:

- ``\delta`` is the regularization length used by the implementation
- ``h_{\text{eff}}`` is the regularized normal distance
- ``\text{const}`` is an arbitrary additive integration constant, which cancels in endpoint differences

For straight segments, the regularized unit-jump segment velocity remains exact:

```math
\mathbf{v}_{\text{seg}} = \frac{1}{2\pi}\hat{\mathbf{t}} \left[F(u_a) - F(u_b)\right]
```

The tangent ``\hat{\mathbf{t}}`` and endpoint coordinates ``u_a,u_b`` have the
same definitions as in the Euler formula above. In the Julia API this SQG
regularization is `δ_sqg`; it is distinct from the surgery threshold also
written ``\delta`` on the surgery page.

## References and Further Reading

- Zabusky, N.J., Hughes, M.H. & Roberts, K.V. (1979). *Contour dynamics for the Euler equations in two dimensions.* J. Comput. Phys. **30**(1), 96--106. [doi:10.1016/0021-9991(79)90089-5](https://doi.org/10.1016/0021-9991(79)90089-5)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Held, I.M., Pierrehumbert, R.T., Garner, S.T. & Swanson, K.L. (1995). *Surface quasi-geostrophic dynamics.* J. Fluid Mech. **282**, 1--20. [doi:10.1017/S0022112095000012](https://doi.org/10.1017/S0022112095000012)

For a broader bibliography, see [References](references.md).
