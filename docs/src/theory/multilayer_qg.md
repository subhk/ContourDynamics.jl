# Multi-Layer QG

## Modal Decomposition

For an ``N``-layer QG system, PV inversion combines the horizontal Laplacian in
each layer with a layer-stretching matrix ``\mathbf{C}``:

```math
q_i = \nabla^2\psi_i + \sum_{j=1}^{N} C_{ij}\psi_j,
\qquad i=1,\ldots,N.
```

Here:

- ``N`` is the number of physical layers
- ``i`` is the affected-layer index and ``j`` is the source-layer index
- ``q_i`` is the PV in layer ``i``
- ``\psi_j`` is the streamfunction in layer ``j``
- ``\nabla^2`` is the horizontal Laplacian, acting on ``\psi_i``
- ``C_{ij}`` measures how strongly layer ``j`` influences layer ``i``
- ``\mathbf{C}`` is the full layer-coupling matrix

Instead of evolving that coupled system directly, the implementation changes
basis into independent vertical modes. When the layer depths are unequal, the
physical matrix ``\mathbf C`` is generally **not symmetric**. Let
``\mathbf W=\operatorname{diag}(H_1,\ldots,H_N)`` contain the positive layer
thicknesses. A physical stretching matrix obeys

```math
\mathbf W\mathbf C=\mathbf C^{\mathsf T}\mathbf W,
\qquad\text{or equivalently}\qquad
H_i C_{ij}=H_j C_{ji}.
```

The implementation first forms the symmetric similarity transform

```math
\mathbf S=\mathbf W^{1/2}\mathbf C\mathbf W^{-1/2}
         =\mathbf P\mathbf\Lambda\mathbf P^{\mathsf T},
```

where ``\mathbf P`` is orthogonal and
``\mathbf\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)``. The modal
PV and the reconstruction of physical-layer streamfunction are different
weighted maps:

```math
\mathbf q_{\mathrm m}=\mathbf P^{\mathsf T}\mathbf W^{1/2}\mathbf q,
\qquad
\boldsymbol\psi=\mathbf W^{-1/2}\mathbf P\boldsymbol\psi_{\mathrm m}.
```

These are stored as `physical_to_modal` and `modal_to_physical`, respectively.
They reduce to ``\mathbf P^{\mathsf T}`` and ``\mathbf P`` for the historical
equal-depth, symmetric input. Each eigenmode ``m`` then evolves independently:

- If ``|\lambda_m| \approx 0``: **barotropic mode** — uses the Euler kernel
- Otherwise: ``L_d^{(\text{mode})} = 1/\sqrt{|\lambda_m|}`` — uses a QG kernel

Here:

- ``\mathbf{P}`` contains the eigenvectors in the thickness-weighted representation
- ``\mathbf P^{\mathsf T}\mathbf W^{1/2}`` transforms physical-layer PV into modal PV
- ``\mathbf W^{-1/2}\mathbf P`` transforms modal streamfunction or velocity back to physical layers
- ``\mathbf{\Lambda}`` is the diagonal matrix of eigenvalues
- ``\lambda_m`` is the eigenvalue for mode ``m``
- ``L_d^{(\text{mode})}`` is the deformation radius associated with that mode

The public `Ld` argument contains the ``N-1`` nonbarotropic modal radii. The
constructor verifies that they equal ``1/\sqrt{|\lambda_m|}`` for the nonzero
eigenvalues of `coupling`; the approximately zero eigenvalue is the barotropic
Euler mode.

The constructor rejects positive coupling eigenvalues: the screened ``K_0``
inversion used by nonbarotropic modes requires a negative-semidefinite
stretching operator. Mode classification uses the same scale-aware eigenvalue
tolerance during construction, velocity evaluation, and diagnostics.

The velocity in physical layers is recovered by projecting back through the
weighted reconstruction matrix. In practical terms, the code solves a set of
uncoupled single-mode problems, then recombines them into layer velocities.

The same weighting is required by the Hamiltonian:

```math
E=-\frac12\int \mathbf q^{\mathsf T}\mathbf W\boldsymbol\psi\,\mathrm dA
  =-\frac12\sum_m\int q_{\mathrm m}^{(m)}\psi_{\mathrm m}^{(m)}\,\mathrm dA.
```

Consequently, explicit layer thicknesses set the overall energy scale as well
as their relative weights. If `coupling` is connected and nonsymmetric,
`MultiLayerQGKernel(Ld, coupling)` can infer the thickness ratios; because the
matrix determines them only up to a common factor, inferred values are
normalized to have mean one. Use
`MultiLayerQGKernel(Ld, coupling, H)` or the `layer_thicknesses=H` keyword when
the physical energy scale matters. A disconnected nonsymmetric matrix requires
explicit thicknesses because the relative scale of its blocks is ambiguous.

## References and Further Reading

- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)
- Dritschel, D.G. & de la Torre Juárez, M. (2002). *Vortex dynamics in rotating and stratified fluids.* Lecture Notes in Physics **555**, 299--340. [doi:10.1007/3-540-45674-0_11](https://doi.org/10.1007/3-540-45674-0_11)
- [pyqg layered-QG equations](https://pyqg.readthedocs.io/en/latest/equations/notation_layered.html), including the ``H_i``-weighted energy.

For more references across contour dynamics and geophysical vortex dynamics, see [References](references.md).
