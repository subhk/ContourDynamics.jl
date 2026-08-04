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
basis into independent vertical modes.

The symmetric coupling matrix is diagonalized as
``\mathbf{C}=\mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}``, where
``\mathbf{\Lambda}=\operatorname{diag}(\lambda_1,\ldots,\lambda_N)``. Each
eigenmode ``m`` evolves independently:

- If ``|\lambda_m| \approx 0``: **barotropic mode** — uses the Euler kernel
- Otherwise: ``L_d^{(\text{mode})} = 1/\sqrt{|\lambda_m|}`` — uses a QG kernel

Here:

- ``\mathbf{P}`` contains the eigenvectors
- ``\mathbf{P}^{-1}`` transforms physical-layer fields into modal fields (for the symmetric matrices accepted by the package, ``\mathbf{P}^{-1}=\mathbf{P}^{\mathsf T}``)
- ``\mathbf{\Lambda}`` is the diagonal matrix of eigenvalues
- ``\lambda_m`` is the eigenvalue for mode ``m``
- ``L_d^{(\text{mode})}`` is the deformation radius associated with that mode

The public `Ld` argument contains the ``N-1`` nonbarotropic modal radii. The
constructor verifies that they equal ``1/\sqrt{|\lambda_m|}`` for the nonzero
eigenvalues of `coupling`; the approximately zero eigenvalue is the barotropic
Euler mode.

The velocity in physical layers is recovered by projecting back through the
eigenvector matrix. In practical terms, the code solves a set of uncoupled
single-mode problems, then recombines them into layer velocities.

## References and Further Reading

- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)
- Dritschel, D.G. & de la Torre Juárez, M. (2002). *Vortex dynamics in rotating and stratified fluids.* Lecture Notes in Physics **555**, 299--340. [doi:10.1007/3-540-45674-0_11](https://doi.org/10.1007/3-540-45674-0_11)

For more references across contour dynamics and geophysical vortex dynamics, see [References](references.md).
