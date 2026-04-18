# Multi-Layer QG

## Modal Decomposition

For an ``N``-layer QG system, the PV in each layer is coupled to the streamfunction via the coupling matrix ``\mathbf{C}``:

```math
q_i = \sum_j C_{ij} \psi_j
```

Here:

- ``q_i`` is the PV in layer ``i``
- ``\psi_j`` is the streamfunction in layer ``j``
- ``C_{ij}`` tells you how strongly layer ``j`` influences layer ``i``
- ``\mathbf{C}`` is the full layer-coupling matrix

Instead of evolving that coupled system directly, the implementation changes
basis into independent vertical modes.

The coupling matrix is diagonalized: ``\mathbf{C} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}``. Each eigenmode with eigenvalue ``\lambda_m`` evolves independently:

- If ``|\lambda_m| \approx 0``: **barotropic mode** — uses the Euler kernel
- Otherwise: ``L_d^{(\text{mode})} = 1/\sqrt{|\lambda_m|}`` — uses a QG kernel

Here:

- ``\mathbf{P}`` contains the eigenvectors
- ``\mathbf{\Lambda}`` is the diagonal matrix of eigenvalues
- ``\lambda_m`` is the eigenvalue for mode ``m``
- ``L_d^{(\text{mode})}`` is the deformation radius associated with that mode

The velocity in physical layers is recovered by projecting back through the
eigenvector matrix. In practical terms, the code solves a set of uncoupled
single-mode problems, then recombines them into layer velocities.

## References and Further Reading

- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed. Springer. [doi:10.1007/978-1-4612-4650-3](https://doi.org/10.1007/978-1-4612-4650-3)
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed. Cambridge University Press. [doi:10.1017/9781107588417](https://doi.org/10.1017/9781107588417)
- Dritschel, D.G. & de la Torre Juárez, M. (2002). *Vortex dynamics in rotating and stratified fluids.* Lecture Notes in Physics **555**, 299--340. [doi:10.1007/3-540-45674-0_11](https://doi.org/10.1007/3-540-45674-0_11)

For more references across contour dynamics and geophysical vortex dynamics, see [References](/theory/references).
