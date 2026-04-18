# Multi-Layer QG

## Modal Decomposition

For an ``N``-layer QG system, the PV in each layer is coupled to the streamfunction via the coupling matrix ``\mathbf{C}``:

```math
q_i = \sum_j C_{ij} \psi_j
```

The coupling matrix is diagonalized: ``\mathbf{C} = \mathbf{P}\mathbf{\Lambda}\mathbf{P}^{-1}``. Each eigenmode with eigenvalue ``\lambda_m`` evolves independently:

- If ``|\lambda_m| \approx 0``: **barotropic mode** — uses the Euler kernel
- Otherwise: ``L_d^{(\text{mode})} = 1/\sqrt{|\lambda_m|}`` — uses a QG kernel

The velocity in physical layers is recovered by projecting back through the eigenvector matrix.
