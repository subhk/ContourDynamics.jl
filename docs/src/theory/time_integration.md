# Time Integration

## RK4

The classical 4th-order Runge-Kutta scheme advances all node positions simultaneously:

```math
\mathbf{x}^{n+1} = \mathbf{x}^n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
```

Here:

- ``\mathbf{x}^n`` is the vector of all contour-node positions at time step ``n``
- ``\Delta t`` is the time step
- ``\mathbf{k}_1, \mathbf{k}_2, \mathbf{k}_3, \mathbf{k}_4`` are velocity evaluations at the usual RK4 stages
- ``\mathbf{x}^{n+1}`` is the updated node position after one time step

Writing ``\mathbf{u}(\mathbf{x})`` for the velocity of the complete contour
state, the stage velocities are

```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{u}(\mathbf{x}^n),\\
\mathbf{k}_2 &= \mathbf{u}(\mathbf{x}^n + \tfrac{\Delta t}{2}\mathbf{k}_1),\\
\mathbf{k}_3 &= \mathbf{u}(\mathbf{x}^n + \tfrac{\Delta t}{2}\mathbf{k}_2),\\
\mathbf{k}_4 &= \mathbf{u}(\mathbf{x}^n + \Delta t\,\mathbf{k}_3).
\end{aligned}
```

Thus each ``\mathbf{k}_r`` has velocity units; the subscript
``r\in\{1,2,3,4\}`` identifies the RK4 stage rather than a contour or node.

This is the recommended integrator for most applications.

## Leapfrog with Robert-Asselin Filter

The leapfrog scheme is a 2nd-order centred three-level method:

```math
\mathbf{x}^{n+1} = \mathbf{x}^{n-1} + 2\,\Delta t \, \mathbf{u}(\mathbf{x}^n) .
```

where ``\mathbf{x}^{n-1}``, ``\mathbf{x}^n``, ``\mathbf{x}^{n+1}`` are the
node positions at three consecutive time levels and
``\mathbf{u}(\mathbf{x}^n)`` is the velocity evaluated at the current level.

Because the recurrence couples only even and odd time indices, it admits a
spurious **computational mode** that flips sign every step and is not damped
by the physics. The Robert–Asselin filter suppresses it by replacing the
middle level ``\mathbf{x}^n`` with a lightly smoothed value **before** it is
used as ``\mathbf{x}^{n-1}`` on the next step:

```math
\tilde{\mathbf{x}}^n = \mathbf{x}^n
  + \tfrac{\nu}{2}\left(\mathbf{x}^{n+1} - 2\mathbf{x}^n + \mathbf{x}^{n-1}\right) .
```

Here ``\nu \in [0, 1]`` is the filter coefficient, controlled by the
`ra_coeff` keyword of [`Problem`](@ref) / [`LeapfrogStepper`](@ref)
(default ``\nu = 0.05``). The bracket is the discrete second time
difference, so the filter is a small amount of numerical diffusion in
time:

The tilde distinguishes the filtered middle level
``\tilde{\mathbf{x}}^n`` from its unfiltered value ``\mathbf{x}^n``; it does
not denote a new physical time level.

- ``\nu = 0`` — pure leapfrog, computational mode undamped.
- ``\nu > 0`` — the computational mode decays by a factor
  ``1 - \nu`` per step, while the physical mode is only perturbed at
  ``\mathcal{O}(\nu\,\Delta t^2)``; the scheme remains 2nd-order accurate
  for ``\nu \lesssim 0.1``.
- Typical range: ``\nu \in [0.01, 0.1]``; the default ``0.05`` is a
  standard choice in geophysical modelling.

Per step, the stepper therefore:

1. forms the unfiltered update ``\mathbf{x}^{n+1} = \mathbf{x}^{n-1}
   + 2\Delta t\,\mathbf{u}(\mathbf{x}^n)``,
2. overwrites the middle level with
   ``\tilde{\mathbf{x}}^n = \mathbf{x}^n + \tfrac{\nu}{2}(\mathbf{x}^{n+1}
   - 2\mathbf{x}^n + \mathbf{x}^{n-1})``,
3. shifts the buffers (``\tilde{\mathbf{x}}^n`` becomes the new
   ``\mathbf{x}^{n-1}``; ``\mathbf{x}^{n+1}`` becomes the new
   ``\mathbf{x}^n``) and proceeds.

The first step is bootstrapped with a 2nd-order midpoint (RK2) method,
since leapfrog needs two past levels before the main recurrence can start.

After surgery, `evolve!` synchronizes every stepper work buffer to the current
node count. Leapfrog additionally discards its old history and repeats the RK2
bootstrap because remeshing changes node correspondence even when the total
node count is unchanged.

## Notes

These are standard time-integration schemes rather than contour-dynamics-specific
theoretical results. The package uses fixed-step RK4 as the default method and
offers leapfrog with a Robert-Asselin filter when a two-level scheme is useful.

For the broader contour-dynamics literature referenced elsewhere in this
section, see [References](references.md).
