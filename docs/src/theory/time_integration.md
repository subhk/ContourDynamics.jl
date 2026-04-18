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

This is the recommended integrator for most applications.

## Leapfrog with Robert-Asselin Filter

The leapfrog scheme is 2nd-order centred:

```math
\mathbf{x}^{n+1} = \mathbf{x}^{n-1} + 2\Delta t \, \mathbf{u}(\mathbf{x}^n)
```

Here:

- ``\mathbf{x}^{n-1}``, ``\mathbf{x}^n``, and ``\mathbf{x}^{n+1}`` are the node positions at three consecutive time levels
- ``\mathbf{u}(\mathbf{x}^n)`` is the velocity evaluated at the current position
- ``\nu`` is the Robert-Asselin filter coefficient

A **Robert-Asselin filter** (``\nu = 0.05`` by default) damps the computational
mode. The first step is bootstrapped with a 2nd-order midpoint (RK2) method,
because leapfrog needs data from two time levels before the main recurrence can
start.

## Notes

These are standard time-integration schemes rather than contour-dynamics-specific
theoretical results. The package uses fixed-step RK4 as the default method and
offers leapfrog with a Robert-Asselin filter when a two-level scheme is useful.

For the broader contour-dynamics literature referenced elsewhere in this
section, see [References](/theory/references).
