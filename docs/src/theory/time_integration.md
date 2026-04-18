# Time Integration

## RK4

The classical 4th-order Runge-Kutta scheme advances all node positions simultaneously:

```math
\mathbf{x}^{n+1} = \mathbf{x}^n + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
```

This is the recommended integrator for most applications.

## Leapfrog with Robert-Asselin Filter

The leapfrog scheme is 2nd-order centred:

```math
\mathbf{x}^{n+1} = \mathbf{x}^{n-1} + 2\Delta t \, \mathbf{u}(\mathbf{x}^n)
```

A **Robert-Asselin filter** (``\nu = 0.05`` by default) damps the computational mode. The first step is bootstrapped with a 2nd-order midpoint (RK2) method.
