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

RK4 is the package's built-in time integrator. After surgery, `evolve!`
synchronizes every RK4 work buffer to the current node count before advancing
the remeshed contour state.

## Notes

RK4 is a standard time-integration scheme rather than a
contour-dynamics-specific theoretical result. The package uses it with a fixed
timestep.

For the broader contour-dynamics literature referenced elsewhere in this
section, see [References](references.md).
