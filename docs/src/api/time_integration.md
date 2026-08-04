# API Reference: Time Integration

`timestep!` is the strict low-level operation and expects correctly sized
stepper work buffers. `evolve!` is topology-aware: it synchronizes every RK4 or
leapfrog work array with the current node count before stepping and after
surgery. Leapfrog history is invalidated after every remesh and re-bootstrapped
with RK2. See [Time Integration](../theory/time_integration.md) for equations
and the [notation glossary](../theory/notation.md) for every symbol.

```@docs
timestep!
evolve!
resize_buffers!
```
