# API Reference: Time Integration

`timestep!` is the strict low-level operation and expects correctly sized
stepper work buffer. `evolve!` is topology-aware: it synchronizes every RK4
work array with the current node count before stepping and after surgery. See
[Time Integration](../theory/time_integration.md) for equations and the
[notation glossary](../theory/notation.md) for every symbol.

```@docs
timestep!
evolve!
resize_buffers!
```
