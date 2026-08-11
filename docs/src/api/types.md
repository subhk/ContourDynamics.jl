# API Reference: Types

## Kernels

```@docs
AbstractKernel
EulerKernel
QGKernel
BetaPlaneQGKernel
SQGKernel
MultiLayerQGKernel
```

## Contours and Domains

```@docs
PVContour
AbstractDomain
UnboundedDomain
PeriodicDomain
```

## Problem Structs

```@docs
ContourProblem
MultiLayerContourProblem
SurgeryParams
Problem
```

The high-level `Problem(; T=...)` constructor keeps matching-precision contour
vectors by identity. If contour or layer inputs use a different floating-point
type, it copies and converts their nodes, PV, wrap, and corner data to `T` before
constructing the contour problem and time-stepper.

### `Problem` keyword reference

| Keyword | Accepted value and meaning |
|---|---|
| `contours` | Single-layer contour vector; required except for multi-layer QG |
| `layers` | Tuple of per-layer contour vectors; required for `kernel=:multilayer_qg` |
| `dt` | Positive fixed timestep; always required |
| `kernel` | `:euler` (default), `:qg`, `:beta_plane_qg`, `:sqg`, or `:multilayer_qg` |
| `Ld` | Finite positive deformation radius for QG/beta-plane QG, or `N-1` modal radii for an `N`-layer problem |
| `beta` | Planetary-PV gradient; required for beta-plane QG |
| `δ_sqg` | Positive SQG regularization length; required for SQG and independent of the surgery `δ` |
| `coupling` | Physical layer-stretching matrix; required for multi-layer QG, required to annihilate the uniform barotropic mode, thickness-symmetrized when needed, and checked against `Ld` |
| `layer_thicknesses` | Positive layer depths/weights for multi-layer QG; optional when `coupling` is symmetric or uniquely determines their ratios |
| `domain` | `:unbounded` (default) or `:periodic` |
| `Lx`, `Ly` | Positive periodic-domain half-widths; required for `domain=:periodic` |
| `stepper` | `:RK4` (the only built-in method; `:leapfrog` was removed in v1.0.21 and now throws with migration guidance) |
| `surgery` | `:standard`, `:conservative`, `:aggressive`, `:none`, or `SurgeryParams` |
| `dev` | `CPU()` (default) or `GPU()` |
| `T` | Floating-point type for converted inputs and internal buffers; default `Float64` |

`contours` and `layers` are mutually exclusive. The mathematical symbols and
their corresponding keywords are collected in the [notation and parameter
glossary](../theory/notation.md).

## Accessors

```@docs
contours
kernel
domain
```

## Time Steppers

```@docs
AbstractTimeStepper
RK4Stepper
```

### Removed steppers

`LeapfrogStepper` (and the `stepper=:leapfrog` / `ra_coeff` keywords) were
removed in v1.0.21. The exported stub below remains only so v1.0.x code fails
with migration guidance instead of an `UndefVarError`; use `RK4Stepper`.

```@docs
LeapfrogStepper
```
