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
| `Ld` | Positive deformation radius for QG/beta-plane QG, or `N-1` modal radii for an `N`-layer problem |
| `beta` | Planetary-PV gradient; required for beta-plane QG |
| `delta_sqg` | Positive SQG regularization length; required for SQG and independent of the surgery `delta` |
| `coupling` | Symmetric layer-stretching matrix; required for multi-layer QG and checked against `Ld` |
| `domain` | `:unbounded` (default) or `:periodic` |
| `Lx`, `Ly` | Positive periodic-domain half-widths; required for `domain=:periodic` |
| `stepper` | `:RK4` (default) or `:leapfrog` |
| `ra_coeff` | Robert--Asselin coefficient for leapfrog; default `0.05` |
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
LeapfrogStepper
```
