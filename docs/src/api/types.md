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
