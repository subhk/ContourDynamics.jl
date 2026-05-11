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
