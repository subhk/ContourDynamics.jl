# API Reference: Devices

`CPU()` keeps contour state in ordinary Julia arrays and all simulation work
uses CPU implementations. `GPU()` keeps the active contour state in
device-resident buffers for supported velocity, timestepping, surgery, and
diagnostic paths. Host contour containers on a GPU problem are initialization
shadows; use `materialize_contours(prob)` only when you intentionally need a CPU
copy for output, plotting, file writing, or interactive inspection.

```@docs
CPU
GPU
DeviceContourState
materialize_contours
device_array
device_zeros
to_cpu
to_device
```
