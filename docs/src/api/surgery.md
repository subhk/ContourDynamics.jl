# API Reference: Surgery

The surgery API handles remeshing, reconnection, and filament removal. Most
users only need `SurgeryParams` and `surgery!`; the lower-level functions are
mainly useful if you want to customize the surgery pipeline.

```@docs
surgery!
remesh
ContourDynamics.find_close_segments
ContourDynamics.build_spatial_index
ContourDynamics.reconnect!
ContourDynamics.remove_filaments!
```
