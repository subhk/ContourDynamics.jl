# API Reference: Surgery

The surgery API handles remeshing, reconnection, and filament removal. Most
workflows use `SurgeryParams` and `surgery!`; the lower-level functions support
custom surgery pipelines.

```@docs
surgery!
remesh
ContourDynamics.find_close_segments
ContourDynamics.build_spatial_index
ContourDynamics.reconnect!
ContourDynamics.remove_filaments!
```
