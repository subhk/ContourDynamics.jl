# API Reference: Surgery

The surgery API handles remeshing, reconnection, and filament removal. Most
workflows use `SurgeryParams` and `surgery!`; the lower-level functions support
custom surgery pipelines.

The distinction between surgery `δ`, SQG `δ_sqg`, and all remeshing
symbols is summarized in the [notation and parameter
glossary](../theory/notation.md).

```@docs
surgery!
remesh
ContourDynamics.find_close_segments
ContourDynamics.build_spatial_index
ContourDynamics.reconnect!
ContourDynamics.remove_filaments!
```
