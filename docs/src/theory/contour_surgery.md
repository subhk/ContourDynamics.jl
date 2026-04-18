# Contour Surgery

The Dritschel surgery algorithm (Dritschel, 1988) handles the topological changes that arise during long-time evolution.

## Node Redistribution (Remeshing)

After each surgery pass, nodes are redistributed along each contour to maintain segment lengths between ``\mu`` (minimum) and ``\Delta_{\max}`` (maximum):

1. Compute cumulative arc lengths along the contour
2. Walk the perimeter, placing new nodes at arc-length intervals
3. Short segments (``< \mu``) are merged; long segments (``> \Delta_{\max}``) are subdivided

## Reconnection

When two contour segments approach within distance ``\delta``:

- **Same contour**: the contour is **split** (pinched) into two daughter contours
- **Different contours with same PV**: the contours are **merged** (stitched together)

Reconnection uses a **spatial index** (a hash-map binned by a ``\delta``-sized
grid) to filter candidate segment pairs before exact distance checks. In
practice this gives near-linear candidate lookup for well-resolved contours,
while still handling long segments by sampling each segment into multiple bins.

## Filament Removal

After reconnection, contours with ``|A| < A_{\min}`` (where ``A`` is the signed area) are removed. Spanning contours (which encode the periodic domain topology) are always preserved.

## Surgery Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `delta` | ``\delta`` | Proximity threshold for detecting close segments |
| `mu` | ``\mu`` | Minimum segment length after remeshing |
| `Delta_max` | ``\Delta_{\max}`` | Maximum segment length after remeshing |
| `area_min` | ``A_{\min}`` | Minimum contour area; smaller contours are removed |
| `n_surgery` | — | Number of time steps between surgery passes |

Typical choices in this implementation are ``\delta \lesssim \mu/4``,
``\Delta_{\max} \approx 10\text{–}40\mu``, and ``A_{\min} \approx \delta^2``.
Choosing ``\delta`` too large relative to ``\mu`` increases the chance of
spurious reconnections and is warned about by the constructor for
[`SurgeryParams`](@ref).
