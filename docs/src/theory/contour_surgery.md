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

Reconnection uses a **spatial index** (hash-map binned by ``\delta``-sized grid) for ``O(N \log C)`` candidate detection, where ``N`` is the total node count and ``C`` is the number of contours.

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

Typical choices: ``\delta \approx \mu``, ``\Delta_{\max} \approx 10\text{–}40\mu``, ``A_{\min} \approx \delta^2``.
