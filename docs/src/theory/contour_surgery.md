# Contour Surgery

The Dritschel surgery algorithm (Dritschel, 1988) handles the topological changes that arise during long-time evolution.

In plain terms, contour surgery is the cleanup step that keeps a contour
simulation usable after the flow stretches and folds the boundaries:

- thin filaments can become too small to resolve accurately
- nearby contour segments may need to reconnect during merger or pinch-off
- node spacing drifts away from a useful resolution unless it is corrected

The package applies surgery as a separate geometric operation between time
steps. That keeps the velocity calculation and the topology-changing logic
conceptually separate.

## Node Redistribution (Remeshing)

After each surgery pass, nodes are redistributed along each contour to maintain segment lengths between ``\mu`` (minimum) and ``\Delta_{\max}`` (maximum):

1. Compute cumulative arc lengths along the contour
2. Walk the perimeter, placing new nodes at arc-length intervals
3. Short segments (``< \mu``) are merged; long segments (``> \Delta_{\max}``) are subdivided

Here:

- ``\mu`` is the minimum desired segment length
- ``\Delta_{\max}`` is the maximum desired segment length

The goal is not to change the large-scale shape of the contour. The goal is to
keep the discretization well behaved, so no part of the contour becomes much
too crowded or much too sparse.

## Reconnection

When two contour segments approach within distance ``\delta``:

- **Same contour**: the contour is **split** (pinched) into two daughter contours
- **Different contours with same PV**: the contours are **merged** (stitched together)

Here ``\delta`` is the proximity threshold used to decide that two segments are
close enough to be considered for reconnection.

Reconnection uses a **spatial index** (a hash-map binned by a ``\delta``-sized
grid) to filter candidate segment pairs before exact distance checks. In
practice this gives near-linear candidate lookup for well-resolved contours,
while still handling long segments by sampling each segment into multiple bins.

This step is what lets the method represent genuine topological events such as:

- two vortices merging into one
- one contour pinching into two separate pieces
- filament bridges being removed once they are no longer resolved

## Filament Removal

After reconnection, contours with ``|A| < A_{\min}`` (where ``A`` is the signed area) are removed. Spanning contours (which encode the periodic domain topology) are always preserved.

Here:

- ``A`` is the signed area enclosed by a contour
- ``A_{\min}`` is the smallest area the simulation keeps

This removes very small filaments and debris that are below the intended
resolution of the contour description.

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

In practice:

- smaller ``\delta`` makes reconnection more conservative
- smaller ``\mu`` increases geometric resolution, but also increases node count
- larger ``\Delta_{\max}`` allows coarser spacing in smooth regions
- larger ``n_surgery`` applies surgery less often

If you are unsure, start from the built-in `:standard` preset and only tune the
parameters once you know whether the issue is under-resolution, over-aggressive
reconnection, or too-frequent cleanup.

## References and Further Reading

- Dritschel, D.G. (1988). *Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)

For related background, see [References](/theory/references).
