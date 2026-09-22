# Contour Surgery

The implementation uses a Dritschel-style contour-surgery loop for the
topological changes that arise during long-time evolution. Contours are
redistributed on cubic Dritschel interpolation arcs; the reconnection test uses
Dritschel's node-to-straight-segment contact rule.

Contour surgery is the geometric cleanup step applied after stretching and
folding produce under-resolved features:

- thin filaments can become too small to resolve accurately
- nearby contour segments may need to reconnect during merger or pinch-off
- node spacing can become inconsistent unless it is redistributed

The package applies surgery as a separate geometric operation between time
steps. That keeps the velocity calculation and the topology-changing logic
conceptually separate.

This follows the standard topological part of Dritschel's contour-surgery
algorithm, specifically the merge/split loop summarized in Dritschel (1988),
Table III:

1. keep node spacing controlled by redistributing nodes,
2. reconnect only close contour segments enclosing the same interior vorticity level,
3. introduce labelled corner nodes at merge/split locations,
4. repeat reconnection until no admissible close pairs remain,
5. redistribute again after reconnection while keeping corner nodes fixed,
6. remove unresolved filament debris.

The surgery geometry uses cubic Dritschel interpolation for redistribution,
repeated compatible-interior-vorticity segment reconnection, explicit labelled
corner nodes, high-curvature corner promotion, curvature-sensitive nonlocal
redistribution with spacing bounds, curved-segment velocity quadrature for the
single-layer Euler, QG, and SQG kernels in direct and KA velocity paths, and an
area/perimeter cutoff for unresolved filaments.

## Algorithm Summary

At a high level, a surgery pass is:

```text
remove already-subgrid debris
demote obtuse labelled corners
promote under-resolved high-curvature bends to labelled corners
remesh all contours
build a spatial index for non-spanning contour segments
find admissible close segment pairs within δ
while close pairs remain:
    select independent pairs, closest first
    split same-contour pairs and merge compatible different-contour pairs
    remesh with labelled corners fixed
    demote corners that have become obtuse
    remove unresolved filament debris
    rebuild the spatial index and close-pair list
```

Surgery itself ends there. In `evolve!`, periodic wrapping runs *before* each
surgery pass (right after the timestep), and the stepper's work buffers are
synchronized with the final node count after surgery returns.

This is the same conceptual structure as Dritschel's contour-surgery loop, but
expressed in the terms used by this package. The velocity solver never performs
topology changes. It only advances the existing nodes; surgery then acts as a
geometric cleanup and topology update between time steps.

## Implementation Map

The main implementation points are:

| Surgery step | Code path |
|-----------------------|-----------|
| Cubic contour representation and curvature suppression at corners | `_signed_node_curvatures`, `_cubic_segment_point`, `curved_segment_velocity` |
| Adapted nonlocal node redistribution with fixed corner spans | `_dritschel_segment_densities`, `remesh`, `_remesh_with_fixed_corners` |
| Search for new high-curvature acute corners | `_promote_high_curvature_corners!` |
| Find close node-to-segment contacts within ``\delta`` | `build_spatial_index`, `find_close_segments`, `_surgery_contact_distance2` |
| Split one contour or merge compatible contours | `reconnect!`, `_reconnect_split!`, `_reconnect_merge!` |
| Repeat reconnection until no admissible pairs remain | `surgery!` reconnection loop |
| Demote corners once they become obtuse | `_demote_obtuse_corners!` after remeshing |
| Remove unresolved debris | `remove_filaments!` |

The multi-layer surgery path applies the same reference algorithm independently
within each layer; contours never reconnect across layers. For `GPU()` problems,
the device-side path mirrors the cleanup, admissibility, independent-pair
selection, topology rewrite, and remeshing predicates on flat device arrays.
It supports single-layer Euler, QG, and SQG in unbounded and periodic domains,
periodic beta-plane QG, and multi-layer QG in unbounded and periodic domains.

## Core Equations

For a closed contour with nodes ``\mathbf{x}_j`` and segment lengths
``\ell_j = |\mathbf{x}_{j+1}-\mathbf{x}_j|``, the local signed curvature is
estimated from adjacent chords:

```math
\kappa_j
= \frac{2\,(\mathbf{a}_j \times \mathbf{b}_j)}
        {|\mathbf{a}_j|\,|\mathbf{b}_j|\,|\mathbf{x}_{j+1}-\mathbf{x}_{j-1}|},
\qquad
\mathbf{a}_j = \mathbf{x}_j-\mathbf{x}_{j-1}, \quad
\mathbf{b}_j = \mathbf{x}_{j+1}-\mathbf{x}_j .
```

Curvature is set to zero at labelled surgery corners and their immediate
neighbours. This keeps sharp surgery junctions fixed during redistribution.
Here ``j`` is a cyclic node index, ``\times`` is the signed scalar 2D cross
product, ``\mathbf a_j`` and ``\mathbf b_j`` are the incoming and outgoing
chords, and ``\kappa_j`` is positive for a local left turn.

Dritschel's nonlocal node-density idea is represented by a curvature scale
``K_j`` at node ``j``. In this implementation, nearby high-curvature segments
from all contours participating in the same surgery pass contribute with
vorticity and inverse-square distance weights:

```math
K_j
=
\frac{
  \sum_i \ell_i\,|\Delta q_i|\,|\kappa_i|\,d_{ij}^{-2}
}{
  \sum_i \ell_i\,|\Delta q_i|\,d_{ij}^{-2}
},
\qquad
d_{ij} = |\mathbf{x}_j-\mathbf{m}_i|,
```

where ``\mathbf{m}_i`` is the midpoint of source segment ``i`` and
``\Delta q_i`` is its PV jump. A small distance floor is used in the code to
avoid division by zero.
The sum runs over source segments ``i`` from all contours in the same surgery
pass; ``\ell_i`` and ``\kappa_i`` are their length and signed curvature, while
``d_{ij}`` is the distance from target node ``j`` to source midpoint ``i``.
Absolute values make the density depend on curvature and PV-jump magnitudes.

The curvature scale is transformed at each node, then averaged over the two
endpoints of segment ``j`` before saturation near the surgery cutoff ``\delta``:

```math
\begin{aligned}
\mu_{\mathrm d} &= \frac{\mu}{L}, \\
\tilde{\kappa}_j
&= \frac{1}{\mu_{\mathrm d} L}(K_j L)^{2/3} + \sqrt{2}\,K_j
 = \frac{1}{\mu}(K_j L)^{2/3} + \sqrt{2}\,K_j, \\
\bar{\kappa}_j &= \frac{\tilde{\kappa}_j+\tilde{\kappa}_{j+1}}{2}, \\
\tilde{\rho}_j
&= \frac{\bar{\kappa}_j}{1 + \delta \bar{\kappa}_j/\sqrt{2}}.
\end{aligned}
```

Here ``L=\max(L_c/(2\pi),\Delta_{\max})`` is estimated from the contour
perimeter, ``\mu`` is the public minimum target segment **length**, and
``\delta`` is the surgery cutoff. The dimensionless density parameter
``\mu_{\mathrm d}=\mu/L`` is a package choice that adapts the literature's
density construction to this length-based API. The literature's dimensionless
parameter must not be substituted directly for `SurgeryParams.μ`.
Both terms of the transformed curvature ``\tilde\kappa_j`` have inverse-length
units. It combines nonlocal curvature ``K_j`` with the large-scale length ``L``.
The segment value ``\bar\kappa_j`` averages the transformed endpoint values
before the nonlinear saturation. The raw segment density ``\tilde\rho_j``
is then rescaled and spacing-clamped to give ``\rho_j``, so the target spacing
stays in the interval

```math
\mu \le \Delta s_j \le \Delta_{\max}.
```

Here ``\Delta s_j`` is the target spacing and ``\Delta_{\max}`` is the
maximum allowed target length. These bounds constrain the density and interval
counts. Cubic interpolation and subsequent area correction can change the final
chord lengths. Scaling all coordinate lengths and length-valued parameters
together scales the density inversely, preserving the relative node distribution.

The implementation forms a discrete weighted measure from **chord lengths**.
For a closed contour with ``N`` input nodes, its polygonal perimeter is
``L_c=\sum_{j=1}^N\ell_j`` and its cumulative measure is

```math
M_1=0,\qquad M_{j+1}=M_j+\ell_j\rho_j,
\qquad
m_k=\frac{k}{N_{\mathrm{seg}}}M_{N+1},
\quad k=0,\ldots,N_{\mathrm{seg}}-1.
```

Here ``\rho_j`` is constant on input segment ``j``, ``N_{\mathrm{seg}}`` is
the chosen number of output segments, and ``m_k`` is the target measure for
output node ``k``. On the segment satisfying ``M_j\le m_k<M_{j+1}``, the
interpolation parameter is

```math
p_k=\frac{m_k-M_j}{\ell_j\rho_j}.
```

The node is evaluated at ``\mathbf X_j(p_k)`` on the cubic arc below. This
procedure approximates density-weighted redistribution using chord lengths;
it does not integrate the cubic arclength element ``|\mathbf X'_j(p)|\,dp``.
Labelled corners divide the contour into spans to which the same construction
is applied with fixed endpoints. Subsequent area correction can move the free
nodes. The interpolation arc is also used for curved-segment velocity
quadrature:

```math
\mathbf{X}(p)
=
\mathbf{a}
+ p(\mathbf{b}-\mathbf{a})
+ \eta(p)\,\mathbf{n},
\qquad 0 \le p \le 1,
```

with

```math
\eta(p)
=
p\left[
-\frac{e(2\kappa_a+\kappa_b)}{6}
+ p\left(
\frac{e\kappa_a}{2}
+ p\frac{e(\kappa_b-\kappa_a)}{6}
\right)
\right],
```

where ``e=|\mathbf{b}-\mathbf{a}|`` and
``\mathbf{n}=(-(b_y-a_y),b_x-a_x)`` is the chord rotated to the left, with
``|\mathbf n|=e``. Thus ``\eta`` is dimensionless and the signed displacement
along the unit normal is ``e\eta``. This is the cubic representation in
Crowdy & Surana (2007), Eqs. (3.9)--(3.11), following Dritschel.
If both endpoint curvatures are numerically zero, the segment reduces
to the straight-line formula.
Here ``\mathbf a`` and ``\mathbf b`` are chord endpoints, ``p`` is the local
coordinate from ``\mathbf a`` to ``\mathbf b``, ``\mathbf X(p)`` is the cubic
arc position, ``\eta(p)`` is its dimensionless normal coefficient, and
``\kappa_a,\kappa_b`` are endpoint curvatures.

For reconnection, two segment parts are considered close when the node-to-segment
contact distance satisfies

```math
d_{\mathrm{contact}}^2 < \delta^2 .
```

Here ``d_{\mathrm{contact}}`` is the minimum accepted node-to-opposing-segment
distance for the candidate pair and ``\delta`` is the surgery proximity
threshold.

Same-contour contacts are split into two daughter contours. Different-contour
contacts are merged only when the PV jumps and the locally enclosed interior
vorticity level agree. This is stricter than comparing PV jumps alone and avoids
incorrect reconnection between different levels of a nested vortex.

Finally, unresolved debris is removed when it is too small to represent at the
chosen resolution. The main area test is

```math
|A| < A_{\min},
```

Here ``A`` is signed enclosed contour area, ``|A|`` is its magnitude, and
``A_{\min}`` is the minimum retained area (`area_min`).

Additional cleanup removes corner-labelled fragments whose effective width is
below the remeshing scale. Spanning contours used for periodic PV staircases
are excluded from this cleanup.

## Node Redistribution (Remeshing)

After each surgery pass, nodes are redistributed along each contour using an
adaptation of Dritschel's nonlocal node-density construction. The density is
larger where the contour has larger curvature and is also increased by nearby
high-curvature parts of all contours participating in the same surgery pass,
weighted by vorticity jump and inverse squared distance.

Since the public surgery parameters intentionally stay compact, the
implementation fixes the curvature exponent at ``2/3``, uses
``\delta`` as the cutoff scale, estimates the large-scale length from the
contour perimeter, derives ``\mu_{\mathrm d}=\mu/L``, and rescales and clamps
the resulting density to the target spacing interval
``[\mu,\Delta_{\max}]``. The parameter mapping and node-budget normalization
are package adaptations; this is not an exact implementation of the paper's
node-density parameterization.

1. Estimate node curvature by the circle through each triplet of adjacent nodes
2. Set curvature to zero at labelled corners and their immediate neighbours
3. Form Dritschel's nonlocal curvature ``K_j`` from all same-pass source contours
4. Average adjacent transformed node curvatures, then saturate the segment
   density to limit the implied spacing near the cutoff scale
5. Rescale the density to keep the existing node budget unless spacing bounds require adding or removing nodes
6. Redistribute nodes by equal increments of the chord-weighted measure,
   mapping each target to a cubic Dritschel interpolation arc
7. Keep labelled surgery corners fixed and remesh only the spans between them

Here:

- ``\mu`` is the minimum desired segment length
- ``\Delta_{\max}`` is the maximum desired segment length

The goal is to preserve the large-scale shape while keeping the discretization
well behaved, so no part of the contour becomes excessively crowded or sparse.

## Curved Segment Velocity

For velocity evaluation, each contour segment is evaluated on the same cubic
Dritschel arc used by redistribution. With endpoint curvatures
``\kappa_a`` and ``\kappa_b``, the segment is parameterized as
``\mathbf X(p)=\mathbf a+p(\mathbf b-\mathbf a)+\eta(p)\mathbf n`` for
``0\le p\le1``. The symbols have the definitions given under Core Equations.

For Euler, the unbounded contribution uses Gauss-Legendre quadrature of the
contour integral
``-\frac{1}{4\pi}\int_0^1 \log|\mathbf x-\mathbf X(p)|^2
\mathbf X'(p)\,dp`` and falls back to the
analytic straight-segment antiderivative when both endpoint curvatures are
zero. QG uses the same curved Euler singular part plus quadrature of the smooth
``K_0(r/L_d) + \log r`` correction. SQG integrates the regularized
``1/\sqrt{r^2 + \epsilon^2}`` kernel along the curved tangent.

Here ``\mathbf x`` is the target position, ``\mathbf X'(p)=d\mathbf X/dp`` is
the tangent derivative, ``r=|\mathbf x-\mathbf X(p)|``, ``L_d`` is QG
deformation radius, ``K_0`` is the modified Bessel function, and ``\epsilon``
is the SQG regularization length (`δ_sqg`). The integration variable ``p``
runs once along the source arc.

On periodic domains, the same curved unbounded singular contribution is
combined with the existing smooth Fourier/Ewald periodic correction, evaluated
on the curved segment. This keeps the direct and KA velocity paths consistent
for Euler, QG, and SQG while preserving the analytic straight-segment path when
the endpoint curvatures are numerically zero.

## Reconnection

When two contour segments approach within distance ``\delta``:

- **Same contour**: the contour is **split** (pinched) into two daughter contours
- **Different contours enclosing the same interior vorticity level**: the contours are **merged** (stitched together)

Here ``\delta`` is the proximity threshold used to decide that two segments are
close enough to be considered for reconnection.

Reconnection uses a **spatial index** (a hash-map binned by a
``\max(\delta, \mu)``-sized grid — ``\mu``-sized for every built-in preset,
since presets keep ``\delta \le \mu/4``) to filter candidate segment pairs
before exact checks; the contact test itself uses ``\delta``. The acceptance
test follows Dritschel's surgery condition: a node on one contour segment must
lie within ``\delta`` of the straight segment on the other contour part. In
practice the spatial index gives near-linear candidate lookup for well-resolved
contours, while still handling long segments by sampling each segment into
multiple bins.

When an under-resolved acute bend has curvature larger than approximately
``1/\delta``, it is promoted to a labelled corner before redistribution. Corners
are demoted again once their local angle becomes obtuse, matching the corner
lifecycle described by Dritschel.

This step lets the method represent topological events such as:

- two vortices merging into one
- one contour pinching into two separate pieces
- filament bridges being removed once they are no longer resolved

## Filament Removal

After reconnection, labelled corner contours with four or fewer nodes are
removed as unresolved surgery debris, matching Dritschel's cleanup rule for
contours with too few nodes. Contours with ``|A| < A_{\min}`` (where ``A`` is
the signed area) are also removed, as are extremely thin corner filaments whose
effective width is below the remeshing scale. Spanning contours (which encode
the periodic domain topology) are always preserved.

Here:

- ``A`` is the signed area enclosed by a contour
- ``A_{\min}`` is the smallest area the simulation keeps

This removes very small filaments and debris that are below the intended
resolution of the contour description.

## Surgery Parameters

| Field (ASCII alias) | Symbol | Description |
|-----------|--------|-------------|
| `δ` (`delta`) | ``\delta`` | Proximity threshold for detecting close segments |
| `μ` (`mu`) | ``\mu`` | Minimum target segment length for remeshing |
| `Δ_max` (`Delta_max`) | ``\Delta_{\max}`` | Maximum target segment length for remeshing |
| `area_min` | ``A_{\min}`` | Minimum contour area; smaller contours are removed |
| `n_surgery` | — | Number of time steps between surgery passes |

The built-in presets use ``\delta \le \mu/4``,
``\Delta_{\max} \approx 7.5\text{–}15\,\mu``, and ``A_{\min}`` between roughly
``\delta^2/25`` and ``\delta^2``.
Choosing ``\delta`` too large relative to ``\mu`` increases the chance of
spurious reconnections and is warned about by the constructor for
[`SurgeryParams`](@ref).

In practice:

- smaller ``\delta`` makes reconnection more conservative
- smaller ``\mu`` increases geometric resolution, but also increases node count
- larger ``\Delta_{\max}`` allows coarser spacing in smooth regions
- larger ``n_surgery`` applies surgery less often

The built-in `:standard` preset is the recommended starting point. Tuning is
usually only needed to address under-resolution, over-aggressive reconnection,
or overly frequent cleanup.

## References and Further Reading

- Crowdy, D. & Surana, A. (2007). *Contour dynamics in complex domains.* J. Fluid Mech. **593**, 235--254. [doi:10.1017/S002211200700866X](https://doi.org/10.1017/S002211200700866X)
- Dritschel, D.G. (1988). *Contour surgery: a topological reconnection scheme for extended integrations using contour dynamics.* J. Comput. Phys. **77**(1), 240--266. [doi:10.1016/0021-9991(88)90165-9](https://doi.org/10.1016/0021-9991(88)90165-9)
- Dritschel, D.G. (1989). *Contour dynamics and contour surgery: numerical algorithms for extended, high-resolution modelling of vortex dynamics in two-dimensional, inviscid, incompressible flows.* Comput. Phys. Rep. **10**(3), 77--146. [doi:10.1016/0167-7977(89)90004-X](https://doi.org/10.1016/0167-7977(89)90004-X)

For related background, see [References](references.md).
