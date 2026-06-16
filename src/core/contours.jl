# Contour geometry helpers and remeshing.
#
# This file contains the mechanics for measuring, interpolating, and
# redistributing nodes on a PV contour. Public users usually enter through
# `remesh`, `arc_lengths`, and `beta_staircase`; the private helpers below are
# grouped by the operation they support:
#
#   polygon moments      -> area/centroid preservation during remeshing
#   curvature estimates  -> Dritschel node-density construction
#   cubic interpolation  -> smooth segment reconstruction between nodes
#   weighted resampling  -> choose new node positions under μ/Δ_max bounds

function _raw_polygon_area(nodes::AbstractVector{SVector{2,T}},
                           wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    n = length(nodes)
    n < 3 && return zero(T)
    area = zero(T)
    @inbounds for i in 1:n
        nxt = i < n ? nodes[i + 1] : nodes[1] + wrap
        area += nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
    end
    return area / 2
end

function _raw_polygon_centroid(nodes::AbstractVector{SVector{2,T}},
                               wrap::SVector{2,T}=zero(SVector{2,T})) where {T}
    n = length(nodes)
    n == 0 && return zero(SVector{2,T})
    area = _raw_polygon_area(nodes, wrap)
    if abs(area) <= eps(T)
        return sum(nodes) / n
    end

    cx = zero(T)
    cy = zero(T)
    @inbounds for i in 1:n
        nxt = i < n ? nodes[i + 1] : nodes[1] + wrap
        cross = nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
        cx += (nodes[i][1] + nxt[1]) * cross
        cy += (nodes[i][2] + nxt[2]) * cross
    end
    inv6A = one(T) / (6 * area)
    return SVector{2,T}(cx * inv6A, cy * inv6A)
end

function _preserve_closed_area!(nodes::Vector{SVector{2,T}}, target_area::T) where {T}
    # Remeshing changes node locations slightly. For closed contours, apply a
    # uniform centroid-centered rescale so the signed polygon area is preserved.
    new_area = _raw_polygon_area(nodes)
    (abs(target_area) <= eps(T) || abs(new_area) <= eps(T)) && return nodes
    sign(target_area) == sign(new_area) || return nodes

    scale = sqrt(abs(target_area / new_area))
    abs(scale - one(T)) <= sqrt(eps(T)) && return nodes
    ctr = _raw_polygon_centroid(nodes)
    @inbounds for i in eachindex(nodes)
        nodes[i] = ctr + scale * (nodes[i] - ctr)
    end
    return nodes
end

@inline _cross2(a::SVector{2,T}, b::SVector{2,T}) where {T} = a[1] * b[2] - a[2] * b[1]
@inline _norm2(a::SVector{2,T}) where {T} = sqrt(a[1]^2 + a[2]^2)

# Smallest-magnitude real root of a t² + b t + c = 0, or `nothing` when there is
# no usable root. Uses the numerically stable form (compute the large root via
# the standard formula, the small one via c/q) so the near-zero root we want is
# not lost to cancellation when c is small.
@inline function _smallest_quadratic_root(a::T, b::T, c::T) where {T}
    if abs(a) <= eps(T)
        abs(b) <= eps(T) && return nothing
        return -c / b
    end
    disc = b * b - 4 * a * c
    disc < 0 && return nothing
    sd = sqrt(disc)
    q = -(b + (b >= 0 ? sd : -sd)) / 2
    r1 = q / a
    abs(q) <= eps(T) && return r1
    r2 = c / q
    return abs(r1) <= abs(r2) ? r1 : r2
end

function _preserve_closed_area_fixed_corners!(nodes::Vector{SVector{2,T}},
                                              corners::AbstractVector{Bool},
                                              target_area::T) where {T}
    # Area-preserving rescale for contours with fixed surgery corners. The
    # corner-free path scales every node about the centroid, but that moves the
    # corners; here the corners must stay exactly put (they are reconnection
    # break points). So only the free (non-corner) nodes move, along
    # d_i = (p_i - centroid). The signed area is a quadratic A0 + B·t + C·t² in
    # the scalar step `t`; solve for the root nearest zero and apply it. Corners
    # keep d_i = 0 and therefore do not move at all.
    n = length(nodes)
    n < 3 && return nodes
    (abs(target_area) <= eps(T)) && return nodes
    A0 = _raw_polygon_area(nodes)
    abs(A0) <= eps(T) && return nodes
    sign(target_area) == sign(A0) || return nodes

    rhs = target_area - A0
    abs(rhs) <= sqrt(eps(T)) * abs(target_area) && return nodes  # already conserved

    ctr = _raw_polygon_centroid(nodes)
    B = zero(T)
    C = zero(T)
    @inbounds for i in 1:n
        j = i < n ? i + 1 : 1
        pi = nodes[i]
        pj = nodes[j]
        di = corners[i] ? zero(SVector{2,T}) : pi - ctr
        dj = corners[j] ? zero(SVector{2,T}) : pj - ctr
        B += _cross2(pi, dj) + _cross2(di, pj)
        C += _cross2(di, dj)
    end
    B /= 2
    C /= 2

    t = _smallest_quadratic_root(C, B, -rhs)
    # Skip when the correction is ill-conditioned (no real root, or so large it
    # would distort the geometry); a small unmeasured drift is safer than that.
    (t === nothing || !isfinite(t) || abs(t) > T(1) / 2) && return nodes

    @inbounds for i in 1:n
        corners[i] && continue
        nodes[i] = nodes[i] + t * (nodes[i] - ctr)
    end
    return nodes
end

@inline function _signed_node_curvature(c::PVContour{T}, i::Int) where {T}
    n = nnodes(c)
    n < 3 && return zero(T)
    @boundscheck (1 <= i <= n || throw(BoundsError(c.nodes, i)))

    @inbounds begin
        prev_i = mod1(i - 1, n)
        next_i = mod1(i + 1, n)
        if c.corners[prev_i] || c.corners[i] || c.corners[next_i]
            return zero(T)
        end

        prev = i == 1 ? c.nodes[n] - c.wrap : c.nodes[i - 1]
        curr = c.nodes[i]
        nxt = next_node(c, i)
        a = curr - prev
        b = nxt - curr
        chord = nxt - prev
        denom = _norm2(a) * _norm2(b) * _norm2(chord)
        denom <= eps(T) && return zero(T)
        return T(2) * _cross2(a, b) / denom
    end
end

function _signed_node_curvatures!(κ::AbstractVector{T}, c::PVContour{T}) where {T}
    # Curvature is signed so cubic interpolation can bend to the correct side of
    # each segment. Surgery corners deliberately suppress curvature through the
    # adjacent nodes to keep fixed break points sharp.
    n = nnodes(c)
    length(κ) >= n || throw(DimensionMismatch("curvature buffer length ($(length(κ))) must be >= node count ($n)"))
    @inbounds for i in 1:n
        κ[i] = _signed_node_curvature(c, i)
    end
    return κ
end

function _signed_node_curvatures(c::PVContour{T}) where {T}
    κ = zeros(T, nnodes(c))
    return _signed_node_curvatures!(κ, c)
end

function _node_curvatures(c::PVContour{T}) where {T}
    κ = _signed_node_curvatures(c)
    @inbounds for i in eachindex(κ)
        κ[i] = abs(κ[i])
    end
    return κ
end

function _dritschel_segment_densities(c::PVContour{T}, params::SurgeryParams,
                                      sources=(c,)) where {T}
    n = nnodes(c)
    densities = Vector{T}(undef, n)
    n == 0 && return densities

    lengths = arc_lengths(c)
    perimeter = sum(lengths)
    if perimeter <= eps(T)
        fill!(densities, one(T) / T(params.Δ_max))
        return densities
    end

    α = T(2) / T(3)
    δ = T(params.δ)
    μ = T(params.μ)
    Δ_max = T(params.Δ_max)
    L = max(perimeter / T(2π), T(params.Δ_max))
    inv_μL = inv(μ * L)
    sqrt2 = sqrt(T(2))
    d2_floor = eps(T) * max(one(T), L)^2
    source_data = map(sources) do source
        (contour=source,
         lengths=arc_lengths(source),
         curvatures=_node_curvatures(source),
         abs_pv=abs(source.pv))
    end

    # Dritschel (1988), Eqs. (2a)-(2d).  First form the nonlocal
    # vorticity-weighted curvature K_j at each node of this contour, using all
    # source contours that participate in the same surgery pass.  The resulting
    # transformed node density is averaged onto segments and saturated so the
    # implied spacing cannot fall below δ/sqrt(2).
    node_density_curvatures = Vector{T}(undef, n)
    @inbounds for j in 1:n
        xj = c.nodes[j]
        numerator = zero(T)
        denominator = zero(T)

        for data in source_data
            source = data.contour
            ns = nnodes(source)
            ns == 0 && continue
            source_lengths = data.lengths
            source_curvatures = data.curvatures
            abs_pv = data.abs_pv

            for i in 1:ns
                ei = source_lengths[i]
                ei <= eps(T) && continue
                mid = (source.nodes[i] + next_node(source, i)) / T(2)
                d = xj - mid
                d2 = max(d[1]^2 + d[2]^2, d2_floor)
                weight = ei * abs_pv / d2
                denominator += weight
                numerator += weight * source_curvatures[i]
            end
        end

        K_j = denominator <= eps(T) ? zero(T) : numerator / denominator
        node_density_curvatures[j] = inv_μL * (K_j * L)^α + sqrt2 * K_j
    end

    raw = Vector{T}(undef, n)
    @inbounds for j in 1:n
        κ̃ = (node_density_curvatures[j] + node_density_curvatures[mod1(j + 1, n)]) / T(2)
        raw[j] = κ̃ <= eps(T) ? zero(T) : κ̃ / (one(T) + δ * κ̃ / sqrt2)
    end

    target_intervals = _target_interval_count(perimeter, n, μ, Δ_max)
    _scale_densities!(densities, raw, lengths, target_intervals, μ, Δ_max)
    return densities
end

function _signed_path_curvatures(path::Vector{SVector{2,T}}, corners::AbstractVector{Bool}) where {T}
    n = length(path)
    κ = zeros(T, n)
    n < 3 && return κ

    @inbounds for i in 2:(n - 1)
        if corners[i - 1] || corners[i] || corners[i + 1]
            continue
        end

        a = path[i] - path[i - 1]
        b = path[i + 1] - path[i]
        chord = path[i + 1] - path[i - 1]
        denom = _norm2(a) * _norm2(b) * _norm2(chord)
        denom <= eps(T) && continue
        κ[i] = T(2) * _cross2(a, b) / denom
    end
    return κ
end

function _cubic_segment_point(a::SVector{2,T}, b::SVector{2,T},
                              κa::T, κb::T, p::T) where {T}
    # Dritschel-style cubic arc: linear segment plus normal displacement whose
    # polynomial is determined by endpoint curvatures.
    t = b - a
    e = _norm2(t)
    e <= eps(T) && return a

    n = SVector{2,T}(-t[2], t[1])
    α = -e * (T(2) * κa + κb) / T(6)
    β = e * κa / T(2)
    γ = e * (κb - κa) / T(6)
    η = p * (α + p * (β + p * γ))
    return a + p * t + η * n
end

function _cubic_segment_tangent(a::SVector{2,T}, b::SVector{2,T},
                                κa::T, κb::T, p::T) where {T}
    t = b - a
    e = _norm2(t)
    e <= eps(T) && return zero(SVector{2,T})

    n = SVector{2,T}(-t[2], t[1])
    α = -e * (T(2) * κa + κb) / T(6)
    β = e * κa / T(2)
    γ = e * (κb - κa) / T(6)
    η′ = α + T(2) * β * p + T(3) * γ * p^2
    return t + η′ * n
end

function _cubic_segment_point(c::PVContour{T}, i::Int, p::T, κ::Vector{T}) where {T}
    return _cubic_segment_point(c.nodes[i], next_node(c, i), κ[i], κ[mod1(i + 1, nnodes(c))], p)
end

function _scale_densities!(densities::Vector{T}, raw::Vector{T}, lengths::AbstractVector{T},
                           target_intervals::Int, μ::T, Δ_max::T) where {T}
    # Normalize raw density so its length-weighted integral gives the desired
    # interval count, then clamp to enforce minimum/maximum segment lengths.
    weighted = zero(T)
    @inbounds for i in eachindex(raw, lengths)
        weighted += raw[i] * lengths[i]
    end

    min_density = one(T) / Δ_max
    max_density = one(T) / μ
    scale = weighted <= eps(T) ? min_density : T(target_intervals) / weighted
    @inbounds for i in eachindex(raw)
        densities[i] = clamp(raw[i] * scale, min_density, max_density)
    end
    return densities
end

function _target_interval_count(total_length::T, current::Int, μ::T, Δ_max::T) where {T}
    min_intervals = max(1, ceil(Int, total_length / Δ_max))
    max_intervals = max(min_intervals, floor(Int, total_length / μ))
    return clamp(current, min_intervals, max_intervals)
end

function _path_segment_lengths(path::Vector{SVector{2,T}}) where {T}
    lengths = Vector{T}(undef, max(0, length(path) - 1))
    @inbounds for i in eachindex(lengths)
        lengths[i] = _norm2(path[i + 1] - path[i])
    end
    return lengths
end

function _weighted_measure(lengths::AbstractVector{T}, densities::AbstractVector{T}) where {T}
    measure = Vector{T}(undef, length(lengths) + 1)
    measure[1] = zero(T)
    @inbounds for i in eachindex(lengths)
        measure[i + 1] = measure[i] + lengths[i] * densities[i]
    end
    return measure
end

function _point_at_weighted_measure(path::Vector{SVector{2,T}},
                                    curvatures::Vector{T},
                                    measure::Vector{T}, s::T) where {T}
    seg = searchsortedlast(measure, s, 1, length(measure), Base.Order.Forward)
    seg = clamp(seg, 1, length(path) - 1)
    seg_measure = measure[seg + 1] - measure[seg]
    seg_measure <= eps(T) && return path[seg]
    p = (s - measure[seg]) / seg_measure
    return _cubic_segment_point(path[seg], path[seg + 1], curvatures[seg], curvatures[seg + 1], p)
end

function _resample_fixed_corner_path(path::Vector{SVector{2,T}}, densities::Vector{T},
                                     μ::T, Δ_max::T) where {T}
    # Used after reconnection. The first and last nodes are surgery corners and
    # stay fixed; only interior points are redistributed by weighted measure.
    length(path) >= 2 || return copy(path), trues(length(path))

    lengths = _path_segment_lengths(path)
    total_length = sum(lengths)
    total_length <= eps(T) && return copy(path), trues(length(path))
    measure = _weighted_measure(lengths, densities)
    path_corners = falses(length(path))
    path_corners[begin] = true
    path_corners[end] = true
    curvatures = _signed_path_curvatures(path, path_corners)
    target_intervals = _target_interval_count(total_length, length(lengths), μ, Δ_max)
    q = measure[end]
    n_intervals = clamp(round(Int, q), ceil(Int, total_length / Δ_max),
                        max(ceil(Int, total_length / Δ_max), floor(Int, total_length / μ)))
    n_intervals = max(1, n_intervals, target_intervals)

    new_nodes = SVector{2,T}[path[1]]
    new_corners = Bool[true]

    for k in 1:(n_intervals - 1)
        s_target = q * T(k) / T(n_intervals)
        push!(new_nodes, _point_at_weighted_measure(path, curvatures, measure, s_target))
        push!(new_corners, false)
    end
    push!(new_nodes, path[end])
    push!(new_corners, true)
    return new_nodes, new_corners
end

function _resample_closed_weighted(c::PVContour{T}, densities::Vector{T}, μ::T, Δ_max::T;
                                   _buf::Union{Nothing, Vector{SVector{2,T}}}=nothing,
                                   _vnodes_buf::Union{Nothing, Vector{SVector{2,T}}}=nothing) where {T}
    # Treat closed/spanning contours as a periodic path by appending the wrapped
    # first node. Uniformly spaced weighted-measure targets then map back to
    # cubic points on that virtual path.
    nodes = c.nodes
    n = length(nodes)
    close_pt = nodes[1] + c.wrap

    vnodes = if _vnodes_buf !== nothing
        resize!(_vnodes_buf, n + 1)
        _vnodes_buf
    else
        Vector{SVector{2,T}}(undef, n + 1)
    end
    copyto!(vnodes, 1, nodes, 1, n)
    vnodes[n + 1] = close_pt

    lengths = _path_segment_lengths(vnodes)
    total_length = sum(lengths)
    total_length <= eps(T) && return copy(nodes)

    measure = _weighted_measure(lengths, densities)
    curvatures = _signed_node_curvatures(c)
    path_curvatures = Vector{T}(undef, n + 1)
    copyto!(path_curvatures, 1, curvatures, 1, n)
    path_curvatures[n + 1] = curvatures[1]
    q = measure[end]
    min_intervals = max(3, ceil(Int, total_length / Δ_max))
    max_intervals = max(min_intervals, floor(Int, total_length / μ))
    n_intervals = clamp(round(Int, q), min_intervals, max_intervals)

    new_nodes = if _buf !== nothing
        empty!(_buf)
        _buf
    else
        SVector{2,T}[]
    end
    sizehint!(new_nodes, n_intervals)
    for k in 0:(n_intervals - 1)
        s_target = q * T(k) / T(n_intervals)
        push!(new_nodes, _point_at_weighted_measure(vnodes, path_curvatures, measure, s_target))
    end
    return new_nodes
end

function _corner_span_nodes_and_densities(c::PVContour{T}, densities::Vector{T},
                                          first_idx::Int, last_idx::Int) where {T}
    nodes = c.nodes
    if first_idx < last_idx
        return nodes[first_idx:last_idx], densities[first_idx:(last_idx - 1)]
    elseif first_idx > last_idx
        return vcat(nodes[first_idx:end], nodes[1:last_idx]),
               vcat(densities[first_idx:end], densities[1:(last_idx - 1)])
    else
        return vcat(nodes[first_idx:end], nodes[1:first_idx]),
               vcat(densities[first_idx:end], densities[1:(first_idx - 1)])
    end
end

function _remesh_with_fixed_corners(c::PVContour{T}, params::SurgeryParams, sources) where {T}
    μ = T(params.μ)
    Δ_max = T(params.Δ_max)
    corner_idxs = corner_indices(c)
    isempty(corner_idxs) && return c
    densities = _dritschel_segment_densities(c, params, sources)

    new_nodes = SVector{2,T}[]
    new_corners = Bool[]
    for k in eachindex(corner_idxs)
        first_idx = corner_idxs[k]
        last_idx = corner_idxs[mod1(k + 1, length(corner_idxs))]
        path, span_densities = _corner_span_nodes_and_densities(c, densities, first_idx, last_idx)
        span_nodes, span_corners = _resample_fixed_corner_path(path, span_densities, μ, Δ_max)
        if k == firstindex(corner_idxs)
            append!(new_nodes, span_nodes)
            append!(new_corners, span_corners)
        else
            append!(new_nodes, @view span_nodes[2:end])
            append!(new_corners, @view span_corners[2:end])
        end
    end

    if length(new_nodes) > 1
        d = new_nodes[end] - new_nodes[1]
        if sqrt(d[1]^2 + d[2]^2) <= eps(T) * (one(T) + abs(new_nodes[1][1]) + abs(new_nodes[1][2]))
            pop!(new_nodes)
            pop!(new_corners)
        end
    end

    length(new_nodes) < 3 && return c

    # Redistributing interior nodes onto cubic arcs changes the enclosed area
    # even though the corners stay fixed. Restore the original polygon area by
    # moving only the free nodes (corners stay exactly put), so reconnection
    # cleanup does not accrue drift. (Non-spanning here, so `c.wrap` is zero and
    # the default-wrap area applies.)
    _preserve_closed_area_fixed_corners!(new_nodes, new_corners, _raw_polygon_area(c.nodes))

    return PVContour(new_nodes, c.pv, c.wrap, new_corners)
end

"""
    remesh(c::PVContour, params::SurgeryParams; _buf=nothing, _arc_buf=nothing, _vnodes_buf=nothing)

Redistribute nodes along contour `c` with a Dritschel-style density that
increases with curvature and vorticity-weighted nonlocal curvature influence,
placing new nodes on cubic interpolation arcs while keeping segment lengths
between `params.μ` and `params.Δ_max`. Closed contours preserve signed polygon
area after redistribution unless the contour has fixed surgery corners. Corner
nodes are held fixed and divide the contour into independently remeshed spans.
Returns a new [`PVContour`](@ref).

The optional `_buf`, `_arc_buf`, and `_vnodes_buf` keywords accept pre-allocated
vectors that are reused across calls to avoid repeated heap allocation (internal
optimisation used by [`surgery!`](@ref)).
"""
function remesh(c::PVContour{T}, params::SurgeryParams;
                _buf::Union{Nothing, Vector{SVector{2,T}}}=nothing,
                _arc_buf::Union{Nothing, Vector{T}}=nothing,
                _vnodes_buf::Union{Nothing, Vector{SVector{2,T}}}=nothing,
                _density_sources=nothing) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c
    sources = _density_sources === nothing ? (c,) : _density_sources
    if any(c.corners) && !is_spanning(c)
        return _remesh_with_fixed_corners(c, params, sources)
    end

    μ = T(params.μ)
    Δ_max = T(params.Δ_max)

    densities = _dritschel_segment_densities(c, params, sources)
    new_nodes = _resample_closed_weighted(c, densities, μ, Δ_max;
                                          _buf=_buf, _vnodes_buf=_vnodes_buf)

    if length(new_nodes) < 3
        return c
    end

    if !is_spanning(c)
        _preserve_closed_area!(new_nodes, _raw_polygon_area(nodes))
    end

    # `_buf` is scratch storage owned by the caller. Never hand it back as the
    # contour's live node vector, or later remesh calls will mutate existing
    # contours in place through shared storage.
    out_nodes = _buf === nothing ? new_nodes : copy(new_nodes)
    return PVContour(out_nodes, c.pv, c.wrap, falses(length(out_nodes)))
end

"""
    beta_staircase(beta, domain::PeriodicDomain, n_beta; T=Float64)

Create spanning contours that discretize the background PV gradient `βy`
into a PV staircase on a periodic domain.

Returns a `Vector{PVContour{T}}` of `n_beta` horizontal spanning contours, each
carrying PV jump `Δq = β * Δy` where `Δy = 2*Ly / n_beta`. Nodes run
left-to-right at the mid-step y-levels `-Ly + (k - 1/2)Δy`.

The `wrap` field is set to `(2*Lx, 0)` so the closing segment connects the
rightmost node back to the leftmost node shifted by one period.
"""
function beta_staircase(beta::T, domain::PeriodicDomain{T}, n_beta::Int;
                        nodes_per_contour::Int=64) where {T}
    nodes_per_contour >= 3 || throw(ArgumentError("nodes_per_contour must be >= 3, got $nodes_per_contour"))
    n_beta >= 2 || throw(ArgumentError("n_beta must be >= 2, got $n_beta"))
    Lx, Ly = domain.Lx, domain.Ly
    dy = 2 * Ly / n_beta
    dq = beta * dy  # PV jump per contour

    contours = PVContour{T}[]
    wrap = SVector{2,T}(2 * Lx, zero(T))

    for k in 1:n_beta
        y_k = -Ly + (T(k) - T(0.5)) * dy
        # Nodes evenly spaced from -Lx to +Lx (not including +Lx, which is the wrap image of -Lx)
        nodes = [SVector{2,T}(-Lx + (2 * Lx) * T(i - 1) / T(nodes_per_contour), y_k)
                 for i in 1:nodes_per_contour]
        push!(contours, PVContour(nodes, dq, wrap))
    end

    return contours
end

"""
    arc_lengths(c::PVContour)

Return a vector of segment lengths for each consecutive node pair in contour `c`.
"""
function arc_lengths(c::PVContour{T}) where {T}
    n = nnodes(c)
    n < 1 && return T[]
    lengths = Vector{T}(undef, n)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        lengths[i] = sqrt(d[1]^2 + d[2]^2)
    end
    return lengths
end
