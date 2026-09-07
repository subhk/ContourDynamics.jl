# remeshing/remesh.jl — geometry and remeshing stage helpers.

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

function _remesh_with_fixed_corners(c::PVContour{T}, params::SurgeryParams, sources;
                                    _arc_buf::Union{Nothing,Vector{T}}=nothing,
                                    _source_data=nothing) where {T}
    μ = T(params.μ)
    Δ_max = T(params.Δ_max)
    corner_idxs = corner_indices(c)
    isempty(corner_idxs) && return c
    densities = _dritschel_segment_densities(
        c, params, sources; _arc_buf=_arc_buf, _source_data=_source_data)

    new_nodes = SVector{2,T}[]
    new_corners = Bool[]
    for k in eachindex(corner_idxs)
        first_idx = corner_idxs[k]
        last_idx = corner_idxs[mod1(k + 1, length(corner_idxs))]
        path, span_densities = _corner_span_nodes_and_densities(c, densities, first_idx, last_idx)
        span_nodes, span_corners = _resample_fixed_corner_path(
            path, span_densities, μ, Δ_max; _arc_buf=_arc_buf)
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
                _density_sources=nothing,
                _density_source_data=nothing) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c
    sources = _density_sources === nothing ? (c,) : _density_sources
    if any(c.corners) && !is_spanning(c)
        return _remesh_with_fixed_corners(
            c, params, sources; _arc_buf=_arc_buf,
            _source_data=_density_source_data)
    end

    μ = T(params.μ)
    Δ_max = T(params.Δ_max)

    densities = _dritschel_segment_densities(
        c, params, sources; _arc_buf=_arc_buf,
        _source_data=_density_source_data)
    new_nodes = _resample_closed_weighted(c, densities, μ, Δ_max;
                                          _buf=_buf, _vnodes_buf=_vnodes_buf,
                                          _arc_buf=_arc_buf)

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
