# remeshing/resampling.jl — geometry and remeshing stage helpers.

function _path_segment_lengths(path::Vector{SVector{2,T}}) where {T}
    lengths = Vector{T}(undef, max(0, length(path) - 1))
    return _path_segment_lengths!(lengths, path)
end

function _path_segment_lengths!(lengths::Vector{T},
                                path::Vector{SVector{2,T}}) where {T}
    resize!(lengths, max(0, length(path) - 1))
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
                                     μ::T, Δ_max::T;
                                     _arc_buf::Union{Nothing,Vector{T}}=nothing) where {T}
    # Used after reconnection. The first and last nodes are surgery corners and
    # stay fixed; only interior points are redistributed by weighted measure.
    length(path) >= 2 || return copy(path), trues(length(path))

    lengths = _arc_buf === nothing ? _path_segment_lengths(path) :
              _path_segment_lengths!(_arc_buf, path)
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
                                   _vnodes_buf::Union{Nothing, Vector{SVector{2,T}}}=nothing,
                                   _arc_buf::Union{Nothing, Vector{T}}=nothing) where {T}
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

    lengths = _arc_buf === nothing ? _path_segment_lengths(vnodes) :
              _path_segment_lengths!(_arc_buf, vnodes)
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
