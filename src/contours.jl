function remesh(c::PVContour{T}, params::SurgeryParams{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c

    mu = params.mu
    Delta_max = params.Delta_max

    new_nodes = SVector{2, T}[]
    push!(new_nodes, nodes[1])

    for i in 2:n
        b = nodes[i]
        last = new_nodes[end]
        d = b - last
        seg_len = sqrt(d[1]^2 + d[2]^2)

        if seg_len < mu
            continue
        elseif seg_len > Delta_max
            n_segments = ceil(Int, seg_len / Delta_max)
            for k in 1:(n_segments - 1)
                t = T(k) / T(n_segments)
                push!(new_nodes, last + t * d)
            end
            push!(new_nodes, b)
        else
            push!(new_nodes, b)
        end
    end

    # Check closing segment (last node back to first, with wrap for spanning contours)
    if length(new_nodes) >= 2
        close_target = new_nodes[1] + c.wrap  # wrap = (0,0) for closed contours
        d_close = close_target - new_nodes[end]
        close_len = sqrt(d_close[1]^2 + d_close[2]^2)
        if close_len < mu && length(new_nodes) > 3
            pop!(new_nodes)
        elseif close_len > Delta_max
            last = new_nodes[end]
            n_segments = ceil(Int, close_len / Delta_max)
            for k in 1:(n_segments - 1)
                t = T(k) / T(n_segments)
                push!(new_nodes, last + t * d_close)
            end
        end
    end

    length(new_nodes) < 3 && return c
    return PVContour(new_nodes, c.pv, c.wrap)
end

function arc_lengths(c::PVContour{T}) where {T}
    n = nnodes(c)
    lengths = Vector{T}(undef, n)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        lengths[i] = sqrt(d[1]^2 + d[2]^2)
    end
    return lengths
end
