function remesh(c::PVContour{T}, params::SurgeryParams{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c

    mu = params.mu
    Delta_max = params.Delta_max

    new_nodes = SVector{2, T}[]
    push!(new_nodes, nodes[1])

    for i in 1:n
        j = mod1(i + 1, n)
        a = nodes[i]
        b = nodes[j]
        d = b - a
        seg_len = sqrt(d[1]^2 + d[2]^2)

        if seg_len < mu && length(new_nodes) > 1 && i < n
            continue
        elseif seg_len > Delta_max
            n_segments = ceil(Int, seg_len / Delta_max)
            for k in 1:(n_segments - 1)
                t = T(k) / T(n_segments)
                push!(new_nodes, a + t * d)
            end
            if j != 1
                push!(new_nodes, b)
            end
        else
            if j != 1
                push!(new_nodes, b)
            end
        end
    end

    length(new_nodes) < 3 && return c
    return PVContour(new_nodes, c.pv)
end

function arc_lengths(c::PVContour{T}) where {T}
    n = nnodes(c)
    lengths = Vector{T}(undef, n)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        d = c.nodes[j] - c.nodes[i]
        lengths[i] = sqrt(d[1]^2 + d[2]^2)
    end
    return lengths
end
