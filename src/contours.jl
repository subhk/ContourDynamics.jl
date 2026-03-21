"""
    remesh(c::PVContour, params::SurgeryParams)

Redistribute nodes along contour `c` so that every segment length lies between
`params.mu` and `params.Delta_max`.  Returns a new [`PVContour`](@ref).
"""
function remesh(c::PVContour{T}, params::SurgeryParams) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c

    mu = T(params.mu)
    Delta_max = T(params.Delta_max)

    # Phase 1: compute cumulative arc lengths along the full contour perimeter,
    # including the closing segment (nodes[n] → nodes[1] + wrap).
    # We store n+1 entries: arc[1..n] for the original nodes, and arc[n+1] for
    # the virtual closing point (= nodes[1] + wrap at the total perimeter length).
    close_pt = nodes[1] + c.wrap  # closing target
    arc = Vector{T}(undef, n + 1)
    arc[1] = zero(T)
    for i in 2:n
        d = nodes[i] - nodes[i-1]
        arc[i] = arc[i-1] + sqrt(d[1]^2 + d[2]^2)
    end
    d_close = close_pt - nodes[n]
    arc[n + 1] = arc[n] + sqrt(d_close[1]^2 + d_close[2]^2)

    # Build a virtual node array that includes the closing point so that
    # interpolation in the subdivision step can reference the closing segment.
    # vnodes[1..n] = nodes[1..n], vnodes[n+1] = close_pt.
    vnodes = Vector{SVector{2,T}}(undef, n + 1)
    copyto!(vnodes, 1, nodes, 1, n)
    vnodes[n + 1] = close_pt
    nv = n + 1  # number of virtual nodes (= segments count for searchsortedlast)

    # Phase 2: walk the full perimeter placing nodes at arc-length positions.
    # The loop now covers nodes 2..n AND the closing point (index n+1),
    # so the closing segment is redistributed by the same logic as interior segments.
    # Pre-size: output typically has similar node count as input
    new_nodes = SVector{2, T}[]
    sizehint!(new_nodes, n + div(n, 4))  # allow for some subdivision
    push!(new_nodes, nodes[1])
    target_s = zero(T)  # arc length of the last emitted node

    for i in 2:nv
        gap = arc[i] - target_s
        if gap < mu
            # Too close to last emitted node — skip.
            # For the closing point (i == nv), skipping means the closing segment
            # is short enough; no extra node needed.
            continue
        elseif gap > Delta_max
            # Subdivide: place nodes at equal arc-length intervals from target_s.
            # For the last sub-node (k == n_segments), skip emitting it when
            # i == nv because that point is the closing image of nodes[1],
            # which is not stored as an explicit node.
            n_segments = ceil(Int, gap / Delta_max)
            k_end = (i == nv) ? n_segments - 1 : n_segments
            for k in 1:k_end
                s_target = target_s + gap * T(k) / T(n_segments)
                # Find virtual segment containing s_target
                seg = searchsortedlast(arc, s_target, 1, nv, Base.Order.Forward)
                seg = clamp(seg, 1, nv - 1)
                seg_len = arc[seg + 1] - arc[seg]
                if seg_len > eps(T)
                    t = (s_target - arc[seg]) / seg_len
                    push!(new_nodes, vnodes[seg] + t * (vnodes[seg + 1] - vnodes[seg]))
                else
                    push!(new_nodes, vnodes[seg])
                end
            end
            target_s = arc[i]
        else
            # Emit the node directly — but not the closing point itself,
            # since it is the periodic image of nodes[1].
            if i < nv
                push!(new_nodes, vnodes[i])
            end
            target_s = arc[i]
        end
    end

    # Final check: if the closing segment (new last node → new_nodes[1] + wrap)
    # is still too short after redistribution, remove the last node.
    # Guard: never drop below 3 nodes (minimum for a valid contour).
    if length(new_nodes) > 3
        close_target = new_nodes[1] + c.wrap
        d_final = close_target - new_nodes[end]
        final_len = sqrt(d_final[1]^2 + d_final[2]^2)
        if final_len < mu
            pop!(new_nodes)
        end
    end

    length(new_nodes) < 3 && return c
    return PVContour(new_nodes, c.pv, c.wrap)
end

"""
    beta_staircase(beta, domain::PeriodicDomain, n_steps; T=Float64)

Create spanning contours that discretize the background PV gradient `βy`
into a PV staircase on a periodic domain.

Returns a `Vector{PVContour{T}}` of horizontal spanning contours, each carrying
PV jump `Δq = β * Δy` where `Δy = 2*Ly / n_steps`.  Nodes run left-to-right
at evenly spaced y-levels from `-Ly + Δy` to `Ly - Δy` (excluding boundaries).

The `wrap` field is set to `(2*Lx, 0)` so the closing segment connects the
rightmost node back to the leftmost node shifted by one period.
"""
function beta_staircase(beta::T, domain::PeriodicDomain{T}, n_steps::Int;
                        nodes_per_contour::Int=64) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    dy = 2 * Ly / n_steps
    dq = beta * dy  # PV jump per contour

    contours = PVContour{T}[]
    wrap = SVector{2,T}(2 * Lx, zero(T))

    for k in 1:(n_steps - 1)
        y_k = -Ly + k * dy
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
    lengths = Vector{T}(undef, n)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        lengths[i] = sqrt(d[1]^2 + d[2]^2)
    end
    return lengths
end
