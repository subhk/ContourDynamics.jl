"""
    remesh(c::PVContour, params::SurgeryParams)

Redistribute nodes along contour `c` so that every segment length lies between
`params.mu` and `params.Delta_max`.  Returns a new [`PVContour`](@ref).
"""
function remesh(c::PVContour{T}, params::SurgeryParams{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return c

    mu = params.mu
    Delta_max = params.Delta_max

    # Phase 1: compute cumulative arc lengths along original contour
    arc = Vector{T}(undef, n)
    arc[1] = zero(T)
    for i in 2:n
        d = nodes[i] - nodes[i-1]
        arc[i] = arc[i-1] + sqrt(d[1]^2 + d[2]^2)
    end

    # Phase 2: walk the contour placing nodes at arc-length positions.
    # When consecutive original nodes are too close (< mu), we advance
    # along the original polyline to the correct arc-length position
    # rather than simply skipping the node — this preserves curvature.
    new_nodes = SVector{2, T}[]
    push!(new_nodes, nodes[1])
    target_s = zero(T)  # arc length of the last emitted node

    for i in 2:n
        gap = arc[i] - target_s
        if gap < mu
            # Too close — don't emit, but don't shortcut either;
            # the next emitted node will be interpolated at the right spot.
            continue
        elseif gap > Delta_max
            # Subdivide: place nodes at equal arc-length intervals from target_s
            n_segments = ceil(Int, gap / Delta_max)
            for k in 1:n_segments
                s_target = target_s + gap * T(k) / T(n_segments)
                # Find original segment containing s_target
                seg = searchsortedlast(arc, s_target, 1, n, Base.Order.Forward)
                seg = clamp(seg, 1, n - 1)
                seg_len = arc[seg + 1] - arc[seg]
                if seg_len > eps(T)
                    t = (s_target - arc[seg]) / seg_len
                    push!(new_nodes, nodes[seg] + t * (nodes[seg + 1] - nodes[seg]))
                else
                    push!(new_nodes, nodes[seg])
                end
            end
            target_s = arc[i]
        else
            push!(new_nodes, nodes[i])
            target_s = arc[i]
        end
    end

    # Check closing segment (last node back to first, with wrap for spanning contours)
    if length(new_nodes) >= 2
        close_target = new_nodes[1] + c.wrap
        d_close = close_target - new_nodes[end]
        close_len = sqrt(d_close[1]^2 + d_close[2]^2)
        if close_len < mu && length(new_nodes) > 3
            pop!(new_nodes)
        elseif close_len > Delta_max
            last_node = new_nodes[end]
            n_segments = ceil(Int, close_len / Delta_max)
            for k in 1:(n_segments - 1)
                t = T(k) / T(n_segments)
                push!(new_nodes, last_node + t * d_close)
            end
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
