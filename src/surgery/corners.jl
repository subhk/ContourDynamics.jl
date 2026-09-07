# surgery/corners.jl — CPU surgery stage.

function _insert_stitch_node(nodes::Vector{SVector{2,T}}, seg_idx::Int, point::SVector{2,T}) where {T}
    # Insert immediately after seg_idx, which is the local representation of a
    # point lying on segment seg_idx -> seg_idx+1. If seg_idx is the closing
    # segment, append so cyclic ordering is preserved.
    n = length(nodes)
    new_nodes = Vector{SVector{2,T}}(undef, n + 1)

    if seg_idx == n
        copyto!(new_nodes, 1, nodes, 1, n)
        new_nodes[n + 1] = point
        return new_nodes, n + 1
    end

    copyto!(new_nodes, 1, nodes, 1, seg_idx)
    new_nodes[seg_idx + 1] = point
    copyto!(new_nodes, seg_idx + 2, nodes, seg_idx + 1, n - seg_idx)

    return new_nodes, seg_idx + 1
end

function _insert_corner_flag(corners::BitVector, seg_idx::Int, flag::Bool=false)
    # Mirror _insert_stitch_node for the corner BitVector. New stitch nodes are
    # usually inserted unlabelled and promoted to fixed corners only when the
    # reconnect operation has decided which topology it is creating.
    n = length(corners)
    new_corners = falses(n + 1)

    if seg_idx == n
        copyto!(new_corners, 1, corners, 1, n)
        new_corners[n + 1] = flag
        return new_corners
    end

    copyto!(new_corners, 1, corners, 1, seg_idx)
    new_corners[seg_idx + 1] = flag
    copyto!(new_corners, seg_idx + 2, corners, seg_idx + 1, n - seg_idx)

    return new_corners
end

@inline function _segment_has_corner(c::PVContour, i::Int)
    # A segment touching a fixed corner is temporarily protected from additional
    # surgery so a fresh stitch is remeshed before being considered again.
    n = nnodes(c)
    return n > 0 && (c.corners[i] || c.corners[mod1(i + 1, n)])
end

function _demote_obtuse_corners(c::PVContour{T}) where {T}
    # Dritschel corners are temporary labels. Once local geometry has opened into
    # an obtuse angle, the node can return to the ordinary remeshing population.
    any(c.corners) || return c

    corners = copy(c.corners)
    changed = false
    n = nnodes(c)
    n < 3 && return c

    @inbounds for i in 1:n
        corners[i] || continue
        prev = c.nodes[mod1(i - 1, n)]
        curr = c.nodes[i]
        nxt = next_node(c, i)

        v_prev = prev - curr
        v_next = nxt - curr

        l_prev = sqrt(v_prev[1]^2 + v_prev[2]^2)
        l_next = sqrt(v_next[1]^2 + v_next[2]^2)

        (l_prev <= eps(T) || l_next <= eps(T)) && continue

        if v_prev[1] * v_next[1] + v_prev[2] * v_next[2] < zero(T)
            corners[i] = false
            changed = true
        end
    end

    return changed ? PVContour(c.nodes, c.pv, c.wrap, corners) : c
end

function _promote_high_curvature_corners(c::PVContour{T}, δ) where {T}
    # If a bend is sharper than the surgical cutoff can represent and is acute,
    # label it as a corner so subsequent remeshing keeps it fixed instead of
    # smearing the cusp over neighbouring nodes.
    is_spanning(c) && return c
    n = nnodes(c)
    n < 3 && return c

    corners = copy(c.corners)
    κ = _signed_node_curvatures(c)
    threshold = inv(T(δ))
    changed = false

    @inbounds for i in 1:n
        corners[i] && continue
        prev_i = mod1(i - 1, n)
        next_i = mod1(i + 1, n)
        (corners[prev_i] || corners[next_i]) && continue
        abs(κ[i]) < threshold && continue

        prev = i == 1 ? c.nodes[n] - c.wrap : c.nodes[prev_i]
        curr = c.nodes[i]
        nxt = next_node(c, i)
        v1 = prev - curr
        v2 = nxt - curr
        acute = v1[1] * v2[1] + v1[2] * v2[2] > zero(T)
        if acute
            corners[i] = true
            changed = true
        end
    end

    return changed ? PVContour(c.nodes, c.pv, c.wrap, corners) : c
end

function _promote_high_curvature_corners!(contours::Vector{PVContour{T}}, δ) where {T}
    for i in eachindex(contours)
        contours[i] = _promote_high_curvature_corners(contours[i], δ)
    end
    return contours
end
