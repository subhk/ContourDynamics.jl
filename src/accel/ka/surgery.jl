# Device-side data layout and kernels for Dritschel-style topology surgery.
#
# The CPU implementation remains the reference contour model. The helpers in
# this file mirror its Table III surgery rules on flat device arrays: cleanup,
# admissible node-to-segment contact tests, independent split/merge selection,
# topology materialization, and Dritschel weighted remeshing. The public GPU
# dispatch currently applies this backend to unbounded single-layer problems.

struct FlatContourTopology{T<:AbstractFloat,
                           FA<:AbstractVector{T},
                           IA<:AbstractVector{Int},
                           BA<:AbstractVector{UInt8}}
    # Flat node coordinates; contour membership is recovered through
    # `offsets`, `lengths`, `contour_of_node`, and `local_index`.
    x::FA
    y::FA
    # Per-contour metadata. `wrapx/wrapy` preserve spanning contour topology.
    pv::FA
    wrapx::FA
    wrapy::FA
    offsets::IA
    lengths::IA
    contour_of_node::IA
    local_index::IA
    corners::BA
    active::BA
end

struct DeviceReconnectionPlan{T<:AbstractFloat,
                              IA<:AbstractVector{Int},
                              FA<:AbstractVector{T},
                              BA<:AbstractVector{UInt8}}
    # Candidate contact is segment i on contour ci against segment j on contour cj.
    ci::IA
    i::IA
    cj::IA
    j::IA
    distance2::FA
    op::BA        # 1 = split, 2 = merge
    selected::BA  # 1 = selected by independent-pair planner
end

struct DeviceClosePairCandidates{IA<:AbstractVector{Int}}
    ci::IA
    i::IA
    cj::IA
    j::IA
end

struct DeviceTopologyRewritePlan{T<:AbstractFloat,
                                 IA<:AbstractVector{Int},
                                 FA<:AbstractVector{T},
                                 BA<:AbstractVector{UInt8}}
    ci::IA
    i::IA
    cj::IA
    j::IA
    op::BA                # 1 = split, 2 = merge
    valid::BA             # 1 = rewrite can be applied, 0 = leave unchanged
    node_from_first::BA   # best stitch node came from ci/i segment
    node_idx::IA          # local index of the chosen stitch node
    seg_idx::IA           # local segment index receiving the inserted stitch node
    inserted_idx::IA      # local index of the inserted stitch node after insertion
    split_reverse1::BA
    split_reverse2::BA
    merge_reverse_second::BA
    stitch_x::FA
    stitch_y::FA
    out_count::IA         # number of output contours for this operation
    out_len1::IA
    out_len2::IA
end

struct DeviceRewriteOutputs{T<:AbstractFloat,
                            FA<:AbstractVector{T},
                            IA<:AbstractVector{Int},
                            BA<:AbstractVector{UInt8}}
    x::FA
    y::FA
    pv::FA
    wrapx::FA
    wrapy::FA
    offsets::IA
    lengths::IA
    corners::BA
end

_flat_ncontours(flat::FlatContourTopology) = length(flat.lengths)
_flat_nnodes(flat::FlatContourTopology) = length(flat.x)

@kernel function _fill_u8_kernel!(out, value, n)
    i = @index(Global)
    i <= n && (out[i] = value)
end

function _device_u8_filled(dev::AbstractDevice, n::Int, value::UInt8)
    out = device_zeros(dev, UInt8, n)
    n > 0 && @_ka_launch dev n _fill_u8_kernel!(out, value, n)
    return out
end

function _flat_topology(state::DeviceContourState{T},
                        dev::AbstractDevice=CPU()) where {T}
    return FlatContourTopology(state.x, state.y, state.pv, state.wrapx,
                               state.wrapy, state.offsets, state.lengths,
                               state.contour_of_node, state.local_index,
                               state.corners,
                               _device_u8_filled(dev, length(state.lengths), UInt8(1)))
end

function _pack_flat_topology(contours::Vector{PVContour{T}}, dev::AbstractDevice=CPU()) where {T}
    # Convert ragged contour vectors into dense buffers. Device kernels cannot
    # follow Julia Vector{Vector}-style topology, so all per-node and per-contour
    # data is flattened with offset/length lookup tables.
    ncontours = length(contours)
    total = sum(nnodes, contours; init=0)
    x = Vector{T}(undef, total)
    y = Vector{T}(undef, total)
    pv = Vector{T}(undef, ncontours)
    wrapx = Vector{T}(undef, ncontours)
    wrapy = Vector{T}(undef, ncontours)
    offsets = Vector{Int}(undef, ncontours)
    lengths = Vector{Int}(undef, ncontours)
    contour_of_node = Vector{Int}(undef, total)
    local_index = Vector{Int}(undef, total)
    corners = Vector{UInt8}(undef, total)
    active = fill(UInt8(1), ncontours)

    cursor = 1
    @inbounds for (ci, c) in pairs(contours)
        offsets[ci] = cursor
        nc = nnodes(c)
        lengths[ci] = nc
        pv[ci] = c.pv
        wrapx[ci] = c.wrap[1]
        wrapy[ci] = c.wrap[2]
        for li in 1:nc
            node = c.nodes[li]
            x[cursor] = node[1]
            y[cursor] = node[2]
            contour_of_node[cursor] = ci
            local_index[cursor] = li
            corners[cursor] = c.corners[li] ? UInt8(1) : UInt8(0)
            cursor += 1
        end
    end

    return FlatContourTopology(to_device(dev, x),
                               to_device(dev, y),
                               to_device(dev, pv),
                               to_device(dev, wrapx),
                               to_device(dev, wrapy),
                               to_device(dev, offsets),
                               to_device(dev, lengths),
                               to_device(dev, contour_of_node),
                               to_device(dev, local_index),
                               to_device(dev, corners),
                               to_device(dev, active))
end

@inline function _flat_segment_end(flat::FlatContourTopology{T}, ci::Int, li::Int) where {T}
    off = flat.offsets[ci]
    nc = flat.lengths[ci]
    g = off + li - 1
    if li < nc
        return flat.x[g + 1], flat.y[g + 1]
    else
        return flat.x[off] + flat.wrapx[ci], flat.y[off] + flat.wrapy[ci]
    end
end

@inline function _flat_spanning(flat::FlatContourTopology{T}, ci::Int) where {T}
    return !iszero(flat.wrapx[ci]) || !iszero(flat.wrapy[ci])
end

@kernel function _mark_filament_contours_kernel!(remove, x, y, wrapx, wrapy, offsets,
                                                 lengths, corners, area_min, μ,
                                                 ncontours)
    ci = @index(Global)
    if ci <= ncontours
        nc = lengths[ci]
        drop = false
        if !iszero(wrapx[ci]) || !iszero(wrapy[ci])
            drop = false
        elseif nc < 3
            drop = true
        else
            off = offsets[ci]
            area2 = zero(area_min)
            perimeter = zero(area_min)
            has_corner = false
            @inbounds for li in 1:nc
                g = off + li - 1
                if li < nc
                    nx = x[g + 1]
                    ny = y[g + 1]
                else
                    nx = x[off] + wrapx[ci]
                    ny = y[off] + wrapy[ci]
                end
                area2 += x[g] * ny - nx * y[g]
                dx = nx - x[g]
                dy = ny - y[g]
                perimeter += sqrt(dx * dx + dy * dy)
                has_corner |= !iszero(corners[g])
            end

            area = abs(area2) / 2
            min_perimeter = μ > zero(μ) ? 4 * μ : zero(μ)
            drop = area < area_min || perimeter < min_perimeter
            if !drop && has_corner
                if nc <= 4
                    drop = true
                elseif μ > zero(μ)
                    width = 2 * area / perimeter
                    drop = area <= μ * μ || width < μ
                end
            end
        end
        remove[ci] = drop ? UInt8(1) : UInt8(0)
    end
end

function _device_filament_flags_buffer(flat::FlatContourTopology{T},
                                       params::SurgeryParams,
                                       dev::AbstractDevice=CPU()) where {T}
    ncontours = _flat_ncontours(flat)
    remove = device_zeros(dev, UInt8, ncontours)
    ncontours == 0 && return remove
    @_ka_launch dev ncontours _mark_filament_contours_kernel!(
        remove, flat.x, flat.y, flat.wrapx, flat.wrapy, flat.offsets,
        flat.lengths, flat.corners, T(params.area_min), T(params.μ), ncontours)
    return remove
end

function _device_filament_flags(contours::Vector{PVContour{T}}, params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    flat = _pack_flat_topology(contours, dev)
    remove = _device_filament_flags_buffer(flat, params, dev)
    length(remove) == 0 && return Bool[]
    return map(!iszero, to_cpu(remove))
end

function _device_remove_filaments!(contours::Vector{PVContour{T}},
                                   params::SurgeryParams,
                                   dev::AbstractDevice=CPU()) where {T}
    isempty(contours) && return contours
    flags = _device_filament_flags(contours, params, dev)
    keep = trues(length(contours))
    @inbounds for i in eachindex(flags)
        keep[i] = !flags[i]
    end
    write = 1
    @inbounds for read in eachindex(contours)
        if keep[read]
            contours[write] = contours[read]
            write += 1
        end
    end
    resize!(contours, write - 1)
    return contours
end

@kernel function _invert_remove_flags_kernel!(keep, remove, n)
    ci = @index(Global)
    ci <= n && (keep[ci] = iszero(remove[ci]) ? UInt8(1) : UInt8(0))
end

@kernel function _compact_kept_contour_metadata_kernel!(out_lengths, out_pv,
                                                        out_wrapx, out_wrapy,
                                                        source_contour,
                                                        keep_slots, keep,
                                                        in_lengths, in_pv,
                                                        in_wrapx, in_wrapy,
                                                        ncontours)
    ci = @index(Global)
    if ci <= ncontours && !iszero(keep[ci])
        slot = keep_slots[ci]
        out_lengths[slot] = in_lengths[ci]
        out_pv[slot] = in_pv[ci]
        out_wrapx[slot] = in_wrapx[ci]
        out_wrapy[slot] = in_wrapy[ci]
        source_contour[slot] = ci
    end
end

@kernel function _compact_kept_state_nodes_kernel!(out_x, out_y, out_corners,
                                                   out_node_contour,
                                                   out_offsets,
                                                   source_contour,
                                                   in_x, in_y, in_corners,
                                                   in_offsets,
                                                   total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        out_ci = out_node_contour[g]
        src_ci = source_contour[out_ci]
        local_idx = g - out_offsets[out_ci] + 1
        in_g = in_offsets[src_ci] + local_idx - 1
        out_x[g] = in_x[in_g]
        out_y[g] = in_y[in_g]
        out_corners[g] = in_corners[in_g]
    end
end

function _device_compact_kept_contours_outputs(flat::FlatContourTopology{T},
                                               keep,
                                               dev::AbstractDevice=CPU()) where {T}
    ncontours = _flat_ncontours(flat)
    keep_slots = device_zeros(dev, Int, ncontours)
    count_store = device_zeros(dev, Int, 1)
    if ncontours > 0
        @_ka_launch dev ncontours _prefix_u8_kernel!(
            keep_slots, count_store, keep, ncontours)
    end
    nout = ncontours == 0 ? 0 : to_cpu(count_store)[1]

    out_lengths = device_zeros(dev, Int, nout)
    out_offsets = device_zeros(dev, Int, nout)
    out_pv = device_zeros(dev, T, nout)
    out_wrapx = device_zeros(dev, T, nout)
    out_wrapy = device_zeros(dev, T, nout)
    source_contour = device_zeros(dev, Int, nout)
    if nout > 0
        @_ka_launch dev ncontours _compact_kept_contour_metadata_kernel!(
            out_lengths, out_pv, out_wrapx, out_wrapy, source_contour,
            keep_slots, keep, flat.lengths, flat.pv, flat.wrapx, flat.wrapy,
            ncontours)
    end

    total_store = device_zeros(dev, Int, 1)
    if nout > 0
        @_ka_launch dev nout _prefix_lengths_kernel!(
            out_offsets, total_store, out_lengths, nout)
    end
    total_out_nodes = nout == 0 ? 0 : to_cpu(total_store)[1]

    out_node_contour = device_zeros(dev, Int, total_out_nodes)
    if nout > 0 && total_out_nodes > 0
        @_ka_launch dev nout _out_node_contour_kernel!(
            out_node_contour, out_offsets, out_lengths, nout)
    end

    out_x = device_zeros(dev, T, total_out_nodes)
    out_y = device_zeros(dev, T, total_out_nodes)
    out_corners = device_zeros(dev, UInt8, total_out_nodes)
    if total_out_nodes > 0
        @_ka_launch dev total_out_nodes _compact_kept_state_nodes_kernel!(
            out_x, out_y, out_corners, out_node_contour, out_offsets,
            source_contour, flat.x, flat.y, flat.corners, flat.offsets,
            total_out_nodes)
    end

    return DeviceRewriteOutputs(out_x, out_y, out_pv, out_wrapx, out_wrapy,
                                out_offsets, out_lengths, out_corners)
end

function _device_remove_filaments!(state::DeviceContourState{T},
                                   params::SurgeryParams,
                                   dev::AbstractDevice=CPU()) where {T}
    flat = _flat_topology(state, dev)
    ncontours = _flat_ncontours(flat)
    ncontours == 0 && return state
    remove = _device_filament_flags_buffer(flat, params, dev)
    keep = device_zeros(dev, UInt8, ncontours)
    @_ka_launch dev ncontours _invert_remove_flags_kernel!(keep, remove, ncontours)
    outputs = _device_compact_kept_contours_outputs(flat, keep, dev)
    return _replace_device_state!(state, outputs, dev)
end

@inline function _flat_point_segment_dist2(px, py, ax, ay, bx, by)
    sx = bx - ax
    sy = by - ay
    len2 = sx * sx + sy * sy
    if len2 <= eps(len2)
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    end
    t = ((px - ax) * sx + (py - ay) * sy) / len2
    t = min(max(t, zero(t)), one(t))
    cx = ax + t * sx
    cy = ay + t * sy
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy
end

@inline function _flat_point_segment_closest(px, py, ax, ay, bx, by)
    sx = bx - ax
    sy = by - ay
    len2 = sx * sx + sy * sy
    if len2 <= eps(len2)
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy, ax, ay
    end
    t = ((px - ax) * sx + (py - ay) * sy) / len2
    t = min(max(t, zero(t)), one(t))
    cx = ax + t * sx
    cy = ay + t * sy
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy, cx, cy
end

@inline function _flat_surgery_contact_distance2(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2)
    d1 = _flat_point_segment_dist2(ax1, ay1, ax2, ay2, bx2, by2)
    d2 = _flat_point_segment_dist2(bx1, by1, ax2, ay2, bx2, by2)
    d3 = _flat_point_segment_dist2(ax2, ay2, ax1, ay1, bx1, by1)
    d4 = _flat_point_segment_dist2(bx2, by2, ax1, ay1, bx1, by1)
    return min(min(d1, d2), min(d3, d4))
end

@kernel function _close_pair_candidate_kernel!(valid, x, y, pv, wrapx, wrapy, offsets,
                                               lengths, contour_of_node, local_index,
                                               corners, δ2, total_nodes)
    pair_idx = @index(Global)
    npairs = total_nodes * total_nodes
    if pair_idx <= npairs
        is_valid = false
        g1 = ((pair_idx - 1) % total_nodes) + 1
        g2 = ((pair_idx - 1) ÷ total_nodes) + 1
        if g1 < g2
            ci = contour_of_node[g1]
            cj = contour_of_node[g2]
            spanning = !iszero(wrapx[ci]) || !iszero(wrapy[ci]) ||
                       !iszero(wrapx[cj]) || !iszero(wrapy[cj])
            if !spanning
                li = local_index[g1]
                lj = local_index[g2]
                g1_next = li < lengths[ci] ? g1 + 1 : offsets[ci]
                g2_next = lj < lengths[cj] ? g2 + 1 : offsets[cj]
                has_corner = !iszero(corners[g1]) || !iszero(corners[g1_next]) ||
                             !iszero(corners[g2]) || !iszero(corners[g2_next])
                if !has_corner
                    admissible = false
                    if ci == cj
                        nc = lengths[ci]
                        dist_along = abs(li - lj)
                        dist_along = min(dist_along, nc - dist_along)
                        admissible = dist_along > 2
                    else
                        tol = sqrt(eps(δ2)) * max(one(δ2), abs(pv[ci]), abs(pv[cj]))
                        admissible = abs(pv[ci] - pv[cj]) <= tol
                    end

                    if admissible
                        ax1 = x[g1]
                        ay1 = y[g1]
                        if li < lengths[ci]
                            bx1 = x[g1 + 1]
                            by1 = y[g1 + 1]
                        else
                            off = offsets[ci]
                            bx1 = x[off] + wrapx[ci]
                            by1 = y[off] + wrapy[ci]
                        end

                        ax2 = x[g2]
                        ay2 = y[g2]
                        if lj < lengths[cj]
                            bx2 = x[g2 + 1]
                            by2 = y[g2 + 1]
                        else
                            off = offsets[cj]
                            bx2 = x[off] + wrapx[cj]
                            by2 = y[off] + wrapy[cj]
                        end

                        d2 = _flat_surgery_contact_distance2(ax1, ay1, bx1, by1,
                                                             ax2, ay2, bx2, by2)
                        is_valid = d2 < δ2
                    end
                end
            end
        end
        valid[pair_idx] = is_valid ? UInt8(1) : UInt8(0)
    end
end

@kernel function _compact_close_pair_candidates_kernel!(pair_ci, pair_i,
                                                        pair_cj, pair_j,
                                                        slots, valid,
                                                        contour_of_node,
                                                        local_index,
                                                        total_nodes, npairs)
    pair_idx = @index(Global)
    if pair_idx <= npairs && !iszero(valid[pair_idx])
        slot = slots[pair_idx]
        g1 = ((pair_idx - 1) % total_nodes) + 1
        g2 = ((pair_idx - 1) ÷ total_nodes) + 1
        pair_ci[slot] = contour_of_node[g1]
        pair_i[slot] = local_index[g1]
        pair_cj[slot] = contour_of_node[g2]
        pair_j[slot] = local_index[g2]
    end
end

function _device_close_pair_candidate_buffer(flat::FlatContourTopology{T}, δ,
                                             dev::AbstractDevice=CPU()) where {T}
    total_nodes = _flat_nnodes(flat)
    if total_nodes == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    npairs = total_nodes * total_nodes
    valid = device_zeros(dev, UInt8, npairs)
    @_ka_launch dev npairs _close_pair_candidate_kernel!(
        valid, flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy, flat.offsets,
        flat.lengths, flat.contour_of_node, flat.local_index, flat.corners,
        T(δ)^2, total_nodes)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    @_ka_launch dev npairs _prefix_u8_kernel!(slots, count_store, valid, npairs)
    ncandidates = to_cpu(count_store)[1]

    pair_ci = device_zeros(dev, Int, ncandidates)
    pair_i = device_zeros(dev, Int, ncandidates)
    pair_cj = device_zeros(dev, Int, ncandidates)
    pair_j = device_zeros(dev, Int, ncandidates)
    if ncandidates > 0
        @_ka_launch dev npairs _compact_close_pair_candidates_kernel!(
            pair_ci, pair_i, pair_cj, pair_j, slots, valid,
            flat.contour_of_node, flat.local_index, total_nodes, npairs)
    end

    return DeviceClosePairCandidates(pair_ci, pair_i, pair_cj, pair_j)
end

function _device_close_pair_candidate_buffer(contours::Vector{PVContour{T}}, δ,
                                             dev::AbstractDevice=CPU()) where {T}
    return _device_close_pair_candidate_buffer(_pack_flat_topology(contours, dev),
                                               δ, dev)
end

function _device_close_pair_candidate_buffer(state::DeviceContourState{T}, δ,
                                             dev::AbstractDevice=CPU()) where {T}
    return _device_close_pair_candidate_buffer(_flat_topology(state, dev),
                                               δ, dev)
end

function _unpack_close_pair_candidates(candidates::DeviceClosePairCandidates)
    ci = to_cpu(candidates.ci)
    i = to_cpu(candidates.i)
    cj = to_cpu(candidates.cj)
    j = to_cpu(candidates.j)
    pairs_out = Vector{Tuple{Int,Int,Int,Int}}(undef, length(ci))
    @inbounds for k in eachindex(ci)
        pairs_out[k] = (ci[k], i[k], cj[k], j[k])
    end
    return pairs_out
end

@inline function _flat_ray_crosses_segment(px, py, ax, ay, bx, by)
    (ay > py) == (by > py) && return false
    x_cross = ax + (py - ay) * (bx - ax) / (by - ay)
    return px < x_cross
end

@inline function _flat_segment_interior_probe(x, y, wrapx, wrapy, offsets,
                                              lengths, ci, i, δ)
    off = offsets[ci]
    n = lengths[ci]
    g = off + i - 1
    ax = x[g]
    ay = y[g]
    bx = i < n ? x[g + 1] : x[off] + wrapx[ci]
    by = i < n ? y[g + 1] : y[off] + wrapy[ci]
    sx = bx - ax
    sy = by - ay
    seg_len = sqrt(sx * sx + sy * sy)
    if seg_len <= eps(δ)
        return (ax + bx) / 2, (ay + by) / 2
    end

    area2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
    left_x = -sy / seg_len
    left_y = sx / seg_len
    inward_x = area2 >= zero(area2) ? left_x : -left_x
    inward_y = area2 >= zero(area2) ? left_y : -left_y
    probe_distance = max(δ / 10,
                         eps(δ) * (one(δ) + abs(ax) + abs(ay) + seg_len))
    return (ax + bx) / 2 + probe_distance * inward_x,
           (ay + by) / 2 + probe_distance * inward_y
end

@inline function _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                               offsets, lengths, ci)
    (!iszero(wrapx[ci]) || !iszero(wrapy[ci])) && return false
    inside = false
    off = offsets[ci]
    n = lengths[ci]
    @inbounds for li in 1:n
        g = off + li - 1
        ax = x[g]
        ay = y[g]
        bx = li < n ? x[g + 1] : x[off] + wrapx[ci]
        by = li < n ? y[g + 1] : y[off] + wrapy[ci]
        inside = inside != _flat_ray_crosses_segment(px, py, ax, ay, bx, by)
    end
    return inside
end

@inline function _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, ci, i, δ,
                                                ncontours)
    px, py = _flat_segment_interior_probe(x, y, wrapx, wrapy, offsets,
                                          lengths, ci, i, δ)
    q = zero(δ)
    @inbounds for ck in 1:ncontours
        if _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                         offsets, lengths, ck)
            q += pv[ck]
        end
    end
    return q
end

@kernel function _admissible_close_pair_kernel!(valid, pair_ci, pair_i,
                                                pair_cj, pair_j, x, y, pv,
                                                wrapx, wrapy, offsets, lengths,
                                                δ, ncontours, npairs)
    k = @index(Global)
    if k <= npairs
        ci = pair_ci[k]
        cj = pair_cj[k]
        ok = ci == cj
        if !ok
            qi = _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, ci,
                                                pair_i[k], δ, ncontours)
            qj = _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, cj,
                                                pair_j[k], δ, ncontours)
            tol = sqrt(eps(δ)) * max(one(δ), abs(qi), abs(qj))
            ok = abs(qi - qj) <= tol
        end
        valid[k] = ok ? UInt8(1) : UInt8(0)
    end
end

function _device_admissible_close_segment_buffer(flat::FlatContourTopology{T}, δ,
                                                 domain::UnboundedDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    candidates = _device_close_pair_candidate_buffer(flat, δ, dev)
    npairs = length(candidates.ci)
    npairs == 0 && return candidates

    ncontours = _flat_ncontours(flat)
    valid = device_zeros(dev, UInt8, npairs)
    @_ka_launch dev npairs _admissible_close_pair_kernel!(
        valid, candidates.ci, candidates.i, candidates.cj, candidates.j,
        flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy, flat.offsets,
        flat.lengths, T(δ), ncontours, npairs)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    @_ka_launch dev npairs _prefix_u8_kernel!(slots, count_store, valid, npairs)
    nadmissible = to_cpu(count_store)[1]

    out_ci = device_zeros(dev, Int, nadmissible)
    out_i = device_zeros(dev, Int, nadmissible)
    out_cj = device_zeros(dev, Int, nadmissible)
    out_j = device_zeros(dev, Int, nadmissible)
    if nadmissible > 0
        @_ka_launch dev npairs _compact_selected_pair_candidates_kernel!(
            out_ci, out_i, out_cj, out_j, slots, valid, candidates.ci,
            candidates.i, candidates.cj, candidates.j, npairs)
    end

    return DeviceClosePairCandidates(out_ci, out_i, out_cj, out_j)
end

function _device_admissible_close_segment_buffer(contours::Vector{PVContour{T}}, δ,
                                                 domain::UnboundedDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_admissible_close_segment_buffer(
        _pack_flat_topology(contours, dev), δ, domain, dev)
end

function _device_admissible_close_segment_buffer(state::DeviceContourState{T}, δ,
                                                 domain::UnboundedDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_admissible_close_segment_buffer(
        _flat_topology(state, dev), δ, domain, dev)
end

function _device_close_pair_candidates(contours::Vector{PVContour{T}}, δ,
                                       dev::AbstractDevice=CPU()) where {T}
    return _unpack_close_pair_candidates(
        _device_close_pair_candidate_buffer(contours, δ, dev))
end

function _pack_pair_vectors(pairs::Vector{Tuple{Int,Int,Int,Int}},
                            dev::AbstractDevice=CPU())
    npairs = length(pairs)
    ci = Vector{Int}(undef, npairs)
    i = Vector{Int}(undef, npairs)
    cj = Vector{Int}(undef, npairs)
    j = Vector{Int}(undef, npairs)
    @inbounds for k in 1:npairs
        ci[k], i[k], cj[k], j[k] = pairs[k]
    end
    return to_device(dev, ci), to_device(dev, i), to_device(dev, cj), to_device(dev, j)
end

function _pack_close_pair_candidates(pairs::Vector{Tuple{Int,Int,Int,Int}},
                                     dev::AbstractDevice=CPU())
    ci, i, cj, j = _pack_pair_vectors(pairs, dev)
    return DeviceClosePairCandidates(ci, i, cj, j)
end

@kernel function _pair_distance_plan_kernel!(distance2, op, pair_ci, pair_i,
                                             pair_cj, pair_j, x, y, wrapx,
                                             wrapy, offsets, lengths, npairs)
    k = @index(Global)
    if k <= npairs
        ci = pair_ci[k]
        i = pair_i[k]
        cj = pair_cj[k]
        j = pair_j[k]

        g1 = offsets[ci] + i - 1
        ax1 = x[g1]
        ay1 = y[g1]
        if i < lengths[ci]
            bx1 = x[g1 + 1]
            by1 = y[g1 + 1]
        else
            off = offsets[ci]
            bx1 = x[off] + wrapx[ci]
            by1 = y[off] + wrapy[ci]
        end

        g2 = offsets[cj] + j - 1
        ax2 = x[g2]
        ay2 = y[g2]
        if j < lengths[cj]
            bx2 = x[g2 + 1]
            by2 = y[g2 + 1]
        else
            off = offsets[cj]
            bx2 = x[off] + wrapx[cj]
            by2 = y[off] + wrapy[cj]
        end

        distance2[k] = _flat_surgery_contact_distance2(ax1, ay1, bx1, by1,
                                                       ax2, ay2, bx2, by2)
        op[k] = ci == cj ? UInt8(1) : UInt8(2)
    end
end

function _device_reconnection_plan_from_vectors(flat::FlatContourTopology{T},
                                                pair_ci, pair_i, pair_cj, pair_j,
                                                dev::AbstractDevice=CPU()) where {T}
    npairs = length(pair_ci)
    distance2 = device_zeros(dev, T, npairs)
    op = device_zeros(dev, UInt8, npairs)
    selected = device_zeros(dev, UInt8, npairs)
    if npairs > 0
        @_ka_launch dev npairs _pair_distance_plan_kernel!(
            distance2, op, pair_ci, pair_i, pair_cj, pair_j,
            flat.x, flat.y, flat.wrapx, flat.wrapy, flat.offsets,
            flat.lengths, npairs)
    end
    return DeviceReconnectionPlan(pair_ci, pair_i, pair_cj, pair_j,
                                  distance2, op, selected)
end

function _device_reconnection_plan_from_vectors(contours::Vector{PVContour{T}},
                                                pair_ci, pair_i, pair_cj, pair_j,
                                                dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(_pack_flat_topology(contours, dev),
                                                  pair_ci, pair_i, pair_cj,
                                                  pair_j, dev)
end

function _device_reconnection_plan(contours::Vector{PVContour{T}},
                                   close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                   dev::AbstractDevice=CPU()) where {T}
    pair_ci, pair_i, pair_cj, pair_j = _pack_pair_vectors(close_pairs, dev)
    return _device_reconnection_plan_from_vectors(contours, pair_ci, pair_i,
                                                  pair_cj, pair_j, dev)
end

function _device_reconnection_plan(contours::Vector{PVContour{T}},
                                   candidates::DeviceClosePairCandidates,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(contours, candidates.ci,
                                                  candidates.i, candidates.cj,
                                                  candidates.j, dev)
end

function _device_reconnection_plan(state::DeviceContourState{T},
                                   candidates::DeviceClosePairCandidates,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(_flat_topology(state, dev),
                                                  candidates.ci, candidates.i,
                                                  candidates.cj, candidates.j,
                                                  dev)
end

function _device_reconnection_plan(flat::FlatContourTopology{T},
                                   candidates::DeviceClosePairCandidates,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(flat, candidates.ci,
                                                  candidates.i, candidates.cj,
                                                  candidates.j, dev)
end

function _device_select_reconnection_pairs_from_plan(contours::Vector{PVContour{T}},
                                                     close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                     plan::DeviceReconnectionPlan{T}) where {T}
    distance2 = to_cpu(plan.distance2)

    ranked = Vector{Tuple{T,Tuple{Int,Int,Int,Int}}}(undef, length(close_pairs))
    @inbounds for k in eachindex(close_pairs)
        ranked[k] = (distance2[k], close_pairs[k])
    end
    sort!(ranked, by=first)

    used_contours = Set{Int}()
    selected_pairs = Tuple{Int,Int,Int,Int}[]
    selected_flags = zeros(UInt8, length(close_pairs))
    sizehint!(selected_pairs, min(length(close_pairs), length(contours)))
    for (_, pair) in ranked
        ci, _, cj, _ = pair
        (ci in used_contours || cj in used_contours) && continue
        push!(selected_pairs, pair)
        push!(used_contours, ci)
        push!(used_contours, cj)
        selected_idx = findfirst(==(pair), close_pairs)
        selected_idx === nothing || (selected_flags[selected_idx] = UInt8(1))
    end

    copyto!(plan.selected, selected_flags)
    return selected_pairs
end

@kernel function _select_independent_pairs_kernel!(selected, used_contours,
                                                   distance2, pair_ci,
                                                   pair_cj, npairs)
    worker = @index(Global)
    if worker == 1
        @inbounds for _ in 1:npairs
            best = 0
            best_d2 = typemax(typeof(distance2[1]))
            for k in 1:npairs
                iszero(selected[k]) || continue
                ci = pair_ci[k]
                cj = pair_cj[k]
                (iszero(used_contours[ci]) && iszero(used_contours[cj])) || continue
                d2 = distance2[k]
                if best == 0 || d2 < best_d2
                    best = k
                    best_d2 = d2
                end
            end
            best == 0 && break
            ci = pair_ci[best]
            cj = pair_cj[best]
            selected[best] = UInt8(1)
            used_contours[ci] = UInt8(1)
            used_contours[cj] = UInt8(1)
        end
    end
end

@kernel function _compact_selected_pair_candidates_kernel!(out_ci, out_i,
                                                           out_cj, out_j,
                                                           slots, selected,
                                                           pair_ci, pair_i,
                                                           pair_cj, pair_j,
                                                           npairs)
    k = @index(Global)
    if k <= npairs && !iszero(selected[k])
        slot = slots[k]
        out_ci[slot] = pair_ci[k]
        out_i[slot] = pair_i[k]
        out_cj[slot] = pair_cj[k]
        out_j[slot] = pair_j[k]
    end
end

function _device_select_reconnection_pair_buffer(flat::FlatContourTopology{T},
                                                 candidates::DeviceClosePairCandidates,
                                                 dev::AbstractDevice=CPU()) where {T}
    npairs = length(candidates.ci)
    if npairs == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    plan = _device_reconnection_plan(flat, candidates, dev)
    used_contours = device_zeros(dev, UInt8, _flat_ncontours(flat))
    @_ka_launch dev 1 _select_independent_pairs_kernel!(
        plan.selected, used_contours, plan.distance2, plan.ci, plan.cj, npairs)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    @_ka_launch dev npairs _prefix_u8_kernel!(slots, count_store, plan.selected, npairs)
    nselected = to_cpu(count_store)[1]

    out_ci = device_zeros(dev, Int, nselected)
    out_i = device_zeros(dev, Int, nselected)
    out_cj = device_zeros(dev, Int, nselected)
    out_j = device_zeros(dev, Int, nselected)
    if nselected > 0
        @_ka_launch dev npairs _compact_selected_pair_candidates_kernel!(
            out_ci, out_i, out_cj, out_j, slots, plan.selected, plan.ci,
            plan.i, plan.cj, plan.j, npairs)
    end

    return DeviceClosePairCandidates(out_ci, out_i, out_cj, out_j)
end

function _device_select_reconnection_pair_buffer(contours::Vector{PVContour{T}},
                                                 candidates::DeviceClosePairCandidates,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_select_reconnection_pair_buffer(
        _pack_flat_topology(contours, dev), candidates, dev)
end

function _device_select_reconnection_pair_buffer(state::DeviceContourState{T},
                                                 candidates::DeviceClosePairCandidates,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_select_reconnection_pair_buffer(
        _flat_topology(state, dev), candidates, dev)
end

function _device_select_reconnection_pair_buffer(contours::Vector{PVContour{T}},
                                                 close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_select_reconnection_pair_buffer(
        contours, _pack_close_pair_candidates(close_pairs, dev), dev)
end

function _device_select_reconnection_pairs(contours::Vector{PVContour{T}},
                                           close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                           dev::AbstractDevice=CPU()) where {T}
    isempty(close_pairs) && return Tuple{Int,Int,Int,Int}[]
    plan = _device_reconnection_plan(contours, close_pairs, dev)
    return _device_select_reconnection_pairs_from_plan(contours, close_pairs, plan)
end

function _device_select_reconnection_pairs(contours::Vector{PVContour{T}},
                                           candidates::DeviceClosePairCandidates,
                                           dev::AbstractDevice=CPU()) where {T}
    return _unpack_close_pair_candidates(
        _device_select_reconnection_pair_buffer(contours, candidates, dev))
end

@inline function _flat_contour_node_xy(x, y, offsets, ci, local_idx)
    g = offsets[ci] + local_idx - 1
    return x[g], y[g]
end

@inline function _flat_oriented_contour_node_xy(x, y, offsets, lengths, ci,
                                                local_idx, reversed)
    n = lengths[ci]
    source_idx = reversed ? n - local_idx + 1 : local_idx
    return _flat_contour_node_xy(x, y, offsets, ci, source_idx)
end

@inline function _flat_inserted_contour_node_xy(x, y, offsets, ci, inserted_idx,
                                                stitch_x, stitch_y, local_idx)
    if inserted_idx > 0 && local_idx == inserted_idx
        return stitch_x, stitch_y
    end
    original_idx = inserted_idx > 0 && local_idx > inserted_idx ? local_idx - 1 : local_idx
    return _flat_contour_node_xy(x, y, offsets, ci, original_idx)
end

@inline function _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
    n = lengths[ci]
    off = offsets[ci]
    area2 = zero(eltype(x))
    @inbounds for li in 1:n
        g = off + li - 1
        if li < n
            nx = x[g + 1]
            ny = y[g + 1]
        else
            nx = x[off] + wrapx[ci]
            ny = y[off] + wrapy[ci]
        end
        area2 += x[g] * ny - nx * y[g]
    end
    return area2
end

@inline function _flat_split_part_area2(x, y, offsets, ci, inserted_idx,
                                        stitch_x, stitch_y, start_idx, len)
    area2 = zero(eltype(x))
    @inbounds for m in 1:len
        local_idx = start_idx + m - 1
        next_idx = m == len ? start_idx : local_idx + 1
        x1, y1 = _flat_inserted_contour_node_xy(x, y, offsets, ci,
                                                inserted_idx, stitch_x,
                                                stitch_y, local_idx)
        x2, y2 = _flat_inserted_contour_node_xy(x, y, offsets, ci,
                                                inserted_idx, stitch_x,
                                                stitch_y, next_idx)
        area2 += x1 * y2 - x2 * y1
    end
    return area2
end

@inline function _flat_wrapped_split_part_area2(x, y, offsets, ci, inserted_idx,
                                                stitch_x, stitch_y, hi, lo,
                                                nc, len)
    area2 = zero(eltype(x))
    @inbounds for m in 1:len
        pos = m <= nc - hi + 1 ? hi + m - 1 : m - (nc - hi + 1)
        next_m = m == len ? 1 : m + 1
        next_pos = next_m <= nc - hi + 1 ? hi + next_m - 1 : next_m - (nc - hi + 1)
        x1, y1 = _flat_inserted_contour_node_xy(x, y, offsets, ci,
                                                inserted_idx, stitch_x,
                                                stitch_y, pos)
        x2, y2 = _flat_inserted_contour_node_xy(x, y, offsets, ci,
                                                inserted_idx, stitch_x,
                                                stitch_y, next_pos)
        area2 += x1 * y2 - x2 * y1
    end
    return area2
end

@kernel function _topology_rewrite_size_kernel!(op, valid, node_from_first,
                                                node_idx, seg_idx, inserted_idx,
                                                split_reverse1, split_reverse2,
                                                merge_reverse_second,
                                                stitch_x, stitch_y, out_count,
                                                out_len1, out_len2, pair_ci,
                                                pair_i, pair_cj, pair_j, x, y,
                                                wrapx, wrapy, offsets, lengths,
                                                npairs)
    k = @index(Global)
    if k <= npairs
        ci = pair_ci[k]
        i = pair_i[k]
        cj = pair_cj[k]
        j = pair_j[k]
        n1 = lengths[ci]
        n2 = lengths[cj]
        op_k = ci == cj ? UInt8(1) : UInt8(2)
        reverse_second = false
        if op_k == UInt8(2)
            area1_2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
            area2_2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, cj)
            area_tol = eps(one(area1_2)) * 2000
            reverse_second = abs(area1_2) > area_tol && abs(area2_2) > area_tol &&
                             ((area1_2 > zero(area1_2)) != (area2_2 > zero(area2_2)))
        end

        g1 = offsets[ci] + i - 1
        ax1 = x[g1]
        ay1 = y[g1]
        i_end = i < n1 ? i + 1 : 1
        g1_end = offsets[ci] + i_end - 1
        bx1 = i < n1 ? x[g1 + 1] : x[offsets[ci]] + wrapx[ci]
        by1 = i < n1 ? y[g1 + 1] : y[offsets[ci]] + wrapy[ci]

        j_eff = reverse_second ? (n2 - j == 0 ? n2 : n2 - j) : j
        j_end = j_eff < n2 ? j_eff + 1 : 1
        ax2, ay2 = _flat_oriented_contour_node_xy(x, y, offsets, lengths, cj,
                                                  j_eff, reverse_second)
        bx2, by2 = _flat_oriented_contour_node_xy(x, y, offsets, lengths, cj,
                                                  j_end, reverse_second)
        if !reverse_second && j_eff == n2
            bx2 = x[offsets[cj]] + wrapx[cj]
            by2 = y[offsets[cj]] + wrapy[cj]
        end

        best_d2, _, _ = _flat_point_segment_closest(ax1, ay1, ax2, ay2, bx2, by2)
        best_x = ax1
        best_y = ay1
        best_node_from_first = UInt8(1)
        best_node_idx = i
        best_seg_idx = j_eff

        x1_end = x[g1_end]
        y1_end = y[g1_end]
        d2, _, _ = _flat_point_segment_closest(x1_end, y1_end, ax2, ay2, bx2, by2)
        if d2 < best_d2
            best_d2 = d2
            best_x = x1_end
            best_y = y1_end
            best_node_from_first = UInt8(1)
            best_node_idx = i_end
            best_seg_idx = j_eff
        end

        d2, _, _ = _flat_point_segment_closest(ax2, ay2, ax1, ay1, bx1, by1)
        if d2 < best_d2
            best_d2 = d2
            best_x = ax2
            best_y = ay2
            best_node_from_first = UInt8(0)
            best_node_idx = j_eff
            best_seg_idx = i
        end

        d2, _, _ = _flat_point_segment_closest(bx2, by2, ax1, ay1, bx1, by1)
        if d2 < best_d2
            best_x = bx2
            best_y = by2
            best_node_from_first = UInt8(0)
            best_node_idx = j_end
            best_seg_idx = i
        end

        inserted = best_seg_idx == (op_k == UInt8(1) ? n1 : (best_node_from_first == UInt8(1) ? n2 : n1)) ?
                   (op_k == UInt8(1) ? n1 : (best_node_from_first == UInt8(1) ? n2 : n1)) + 1 :
                   best_seg_idx + 1

        valid_k = UInt8(1)
        count_k = 1
        len1 = 0
        len2 = 0
        reverse1 = UInt8(0)
        reverse2 = UInt8(0)
        if op_k == UInt8(1)
            adjusted_node = best_seg_idx < best_node_idx ? best_node_idx + 1 : best_node_idx
            lo = min(adjusted_node, inserted)
            hi = max(adjusted_node, inserted)
            len1 = hi - lo
            len2 = n1 + 1 - len1
            if len1 >= 3 && len2 >= 3
                count_k = 2
                parent_area2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
                if parent_area2 != zero(parent_area2)
                    area1_2 = _flat_split_part_area2(x, y, offsets, ci, inserted,
                                                     best_x, best_y, lo, len1)
                    area2_2 = _flat_wrapped_split_part_area2(x, y, offsets, ci,
                                                             inserted, best_x,
                                                             best_y, hi, lo,
                                                             n1 + 1, len2)
                    parent_pos = parent_area2 > zero(parent_area2)
                    reverse1 = ((parent_pos && area1_2 < zero(area1_2)) ||
                                (!parent_pos && area1_2 > zero(area1_2))) ? UInt8(1) : UInt8(0)
                    reverse2 = ((parent_pos && area2_2 < zero(area2_2)) ||
                                (!parent_pos && area2_2 > zero(area2_2))) ? UInt8(1) : UInt8(0)
                end
            else
                valid_k = UInt8(0)
                count_k = 1
                len1 = n1
                len2 = 0
            end
        else
            len1 = n1 + n2 + 1
            len2 = 0
        end

        op[k] = op_k
        valid[k] = valid_k
        node_from_first[k] = best_node_from_first
        node_idx[k] = best_node_idx
        seg_idx[k] = best_seg_idx
        inserted_idx[k] = inserted
        split_reverse1[k] = reverse1
        split_reverse2[k] = reverse2
        merge_reverse_second[k] = reverse_second ? UInt8(1) : UInt8(0)
        stitch_x[k] = best_x
        stitch_y[k] = best_y
        out_count[k] = count_k
        out_len1[k] = len1
        out_len2[k] = len2
    end
end

function _device_topology_rewrite_plan_from_vectors(flat::FlatContourTopology{T},
                                                    pair_ci, pair_i, pair_cj, pair_j,
                                                    dev::AbstractDevice=CPU()) where {T}
    npairs = length(pair_ci)
    op = device_zeros(dev, UInt8, npairs)
    valid = device_zeros(dev, UInt8, npairs)
    node_from_first = device_zeros(dev, UInt8, npairs)
    node_idx = device_zeros(dev, Int, npairs)
    seg_idx = device_zeros(dev, Int, npairs)
    inserted_idx = device_zeros(dev, Int, npairs)
    split_reverse1 = device_zeros(dev, UInt8, npairs)
    split_reverse2 = device_zeros(dev, UInt8, npairs)
    merge_reverse_second = device_zeros(dev, UInt8, npairs)
    stitch_x = device_zeros(dev, T, npairs)
    stitch_y = device_zeros(dev, T, npairs)
    out_count = device_zeros(dev, Int, npairs)
    out_len1 = device_zeros(dev, Int, npairs)
    out_len2 = device_zeros(dev, Int, npairs)

    if npairs > 0
        @_ka_launch dev npairs _topology_rewrite_size_kernel!(
            op, valid, node_from_first, node_idx, seg_idx, inserted_idx,
            split_reverse1, split_reverse2, merge_reverse_second, stitch_x,
            stitch_y, out_count, out_len1, out_len2, pair_ci, pair_i,
            pair_cj, pair_j, flat.x, flat.y, flat.wrapx, flat.wrapy,
            flat.offsets, flat.lengths, npairs)
    end

    return DeviceTopologyRewritePlan(pair_ci, pair_i, pair_cj, pair_j, op,
                                     valid, node_from_first, node_idx, seg_idx,
                                     inserted_idx, split_reverse1,
                                     split_reverse2, merge_reverse_second,
                                     stitch_x, stitch_y, out_count, out_len1,
                                     out_len2)
end

function _device_topology_rewrite_plan_from_vectors(contours::Vector{PVContour{T}},
                                                    pair_ci, pair_i, pair_cj, pair_j,
                                                    dev::AbstractDevice=CPU()) where {T}
    return _device_topology_rewrite_plan_from_vectors(_pack_flat_topology(contours, dev),
                                                      pair_ci, pair_i, pair_cj,
                                                      pair_j, dev)
end

function _device_topology_rewrite_plan(contours::Vector{PVContour{T}},
                                       selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                       dev::AbstractDevice=CPU()) where {T}
    pair_ci, pair_i, pair_cj, pair_j = _pack_pair_vectors(selected_pairs, dev)
    return _device_topology_rewrite_plan_from_vectors(contours, pair_ci, pair_i,
                                                      pair_cj, pair_j, dev)
end

function _device_topology_rewrite_plan(contours::Vector{PVContour{T}},
                                       selected_pairs::DeviceClosePairCandidates,
                                       dev::AbstractDevice=CPU()) where {T}
    return _device_topology_rewrite_plan_from_vectors(contours, selected_pairs.ci,
                                                      selected_pairs.i,
                                                      selected_pairs.cj,
                                                      selected_pairs.j, dev)
end

function _device_topology_rewrite_plan(state::DeviceContourState{T},
                                       selected_pairs::DeviceClosePairCandidates,
                                       dev::AbstractDevice=CPU()) where {T}
    return _device_topology_rewrite_plan_from_vectors(_flat_topology(state, dev),
                                                      selected_pairs.ci,
                                                      selected_pairs.i,
                                                      selected_pairs.cj,
                                                      selected_pairs.j, dev)
end

function _device_topology_rewrite_plan(flat::FlatContourTopology{T},
                                       selected_pairs::DeviceClosePairCandidates,
                                       dev::AbstractDevice=CPU()) where {T}
    return _device_topology_rewrite_plan_from_vectors(flat, selected_pairs.ci,
                                                      selected_pairs.i,
                                                      selected_pairs.cj,
                                                      selected_pairs.j, dev)
end

@inline function _inserted_contour_node(x, y, corners, offsets, ci, inserted_idx,
                                        stitch_x, stitch_y, local_idx)
    if inserted_idx > 0 && local_idx == inserted_idx
        return stitch_x, stitch_y, UInt8(0)
    end
    original_idx = inserted_idx > 0 && local_idx > inserted_idx ? local_idx - 1 : local_idx
    g = offsets[ci] + original_idx - 1
    return x[g], y[g], corners[g]
end

@inline function _inserted_oriented_contour_node(x, y, corners, offsets, lengths,
                                                 ci, inserted_idx, stitch_x,
                                                 stitch_y, local_idx, reversed)
    if inserted_idx > 0 && local_idx == inserted_idx
        return stitch_x, stitch_y, UInt8(0)
    end
    oriented_idx = inserted_idx > 0 && local_idx > inserted_idx ? local_idx - 1 : local_idx
    original_idx = reversed ? lengths[ci] - oriented_idx + 1 : oriented_idx
    g = offsets[ci] + original_idx - 1
    return x[g], y[g], corners[g]
end

@kernel function _materialize_rewrite_outputs_kernel!(out_x, out_y, out_corners,
                                                       out_offsets, out_lengths,
                                                       out_node_contour, out_op_index,
                                                       out_source_contour, out_part,
                                                       pair_ci, pair_cj,
                                                       op, valid, node_from_first,
                                                       node_idx, seg_idx, inserted_idx,
                                                       split_reverse1, split_reverse2,
                                                       merge_reverse_second,
                                                       stitch_x, stitch_y, in_x, in_y,
                                                       in_corners, in_offsets,
                                                       in_lengths, total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        out_ci = out_node_contour[g]
        op_idx = out_op_index[out_ci]
        part = out_part[out_ci]
        out_local = g - out_offsets[out_ci] + 1

        ox = zero(eltype(out_x))
        oy = zero(eltype(out_y))
        corner = UInt8(0)

        if op_idx == 0
            ci = out_source_contour[out_ci]
            in_g = in_offsets[ci] + out_local - 1
            ox = in_x[in_g]
            oy = in_y[in_g]
            corner = in_corners[in_g]
        elseif !iszero(valid[op_idx])
            ci = pair_ci[op_idx]
            cj = pair_cj[op_idx]
            if op[op_idx] == UInt8(1)
                n = in_lengths[ci]
                inserted = inserted_idx[op_idx]
                adjusted_node = seg_idx[op_idx] < node_idx[op_idx] ? node_idx[op_idx] + 1 : node_idx[op_idx]
                lo = min(adjusted_node, inserted)
                hi = max(adjusted_node, inserted)
                nc = n + 1
                out_len = out_lengths[out_ci]
                reverse_part = part == 1 ? !iszero(split_reverse1[op_idx]) :
                                           !iszero(split_reverse2[op_idx])
                logical_local = reverse_part ? out_len - out_local + 1 : out_local
                source_local = 1
                if part == 1
                    source_local = lo + logical_local - 1
                else
                    first_span = nc - hi + 1
                    source_local = logical_local <= first_span ? hi + logical_local - 1 :
                                                           logical_local - first_span
                end
                ox, oy, corner = _inserted_contour_node(in_x, in_y, in_corners,
                                                        in_offsets, ci, inserted,
                                                        stitch_x[op_idx],
                                                        stitch_y[op_idx],
                                                        source_local)
                (reverse_part ? out_local == out_len : out_local == 1) && (corner = UInt8(1))
            else
                n1 = in_lengths[pair_ci[op_idx]]
                n2 = in_lengths[pair_cj[op_idx]]
                from_first = !iszero(node_from_first[op_idx])
                reverse_second = !iszero(merge_reverse_second[op_idx])
                c1_inserted = from_first ? 0 : inserted_idx[op_idx]
                c2_inserted = from_first ? inserted_idx[op_idx] : 0
                c1_len = n1 + (from_first ? 0 : 1)
                c2_len = n2 + (from_first ? 1 : 0)
                c1_start = from_first ? node_idx[op_idx] : inserted_idx[op_idx]
                c2_start = from_first ? inserted_idx[op_idx] : node_idx[op_idx]

                if out_local <= c1_len
                    source_local = c1_start + out_local - 1
                    source_local = source_local > c1_len ? source_local - c1_len : source_local
                    ox, oy, corner = _inserted_oriented_contour_node(
                        in_x, in_y, in_corners, in_offsets, in_lengths, ci,
                        c1_inserted, stitch_x[op_idx], stitch_y[op_idx],
                        source_local, false)
                    out_local == 1 && (corner = UInt8(1))
                else
                    local2 = out_local - c1_len
                    source_local = c2_start + local2 - 1
                    source_local = source_local > c2_len ? source_local - c2_len : source_local
                    ox, oy, corner = _inserted_oriented_contour_node(
                        in_x, in_y, in_corners, in_offsets, in_lengths, cj,
                        c2_inserted, stitch_x[op_idx], stitch_y[op_idx],
                        source_local, reverse_second)
                    local2 == 1 && (corner = UInt8(1))
                end
            end
        end

        out_x[g] = ox
        out_y[g] = oy
        out_corners[g] = corner
    end
end

function _rewrite_output_layout(contours::Vector{PVContour{T}},
                                plan::DeviceTopologyRewritePlan,
                                dev::AbstractDevice=CPU()) where {T}
    op = to_cpu(plan.op)
    valid = to_cpu(plan.valid)
    out_count = to_cpu(plan.out_count)
    out_len1 = to_cpu(plan.out_len1)
    out_len2 = to_cpu(plan.out_len2)
    pair_ci = to_cpu(plan.ci)

    offsets = Int[]
    lengths = Int[]
    op_index = Int[]
    source_contour = Int[]
    part = Int[]
    pv = T[]
    wrapx = T[]
    wrapy = T[]
    cursor = 1
    @inbounds for k in eachindex(op)
        iszero(valid[k]) && continue
        ci = pair_ci[k]
        push!(offsets, cursor)
        push!(lengths, out_len1[k])
        push!(op_index, k)
        push!(source_contour, ci)
        push!(part, 1)
        push!(pv, contours[ci].pv)
        push!(wrapx, contours[ci].wrap[1])
        push!(wrapy, contours[ci].wrap[2])
        cursor += out_len1[k]
        if out_count[k] == 2
            push!(offsets, cursor)
            push!(lengths, out_len2[k])
            push!(op_index, k)
            push!(source_contour, ci)
            push!(part, 2)
            push!(pv, contours[ci].pv)
            push!(wrapx, contours[ci].wrap[1])
            push!(wrapy, contours[ci].wrap[2])
            cursor += out_len2[k]
        end
    end

    total_nodes = cursor - 1
    out_node_contour = Vector{Int}(undef, total_nodes)
    @inbounds for out_ci in eachindex(offsets)
        for g in offsets[out_ci]:(offsets[out_ci] + lengths[out_ci] - 1)
            out_node_contour[g] = out_ci
        end
    end

    return (offsets=to_device(dev, offsets),
            lengths=to_device(dev, lengths),
            op_index=to_device(dev, op_index),
            source_contour=to_device(dev, source_contour),
            part=to_device(dev, part),
            pv=to_device(dev, pv),
            wrapx=to_device(dev, wrapx),
            wrapy=to_device(dev, wrapy),
            out_node_contour=to_device(dev, out_node_contour),
            total_nodes=total_nodes)
end

function _full_rewrite_output_layout(contours::Vector{PVContour{T}},
                                     plan::DeviceTopologyRewritePlan,
                                     dev::AbstractDevice=CPU()) where {T}
    ncontours = length(contours)
    op = to_cpu(plan.op)
    valid = to_cpu(plan.valid)
    out_count = to_cpu(plan.out_count)
    out_len1 = to_cpu(plan.out_len1)
    out_len2 = to_cpu(plan.out_len2)
    pair_ci = to_cpu(plan.ci)
    pair_cj = to_cpu(plan.cj)

    replacement_op = zeros(Int, ncontours)
    deleted = falses(ncontours)
    @inbounds for k in eachindex(op)
        iszero(valid[k]) && continue
        ci = pair_ci[k]
        replacement_op[ci] = k
        if op[k] == UInt8(2)
            deleted[pair_cj[k]] = true
        end
    end

    offsets = Int[]
    lengths = Int[]
    op_index = Int[]
    source_contour = Int[]
    part = Int[]
    pv = T[]
    wrapx = T[]
    wrapy = T[]
    cursor = 1

    @inbounds for ci in 1:ncontours
        deleted[ci] && continue
        k = replacement_op[ci]
        if k == 0
            len = nnodes(contours[ci])
            push!(offsets, cursor)
            push!(lengths, len)
            push!(op_index, 0)
            push!(source_contour, ci)
            push!(part, 0)
            push!(pv, contours[ci].pv)
            push!(wrapx, contours[ci].wrap[1])
            push!(wrapy, contours[ci].wrap[2])
            cursor += len
        else
            push!(offsets, cursor)
            push!(lengths, out_len1[k])
            push!(op_index, k)
            push!(source_contour, ci)
            push!(part, 1)
            push!(pv, contours[ci].pv)
            push!(wrapx, contours[ci].wrap[1])
            push!(wrapy, contours[ci].wrap[2])
            cursor += out_len1[k]
        end
    end

    @inbounds for k in eachindex(op)
        (iszero(valid[k]) || out_count[k] != 2) && continue
        ci = pair_ci[k]
        push!(offsets, cursor)
        push!(lengths, out_len2[k])
        push!(op_index, k)
        push!(source_contour, ci)
        push!(part, 2)
        push!(pv, contours[ci].pv)
        push!(wrapx, contours[ci].wrap[1])
        push!(wrapy, contours[ci].wrap[2])
        cursor += out_len2[k]
    end

    total_nodes = cursor - 1
    out_node_contour = Vector{Int}(undef, total_nodes)
    @inbounds for out_ci in eachindex(offsets)
        for g in offsets[out_ci]:(offsets[out_ci] + lengths[out_ci] - 1)
            out_node_contour[g] = out_ci
        end
    end

    return (offsets=to_device(dev, offsets),
            lengths=to_device(dev, lengths),
            op_index=to_device(dev, op_index),
            source_contour=to_device(dev, source_contour),
            part=to_device(dev, part),
            pv=to_device(dev, pv),
            wrapx=to_device(dev, wrapx),
            wrapy=to_device(dev, wrapy),
            out_node_contour=to_device(dev, out_node_contour),
            total_nodes=total_nodes)
end

@kernel function _full_rewrite_roles_kernel!(replacement_op, deleted,
                                             pair_ci, pair_cj, op, valid, npairs)
    k = @index(Global)
    if k <= npairs && !iszero(valid[k])
        replacement_op[pair_ci[k]] = k
        if op[k] == UInt8(2)
            deleted[pair_cj[k]] = UInt8(1)
        end
    end
end

@kernel function _full_rewrite_keep_flags_kernel!(main_keep, extra_keep,
                                                  replacement_op, deleted,
                                                  valid, out_count,
                                                  ncontours, npairs)
    idx = @index(Global)
    if idx <= ncontours
        main_keep[idx] = iszero(deleted[idx]) ? UInt8(1) : UInt8(0)
    end
    if idx <= npairs
        extra_keep[idx] = !iszero(valid[idx]) && out_count[idx] == 2 ? UInt8(1) : UInt8(0)
    end
end

@kernel function _prefix_u8_kernel!(slots, total, flags, n)
    idx = @index(Global)
    if idx <= n
        s = 0
        @inbounds for i in 1:(idx - 1)
            s += Int(flags[i])
        end
        slots[idx] = !iszero(flags[idx]) ? s + 1 : 0
        if idx == n
            count = s + Int(flags[idx])
            total[1] = count
        end
    end
end

@kernel function _full_rewrite_fill_main_layout_kernel!(out_lengths, out_op_index,
                                                        out_source_contour,
                                                        out_part, out_pv,
                                                        out_wrapx, out_wrapy,
                                                        main_keep, main_slot,
                                                        replacement_op,
                                                        in_lengths, in_pv,
                                                        in_wrapx, in_wrapy,
                                                        out_len1, ncontours)
    ci = @index(Global)
    if ci <= ncontours && !iszero(main_keep[ci])
        slot = main_slot[ci]
        op_idx = replacement_op[ci]
        out_lengths[slot] = op_idx == 0 ? in_lengths[ci] : out_len1[op_idx]
        out_op_index[slot] = op_idx
        out_source_contour[slot] = ci
        out_part[slot] = op_idx == 0 ? 0 : 1
        out_pv[slot] = in_pv[ci]
        out_wrapx[slot] = in_wrapx[ci]
        out_wrapy[slot] = in_wrapy[ci]
    end
end

@kernel function _full_rewrite_fill_extra_layout_kernel!(out_lengths, out_op_index,
                                                         out_source_contour,
                                                         out_part, out_pv,
                                                         out_wrapx, out_wrapy,
                                                         extra_keep, extra_slot,
                                                         main_count, pair_ci,
                                                         out_len2, in_pv,
                                                         in_wrapx, in_wrapy,
                                                         npairs)
    k = @index(Global)
    if k <= npairs && !iszero(extra_keep[k])
        ci = pair_ci[k]
        slot = main_count[1] + extra_slot[k]
        out_lengths[slot] = out_len2[k]
        out_op_index[slot] = k
        out_source_contour[slot] = ci
        out_part[slot] = 2
        out_pv[slot] = in_pv[ci]
        out_wrapx[slot] = in_wrapx[ci]
        out_wrapy[slot] = in_wrapy[ci]
    end
end

@kernel function _prefix_lengths_kernel!(offsets, total_nodes, lengths, n)
    idx = @index(Global)
    if idx <= n
        offset = 1
        @inbounds for i in 1:(idx - 1)
            offset += lengths[i]
        end
        offsets[idx] = offset
        if idx == n
            total_nodes[1] = offset + lengths[idx] - 1
        end
    end
end

@kernel function _out_node_contour_kernel!(out_node_contour, offsets, lengths, nout)
    out_ci = @index(Global)
    if out_ci <= nout
        first = offsets[out_ci]
        last = first + lengths[out_ci] - 1
        @inbounds for g in first:last
            out_node_contour[g] = out_ci
        end
    end
end

@kernel function _out_node_local_index_kernel!(local_index, contour_of_node,
                                               offsets, total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        local_index[g] = g - offsets[ci] + 1
    end
end

function _device_state_from_outputs(outputs::DeviceRewriteOutputs{T},
                                    dev::AbstractDevice=CPU()) where {T}
    ncontours = length(outputs.lengths)
    total_nodes = length(outputs.x)
    contour_of_node = device_zeros(dev, Int, total_nodes)
    local_index = device_zeros(dev, Int, total_nodes)
    if ncontours > 0 && total_nodes > 0
        @_ka_launch dev ncontours _out_node_contour_kernel!(
            contour_of_node, outputs.offsets, outputs.lengths, ncontours)
        @_ka_launch dev total_nodes _out_node_local_index_kernel!(
            local_index, contour_of_node, outputs.offsets, total_nodes)
    end
    return DeviceContourState(outputs.x, outputs.y, outputs.pv, outputs.wrapx,
                              outputs.wrapy, outputs.offsets, outputs.lengths,
                              outputs.corners, contour_of_node, local_index)
end

function _replace_device_state!(state::DeviceContourState{T},
                                outputs::DeviceRewriteOutputs{T},
                                dev::AbstractDevice=CPU()) where {T}
    replacement = _device_state_from_outputs(outputs, dev)
    state.x = replacement.x
    state.y = replacement.y
    state.pv = replacement.pv
    state.wrapx = replacement.wrapx
    state.wrapy = replacement.wrapy
    state.offsets = replacement.offsets
    state.lengths = replacement.lengths
    state.corners = replacement.corners
    state.contour_of_node = replacement.contour_of_node
    state.local_index = replacement.local_index
    return state
end

@inline function _flat_target_interval_count(total_length, current::Int, μ, Δ_max)
    min_intervals = max(1, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    return clamp(current, min_intervals, max_intervals)
end

@inline function _flat_closed_remesh_interval_count(total_length, q, μ, Δ_max)
    min_intervals = max(3, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    return clamp(Int(round(q)), min_intervals, max_intervals)
end

@inline function _flat_fixed_span_interval_count(total_length, q, current::Int,
                                                 μ, Δ_max)
    if total_length <= eps(total_length)
        return max(1, current)
    end
    min_intervals = max(1, Int(ceil(total_length / Δ_max)))
    max_intervals = max(min_intervals, Int(floor(total_length / μ)))
    target_intervals = _flat_target_interval_count(total_length, current, μ, Δ_max)
    n_intervals = clamp(Int(round(q)), min_intervals, max_intervals)
    return max(1, n_intervals, target_intervals)
end

@inline function _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                             lengths, corners, ci, li)
    n = lengths[ci]
    n < 3 && return zero(eltype(x))
    off = offsets[ci]
    prev_i = li == 1 ? n : li - 1
    next_i = li == n ? 1 : li + 1
    (!iszero(corners[off + prev_i - 1]) ||
     !iszero(corners[off + li - 1]) ||
     !iszero(corners[off + next_i - 1])) && return zero(eltype(x))

    g = off + li - 1
    curr_x = x[g]
    curr_y = y[g]
    prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
    prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
    next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
    next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]

    ax = curr_x - prev_x
    ay = curr_y - prev_y
    bx = next_x - curr_x
    by = next_y - curr_y
    chord_x = next_x - prev_x
    chord_y = next_y - prev_y
    a_norm = sqrt(ax * ax + ay * ay)
    b_norm = sqrt(bx * bx + by * by)
    chord_norm = sqrt(chord_x * chord_x + chord_y * chord_y)
    denom = a_norm * b_norm * chord_norm
    denom <= eps(denom) && return zero(denom)
    return 2 * (ax * by - ay * bx) / denom
end

@kernel function _demote_obtuse_state_corners_kernel!(corners, x, y, wrapx,
                                                      wrapy, offsets, lengths,
                                                      contour_of_node,
                                                      local_index, total_nodes)
    g = @index(Global)
    if g <= total_nodes && !iszero(corners[g])
        ci = contour_of_node[g]
        li = local_index[g]
        n = lengths[ci]
        if n >= 3
            off = offsets[ci]
            curr_x = x[g]
            curr_y = y[g]
            prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
            prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
            next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
            next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]
            vprev_x = prev_x - curr_x
            vprev_y = prev_y - curr_y
            vnext_x = next_x - curr_x
            vnext_y = next_y - curr_y
            if vprev_x * vnext_x + vprev_y * vnext_y < zero(curr_x)
                corners[g] = UInt8(0)
            end
        end
    end
end

@kernel function _promote_high_curvature_state_corners_kernel!(
    out_corners, corners, x, y, wrapx, wrapy, offsets, lengths,
    δ, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        @inbounds for li in 1:n
            g = off + li - 1
            out_corners[g] = corners[g]
        end
        if n >= 3 && iszero(wrapx[ci]) && iszero(wrapy[ci])
            @inbounds for li in 1:n
                g = off + li - 1
                iszero(out_corners[g]) || continue
                prev_i = li == 1 ? n : li - 1
                next_i = li == n ? 1 : li + 1
                prev_g = off + prev_i - 1
                next_g = off + next_i - 1
                (iszero(out_corners[prev_g]) && iszero(out_corners[next_g])) || continue
                κ = _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                                 lengths, corners, ci, li)
                abs(κ) >= inv(δ) || continue
                curr_x = x[g]
                curr_y = y[g]
                prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
                prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
                next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
                next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]
                v1x = prev_x - curr_x
                v1y = prev_y - curr_y
                v2x = next_x - curr_x
                v2y = next_y - curr_y
                if v1x * v2x + v1y * v2y > zero(curr_x)
                    out_corners[g] = UInt8(1)
                end
            end
        end
    end
end

function _demote_obtuse_corners!(state::DeviceContourState,
                                 dev::AbstractDevice=CPU())
    total_nodes = length(state.x)
    total_nodes == 0 && return state
    @_ka_launch dev total_nodes _demote_obtuse_state_corners_kernel!(
        state.corners, state.x, state.y, state.wrapx, state.wrapy,
        state.offsets, state.lengths, state.contour_of_node,
        state.local_index, total_nodes)
    return state
end

function _promote_high_curvature_corners!(state::DeviceContourState{T}, δ,
                                          dev::AbstractDevice=CPU()) where {T}
    total_nodes = length(state.x)
    out_corners = device_zeros(dev, UInt8, total_nodes)
    ncontours = length(state.lengths)
    if ncontours > 0
        @_ka_launch dev ncontours _promote_high_curvature_state_corners_kernel!(
            out_corners, state.corners, state.x, state.y, state.wrapx,
            state.wrapy, state.offsets, state.lengths, T(δ), ncontours)
    end
    state.corners = out_corners
    return state
end

@kernel function _remesh_input_geometry_kernel!(seg_lengths, signed_curvatures,
                                                abs_curvatures, perimeters,
                                                target_area, x, y, pv, wrapx,
                                                wrapy, offsets, lengths,
                                                contour_of_node, local_index,
                                                corners, total_nodes,
                                                ncontours)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        li = local_index[g]
        off = offsets[ci]
        ax = x[g]
        ay = y[g]
        bx = li < lengths[ci] ? x[g + 1] : x[off] + wrapx[ci]
        by = li < lengths[ci] ? y[g + 1] : y[off] + wrapy[ci]
        dx = bx - ax
        dy = by - ay
        seg_lengths[g] = sqrt(dx * dx + dy * dy)
        κ = _flat_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                         lengths, corners, ci, li)
        signed_curvatures[g] = κ
        abs_curvatures[g] = abs(κ)
    end

    if g <= ncontours
        ci = g
        off = offsets[ci]
        n = lengths[ci]
        perimeter = zero(eltype(perimeters))
        area2 = zero(eltype(perimeters))
        @inbounds for li in 1:n
            gi = off + li - 1
            nx = li < n ? x[gi + 1] : x[off] + wrapx[ci]
            ny = li < n ? y[gi + 1] : y[off] + wrapy[ci]
            dx = nx - x[gi]
            dy = ny - y[gi]
            perimeter += sqrt(dx * dx + dy * dy)
            area2 += x[gi] * ny - nx * y[gi]
        end
        perimeters[ci] = perimeter
        target_area[ci] = area2 / 2
    end
end

@kernel function _remesh_node_density_kernel!(node_density_curvatures, x, y, pv,
                                              wrapx, wrapy, offsets, lengths,
                                              contour_of_node, local_index,
                                              seg_lengths, abs_curvatures,
                                              perimeters, μ, Δ_max,
                                              total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        xj = x[g]
        yj = y[g]
        L = max(perimeters[ci] / eltype(x)(2π), Δ_max)
        d2_floor = eps(L) * max(one(L), L)^2
        numerator = zero(L)
        denominator = zero(L)

        @inbounds for sg in 1:total_nodes
            sc = contour_of_node[sg]
            sli = local_index[sg]
            off = offsets[sc]
            ei = seg_lengths[sg]
            ei <= eps(ei) && continue
            mx = sli < lengths[sc] ? (x[sg] + x[sg + 1]) / 2 :
                 (x[sg] + x[off] + wrapx[sc]) / 2
            my = sli < lengths[sc] ? (y[sg] + y[sg + 1]) / 2 :
                 (y[sg] + y[off] + wrapy[sc]) / 2
            dx = xj - mx
            dy = yj - my
            d2 = max(dx * dx + dy * dy, d2_floor)
            weight = ei * abs(pv[sc]) / d2
            denominator += weight
            numerator += weight * abs_curvatures[sg]
        end

        K_j = denominator <= eps(denominator) ? zero(denominator) : numerator / denominator
        α = eltype(x)(2) / eltype(x)(3)
        sqrt2 = sqrt(eltype(x)(2))
        node_density_curvatures[g] = inv(μ * L) * (K_j * L)^α + sqrt2 * K_j
    end
end

@kernel function _remesh_raw_density_kernel!(raw_densities, x, offsets, lengths,
                                             contour_of_node, local_index,
                                             node_density_curvatures, δ,
                                             total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        li = local_index[g]
        next_g = li < lengths[ci] ? g + 1 : offsets[ci]
        sqrt2 = sqrt(eltype(x)(2))
        κ̃ = (node_density_curvatures[g] + node_density_curvatures[next_g]) / 2
        raw_densities[g] = κ̃ <= eps(κ̃) ? zero(κ̃) :
                           κ̃ / (one(κ̃) + δ * κ̃ / sqrt2)
    end
end

@kernel function _remesh_density_scale_kernel!(density_scale, x, offsets,
                                               lengths, seg_lengths,
                                               raw_densities, perimeters,
                                               μ, Δ_max, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        weighted = zero(eltype(x))
        @inbounds for li in 1:n
            gi = off + li - 1
            weighted += raw_densities[gi] * seg_lengths[gi]
        end

        min_density = one(eltype(x)) / Δ_max
        target_intervals = _flat_target_interval_count(perimeters[ci], n, μ, Δ_max)
        density_scale[ci] = weighted <= eps(weighted) ?
                            min_density :
                            eltype(x)(target_intervals) / weighted
    end
end

@kernel function _remesh_measure_kernel!(densities, measure_start,
                                         q_measure, out_lengths,
                                         remesh_mode,
                                         out_pv, out_wrapx, out_wrapy,
                                         raw_densities, density_scale,
                                         corners,
                                         seg_lengths, perimeters,
                                         in_pv, in_wrapx, in_wrapy,
                                         offsets, lengths, μ, Δ_max,
                                         ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = offsets[ci]
        n = lengths[ci]
        min_density = one(eltype(densities)) / Δ_max
        max_density = one(eltype(densities)) / μ
        measure = zero(eltype(densities))
        @inbounds for li in 1:n
            g = off + li - 1
            density = clamp(raw_densities[g] * density_scale[ci],
                            min_density, max_density)
            densities[g] = density
            measure_start[g] = measure
            measure += seg_lengths[g] * density
        end
        q_measure[ci] = measure
        out_pv[ci] = in_pv[ci]
        out_wrapx[ci] = in_wrapx[ci]
        out_wrapy[ci] = in_wrapy[ci]

        corner_count = 0
        @inbounds for li in 1:n
            corner_count += Int(!iszero(corners[off + li - 1]))
        end

        fixed_corners = corner_count > 0 && iszero(in_wrapx[ci]) && iszero(in_wrapy[ci])
        if fixed_corners
            total_out = 0
            first_corner = 0
            prev_corner = 0
            @inbounds for li in 1:n
                if !iszero(corners[off + li - 1])
                    first_corner == 0 && (first_corner = li)
                    if prev_corner != 0
                        span_len = zero(eltype(densities))
                        span_measure = zero(eltype(densities))
                        span_segments = 0
                        seg = prev_corner
                        while seg != li
                            gi = off + seg - 1
                            span_len += seg_lengths[gi]
                            span_measure += seg_lengths[gi] * densities[gi]
                            span_segments += 1
                            seg = seg < n ? seg + 1 : 1
                        end
                        total_out += _flat_fixed_span_interval_count(
                            span_len, span_measure, span_segments, μ, Δ_max)
                    end
                    prev_corner = li
                end
            end
            if prev_corner != 0
                span_len = zero(eltype(densities))
                span_measure = zero(eltype(densities))
                span_segments = 0
                seg = prev_corner
                while seg != first_corner
                    gi = off + seg - 1
                    span_len += seg_lengths[gi]
                    span_measure += seg_lengths[gi] * densities[gi]
                    span_segments += 1
                    seg = seg < n ? seg + 1 : 1
                end
                if prev_corner == first_corner
                    @inbounds for li in 1:n
                        gi = off + li - 1
                        span_len += seg_lengths[gi]
                        span_measure += seg_lengths[gi] * densities[gi]
                    end
                    span_segments = n
                end
                total_out += _flat_fixed_span_interval_count(
                    span_len, span_measure, span_segments, μ, Δ_max)
            end

            if total_out >= 3
                remesh_mode[ci] = UInt8(1)
                out_lengths[ci] = total_out
            else
                remesh_mode[ci] = UInt8(2)
                out_lengths[ci] = n
            end
        else
            remesh_mode[ci] = UInt8(0)
            out_lengths[ci] = _flat_closed_remesh_interval_count(perimeters[ci],
                                                                 measure, μ, Δ_max)
        end
    end
end

@inline function _flat_cubic_point_xy(ax, ay, bx, by, κa, κb, p)
    tx = bx - ax
    ty = by - ay
    e = sqrt(tx * tx + ty * ty)
    e <= eps(e) && return ax, ay
    nx = -ty
    ny = tx
    α = -e * (2 * κa + κb) / 6
    β = e * κa / 2
    γ = e * (κb - κa) / 6
    η = p * (α + p * (β + p * γ))
    return ax + p * tx + η * nx, ay + p * ty + η * ny
end

@kernel function _materialize_remesh_outputs_kernel!(out_x, out_y, out_corners,
                                                     out_offsets, out_lengths,
                                                     out_node_contour, x, y,
                                                     wrapx, wrapy, offsets,
                                                     lengths, seg_lengths,
                                                     signed_curvatures,
                                                     densities, measure_start,
                                                     q_measure, remesh_mode,
                                                     corners, μ, Δ_max,
                                                     total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        ci = out_node_contour[g]
        out_local = g - out_offsets[ci] + 1
        off = offsets[ci]
        n = lengths[ci]
        ox = zero(eltype(out_x))
        oy = zero(eltype(out_y))
        corner = UInt8(0)
        written = false

        if remesh_mode[ci] == UInt8(2)
            in_g = off + out_local - 1
            ox = x[in_g]
            oy = y[in_g]
            corner = corners[in_g]
            written = true
        elseif remesh_mode[ci] == UInt8(1)
            remaining = out_local
            first_corner = 0
            prev_corner = 0
            chosen_start = 0
            chosen_intervals = 0
            chosen_segments = 0
            chosen_measure = zero(eltype(out_x))

            @inbounds for li in 1:n
                if !iszero(corners[off + li - 1])
                    first_corner == 0 && (first_corner = li)
                    if prev_corner != 0 && chosen_start == 0
                        span_len = zero(eltype(out_x))
                        span_measure = zero(eltype(out_x))
                        span_segments = 0
                        seg = prev_corner
                        while seg != li
                            gi = off + seg - 1
                            span_len += seg_lengths[gi]
                            span_measure += seg_lengths[gi] * densities[gi]
                            span_segments += 1
                            seg = seg < n ? seg + 1 : 1
                        end
                        n_intervals = _flat_fixed_span_interval_count(
                            span_len, span_measure, span_segments, μ, Δ_max)
                        if remaining > n_intervals
                            remaining -= n_intervals
                        else
                            chosen_start = prev_corner
                            chosen_intervals = n_intervals
                            chosen_segments = span_segments
                            chosen_measure = span_measure
                        end
                    end
                    prev_corner = li
                end
            end

            if chosen_start == 0
                span_len = zero(eltype(out_x))
                span_measure = zero(eltype(out_x))
                span_segments = 0
                seg = prev_corner
                while seg != first_corner
                    gi = off + seg - 1
                    span_len += seg_lengths[gi]
                    span_measure += seg_lengths[gi] * densities[gi]
                    span_segments += 1
                    seg = seg < n ? seg + 1 : 1
                end
                if prev_corner == first_corner
                    @inbounds for li in 1:n
                        gi = off + li - 1
                        span_len += seg_lengths[gi]
                        span_measure += seg_lengths[gi] * densities[gi]
                    end
                    span_segments = n
                end
                chosen_start = prev_corner
                chosen_intervals = _flat_fixed_span_interval_count(
                    span_len, span_measure, span_segments, μ, Δ_max)
                chosen_segments = span_segments
                chosen_measure = span_measure
            end

            if remaining == 1
                in_g = off + chosen_start - 1
                ox = x[in_g]
                oy = y[in_g]
                corner = UInt8(1)
                written = true
            else
                s_target = chosen_measure * (remaining - 1) / chosen_intervals
                span_measure = zero(eltype(out_x))
                seg = chosen_start
                @inbounds for count in 1:chosen_segments
                    in_g = off + seg - 1
                    next_measure = span_measure + seg_lengths[in_g] * densities[in_g]
                    if s_target <= next_measure || count == chosen_segments
                        seg_measure = next_measure - span_measure
                        p = seg_measure <= eps(seg_measure) ? zero(seg_measure) :
                            (s_target - span_measure) / seg_measure
                        p = min(max(p, zero(p)), one(p))
                        next_li = seg < n ? seg + 1 : 1
                        next_g = off + next_li - 1
                        ox, oy = _flat_cubic_point_xy(x[in_g], y[in_g],
                                                      x[next_g], y[next_g],
                                                      signed_curvatures[in_g],
                                                      signed_curvatures[next_g], p)
                        corner = UInt8(0)
                        written = true
                        break
                    end
                    span_measure = next_measure
                    seg = seg < n ? seg + 1 : 1
                end
            end
        end

        if !written
            nout = out_lengths[ci]
            s_target = q_measure[ci] * (out_local - 1) / nout
            seg = 1
            @inbounds for li in 1:n
                gi = off + li - 1
                next_measure = measure_start[gi] + seg_lengths[gi] * densities[gi]
                if s_target <= next_measure || li == n
                    seg = li
                    break
                end
            end

            in_g = off + seg - 1
            seg_measure = seg_lengths[in_g] * densities[in_g]
            p = seg_measure <= eps(seg_measure) ? zero(seg_measure) :
                (s_target - measure_start[in_g]) / seg_measure
            p = min(max(p, zero(p)), one(p))
            next_g = seg < n ? in_g + 1 : off
            ax = x[in_g]
            ay = y[in_g]
            bx = seg < n ? x[next_g] : x[off] + wrapx[ci]
            by = seg < n ? y[next_g] : y[off] + wrapy[ci]
            ox, oy = _flat_cubic_point_xy(ax, ay, bx, by,
                                          signed_curvatures[in_g],
                                          signed_curvatures[next_g], p)
            corner = UInt8(0)
        end

        out_x[g] = ox
        out_y[g] = oy
        out_corners[g] = corner
    end
end

@kernel function _remesh_output_moments_kernel!(out_area, out_centroid_x,
                                                out_centroid_y, out_x, out_y,
                                                out_offsets, out_lengths,
                                                ncontours)
    ci = @index(Global)
    if ci <= ncontours
        off = out_offsets[ci]
        n = out_lengths[ci]
        area2 = zero(eltype(out_area))
        cx_num = zero(eltype(out_area))
        cy_num = zero(eltype(out_area))
        sx = zero(eltype(out_area))
        sy = zero(eltype(out_area))
        @inbounds for li in 1:n
            g = off + li - 1
            ng = li < n ? g + 1 : off
            cross = out_x[g] * out_y[ng] - out_x[ng] * out_y[g]
            area2 += cross
            cx_num += (out_x[g] + out_x[ng]) * cross
            cy_num += (out_y[g] + out_y[ng]) * cross
            sx += out_x[g]
            sy += out_y[g]
        end

        area = area2 / 2
        out_area[ci] = area
        if abs(area) <= eps(area)
            out_centroid_x[ci] = sx / n
            out_centroid_y[ci] = sy / n
        else
            inv6A = one(area) / (6 * area)
            out_centroid_x[ci] = cx_num * inv6A
            out_centroid_y[ci] = cy_num * inv6A
        end
    end
end

@kernel function _preserve_remesh_area_kernel!(out_x, out_y, out_node_contour,
                                               out_area, out_centroid_x,
                                               out_centroid_y, target_area,
                                               wrapx, wrapy, remesh_mode,
                                               total_out_nodes)
    g = @index(Global)
    if g <= total_out_nodes
        ci = out_node_contour[g]
        if remesh_mode[ci] == UInt8(0) && iszero(wrapx[ci]) && iszero(wrapy[ci])
            target = target_area[ci]
            current = out_area[ci]
            if abs(target) > eps(target) && abs(current) > eps(current) &&
               ((target > zero(target)) == (current > zero(current)))
                scale = sqrt(abs(target / current))
                if abs(scale - one(scale)) > sqrt(eps(scale))
                    cx = out_centroid_x[ci]
                    cy = out_centroid_y[ci]
                    out_x[g] = cx + scale * (out_x[g] - cx)
                    out_y[g] = cy + scale * (out_y[g] - cy)
                end
            end
        end
    end
end

function _device_remesh_supported(contours::Vector{PVContour{T}}) where {T}
    @inbounds for c in contours
        nnodes(c) < 3 && return false
    end
    return true
end

function _device_remesh_outputs(flat::FlatContourTopology{T},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    ncontours = _flat_ncontours(flat)
    total_nodes = _flat_nnodes(flat)
    if ncontours == 0 || total_nodes == 0
        empty_t = device_zeros(dev, T, 0)
        empty_i = device_zeros(dev, Int, 0)
        empty_b = device_zeros(dev, UInt8, 0)
        return DeviceRewriteOutputs(empty_t, empty_t, empty_t, empty_t,
                                    empty_t, empty_i, empty_i, empty_b)
    end

    seg_lengths = device_zeros(dev, T, total_nodes)
    signed_curvatures = device_zeros(dev, T, total_nodes)
    abs_curvatures = device_zeros(dev, T, total_nodes)
    perimeters = device_zeros(dev, T, ncontours)
    target_area = device_zeros(dev, T, ncontours)
    @_ka_launch dev max(total_nodes, ncontours) _remesh_input_geometry_kernel!(
        seg_lengths, signed_curvatures, abs_curvatures, perimeters,
        target_area, flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy,
        flat.offsets, flat.lengths, flat.contour_of_node, flat.local_index,
        flat.corners, total_nodes, ncontours)

    node_density_curvatures = device_zeros(dev, T, total_nodes)
    @_ka_launch dev total_nodes _remesh_node_density_kernel!(
        node_density_curvatures, flat.x, flat.y, flat.pv, flat.wrapx,
        flat.wrapy, flat.offsets, flat.lengths, flat.contour_of_node,
        flat.local_index, seg_lengths, abs_curvatures, perimeters,
        T(params.μ), T(params.Δ_max), total_nodes)

    raw_densities = device_zeros(dev, T, total_nodes)
    density_scale = device_zeros(dev, T, ncontours)
    @_ka_launch dev total_nodes _remesh_raw_density_kernel!(
        raw_densities, flat.x, flat.offsets, flat.lengths,
        flat.contour_of_node, flat.local_index, node_density_curvatures,
        T(params.δ), total_nodes)
    @_ka_launch dev ncontours _remesh_density_scale_kernel!(
        density_scale, flat.x, flat.offsets, flat.lengths, seg_lengths,
        raw_densities, perimeters, T(params.μ), T(params.Δ_max), ncontours)

    densities = device_zeros(dev, T, total_nodes)
    measure_start = device_zeros(dev, T, total_nodes)
    q_measure = device_zeros(dev, T, ncontours)
    out_lengths = device_zeros(dev, Int, ncontours)
    remesh_mode = device_zeros(dev, UInt8, ncontours)
    out_pv = device_zeros(dev, T, ncontours)
    out_wrapx = device_zeros(dev, T, ncontours)
    out_wrapy = device_zeros(dev, T, ncontours)
    @_ka_launch dev ncontours _remesh_measure_kernel!(
        densities, measure_start, q_measure, out_lengths, remesh_mode,
        out_pv, out_wrapx, out_wrapy, raw_densities, density_scale,
        flat.corners, seg_lengths, perimeters, flat.pv, flat.wrapx,
        flat.wrapy, flat.offsets, flat.lengths, T(params.μ),
        T(params.Δ_max), ncontours)

    out_offsets = device_zeros(dev, Int, ncontours)
    total_store = device_zeros(dev, Int, 1)
    @_ka_launch dev ncontours _prefix_lengths_kernel!(
        out_offsets, total_store, out_lengths, ncontours)
    total_out_nodes = to_cpu(total_store)[1]

    out_node_contour = device_zeros(dev, Int, total_out_nodes)
    @_ka_launch dev ncontours _out_node_contour_kernel!(
        out_node_contour, out_offsets, out_lengths, ncontours)

    out_x = device_zeros(dev, T, total_out_nodes)
    out_y = device_zeros(dev, T, total_out_nodes)
    out_corners = device_zeros(dev, UInt8, total_out_nodes)
    @_ka_launch dev total_out_nodes _materialize_remesh_outputs_kernel!(
        out_x, out_y, out_corners, out_offsets, out_lengths,
        out_node_contour, flat.x, flat.y, flat.wrapx, flat.wrapy,
        flat.offsets, flat.lengths, seg_lengths, signed_curvatures,
        densities, measure_start, q_measure, remesh_mode, flat.corners,
        T(params.μ), T(params.Δ_max), total_out_nodes)

    out_area = device_zeros(dev, T, ncontours)
    out_centroid_x = device_zeros(dev, T, ncontours)
    out_centroid_y = device_zeros(dev, T, ncontours)
    @_ka_launch dev ncontours _remesh_output_moments_kernel!(
        out_area, out_centroid_x, out_centroid_y, out_x, out_y,
        out_offsets, out_lengths, ncontours)
    @_ka_launch dev total_out_nodes _preserve_remesh_area_kernel!(
        out_x, out_y, out_node_contour, out_area, out_centroid_x,
        out_centroid_y, target_area, out_wrapx, out_wrapy, remesh_mode,
        total_out_nodes)

    return DeviceRewriteOutputs(out_x, out_y, out_pv, out_wrapx, out_wrapy,
                                out_offsets, out_lengths, out_corners)
end

function _device_remesh_outputs(contours::Vector{PVContour{T}},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    _device_remesh_supported(contours) || return nothing
    return _device_remesh_outputs(_pack_flat_topology(contours, dev), params, dev)
end

function _device_remesh_outputs(state::DeviceContourState{T},
                                params::SurgeryParams,
                                dev::AbstractDevice=CPU()) where {T}
    return _device_remesh_outputs(_flat_topology(state, dev), params, dev)
end

function _device_remesh_contours(contours::Vector{PVContour{T}},
                                 params::SurgeryParams,
                                 dev::AbstractDevice=CPU()) where {T}
    outputs = _device_remesh_outputs(contours, params, dev)
    outputs === nothing && return nothing
    return _unpack_rewrite_outputs(outputs)
end

function _device_full_rewrite_output_layout(flat::FlatContourTopology{T},
                                            plan::DeviceTopologyRewritePlan,
                                            dev::AbstractDevice=CPU()) where {T}
    ncontours = _flat_ncontours(flat)
    npairs = length(plan.op)

    replacement_op = device_zeros(dev, Int, ncontours)
    deleted = device_zeros(dev, UInt8, ncontours)
    if npairs > 0
        @_ka_launch dev npairs _full_rewrite_roles_kernel!(
            replacement_op, deleted, plan.ci, plan.cj, plan.op, plan.valid, npairs)
    end

    main_keep = device_zeros(dev, UInt8, ncontours)
    extra_keep = device_zeros(dev, UInt8, npairs)
    if max(ncontours, npairs) > 0
        @_ka_launch dev max(ncontours, npairs) _full_rewrite_keep_flags_kernel!(
            main_keep, extra_keep, replacement_op, deleted, plan.valid,
            plan.out_count, ncontours, npairs)
    end

    main_slot = device_zeros(dev, Int, ncontours)
    extra_slot = device_zeros(dev, Int, npairs)
    main_count = device_zeros(dev, Int, 1)
    extra_count = device_zeros(dev, Int, 1)
    if ncontours > 0
        @_ka_launch dev ncontours _prefix_u8_kernel!(
            main_slot, main_count, main_keep, ncontours)
    end
    if npairs > 0
        @_ka_launch dev npairs _prefix_u8_kernel!(
            extra_slot, extra_count, extra_keep, npairs)
    end

    nmain = ncontours == 0 ? 0 : to_cpu(main_count)[1]
    nextra = npairs == 0 ? 0 : to_cpu(extra_count)[1]
    nout = nmain + nextra

    offsets = device_zeros(dev, Int, nout)
    lengths = device_zeros(dev, Int, nout)
    op_index = device_zeros(dev, Int, nout)
    source_contour = device_zeros(dev, Int, nout)
    part = device_zeros(dev, Int, nout)
    pv = device_zeros(dev, T, nout)
    wrapx = device_zeros(dev, T, nout)
    wrapy = device_zeros(dev, T, nout)

    if ncontours > 0
        @_ka_launch dev ncontours _full_rewrite_fill_main_layout_kernel!(
            lengths, op_index, source_contour, part, pv, wrapx, wrapy,
            main_keep, main_slot, replacement_op, flat.lengths, flat.pv,
            flat.wrapx, flat.wrapy, plan.out_len1, ncontours)
    end
    if npairs > 0
        @_ka_launch dev npairs _full_rewrite_fill_extra_layout_kernel!(
            lengths, op_index, source_contour, part, pv, wrapx, wrapy,
            extra_keep, extra_slot, main_count, plan.ci, plan.out_len2,
            flat.pv, flat.wrapx, flat.wrapy, npairs)
    end

    total_store = device_zeros(dev, Int, 1)
    if nout > 0
        @_ka_launch dev nout _prefix_lengths_kernel!(offsets, total_store, lengths, nout)
    end
    total_nodes = nout == 0 ? 0 : to_cpu(total_store)[1]

    out_node_contour = device_zeros(dev, Int, total_nodes)
    if nout > 0 && total_nodes > 0
        @_ka_launch dev nout _out_node_contour_kernel!(out_node_contour, offsets, lengths, nout)
    end

    return (offsets=offsets,
            lengths=lengths,
            op_index=op_index,
            source_contour=source_contour,
            part=part,
            pv=pv,
            wrapx=wrapx,
            wrapy=wrapy,
            out_node_contour=out_node_contour,
            total_nodes=total_nodes)
end

function _device_full_rewrite_output_layout(contours::Vector{PVContour{T}},
                                            plan::DeviceTopologyRewritePlan,
                                            dev::AbstractDevice=CPU()) where {T}
    return _device_full_rewrite_output_layout(_pack_flat_topology(contours, dev),
                                              plan, dev)
end

function _device_full_rewrite_output_layout(state::DeviceContourState{T},
                                            plan::DeviceTopologyRewritePlan,
                                            dev::AbstractDevice=CPU()) where {T}
    return _device_full_rewrite_output_layout(_flat_topology(state, dev),
                                              plan, dev)
end

function _materialize_rewrite_outputs(flat::FlatContourTopology{T},
                                      plan::DeviceTopologyRewritePlan,
                                      layout,
                                      dev::AbstractDevice=CPU()) where {T}
    out_x = device_zeros(dev, T, layout.total_nodes)
    out_y = device_zeros(dev, T, layout.total_nodes)
    out_corners = device_zeros(dev, UInt8, layout.total_nodes)

    if layout.total_nodes > 0
        @_ka_launch dev layout.total_nodes _materialize_rewrite_outputs_kernel!(
            out_x, out_y, out_corners, layout.offsets, layout.lengths,
            layout.out_node_contour, layout.op_index,
            layout.source_contour, layout.part, plan.ci, plan.cj,
            plan.op, plan.valid, plan.node_from_first, plan.node_idx,
            plan.seg_idx, plan.inserted_idx, plan.split_reverse1,
            plan.split_reverse2, plan.merge_reverse_second, plan.stitch_x,
            plan.stitch_y, flat.x, flat.y, flat.corners, flat.offsets,
            flat.lengths, layout.total_nodes)
    end

    return DeviceRewriteOutputs(out_x, out_y, layout.pv, layout.wrapx,
                                layout.wrapy, layout.offsets, layout.lengths,
                                out_corners)
end

function _materialize_rewrite_outputs(contours::Vector{PVContour{T}},
                                      plan::DeviceTopologyRewritePlan,
                                      layout,
                                      dev::AbstractDevice=CPU()) where {T}
    return _materialize_rewrite_outputs(_pack_flat_topology(contours, dev),
                                        plan, layout, dev)
end

function _materialize_rewrite_outputs(state::DeviceContourState{T},
                                      plan::DeviceTopologyRewritePlan,
                                      layout,
                                      dev::AbstractDevice=CPU()) where {T}
    return _materialize_rewrite_outputs(_flat_topology(state, dev),
                                        plan, layout, dev)
end

function _device_materialize_rewrite_outputs(contours::Vector{PVContour{T}},
                                             selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                             dev::AbstractDevice=CPU()) where {T}
    plan = _device_topology_rewrite_plan(contours, selected_pairs, dev)
    layout = _rewrite_output_layout(contours, plan, dev)
    return _materialize_rewrite_outputs(contours, plan, layout, dev)
end

function _device_materialize_full_rewrite_outputs(contours::Vector{PVContour{T}},
                                                  selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                  dev::AbstractDevice=CPU()) where {T}
    plan = _device_topology_rewrite_plan(contours, selected_pairs, dev)
    layout = _device_full_rewrite_output_layout(contours, plan, dev)
    return _materialize_rewrite_outputs(contours, plan, layout, dev)
end

function _device_materialize_full_rewrite_outputs(contours::Vector{PVContour{T}},
                                                  selected_pairs::DeviceClosePairCandidates,
                                                  dev::AbstractDevice=CPU()) where {T}
    plan = _device_topology_rewrite_plan(contours, selected_pairs, dev)
    layout = _device_full_rewrite_output_layout(contours, plan, dev)
    return _materialize_rewrite_outputs(contours, plan, layout, dev)
end

function _device_materialize_full_rewrite_outputs(state::DeviceContourState{T},
                                                  selected_pairs::DeviceClosePairCandidates,
                                                  dev::AbstractDevice=CPU()) where {T}
    plan = _device_topology_rewrite_plan(state, selected_pairs, dev)
    layout = _device_full_rewrite_output_layout(state, plan, dev)
    return _materialize_rewrite_outputs(state, plan, layout, dev)
end

function _device_materialize_full_rewrite_outputs(state::DeviceContourState{T},
                                                  selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                  dev::AbstractDevice=CPU()) where {T}
    packed = _pack_close_pair_candidates(selected_pairs, dev)
    return _device_materialize_full_rewrite_outputs(state, packed, dev)
end

function _device_rewrite_contours(contours::Vector{PVContour{T}},
                                  selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                  dev::AbstractDevice=CPU()) where {T}
    return _unpack_rewrite_outputs(
        _device_materialize_full_rewrite_outputs(contours, selected_pairs, dev))
end

function _device_rewrite_contours(contours::Vector{PVContour{T}},
                                  selected_pairs::DeviceClosePairCandidates,
                                  dev::AbstractDevice=CPU()) where {T}
    return _unpack_rewrite_outputs(
        _device_materialize_full_rewrite_outputs(contours, selected_pairs, dev))
end

function _device_rewrite_state!(state::DeviceContourState{T},
                                selected_pairs::Union{DeviceClosePairCandidates,
                                                      Vector{Tuple{Int,Int,Int,Int}}},
                                dev::AbstractDevice=CPU()) where {T}
    outputs = _device_materialize_full_rewrite_outputs(state, selected_pairs, dev)
    return _replace_device_state!(state, outputs, dev)
end

function _device_remesh_state!(state::DeviceContourState{T},
                               params::SurgeryParams,
                               dev::AbstractDevice=CPU()) where {T}
    outputs = _device_remesh_outputs(state, params, dev)
    return _replace_device_state!(state, outputs, dev)
end

function _device_admissible_close_segments(contours::Vector{PVContour{T}}, δ,
                                           domain::UnboundedDomain,
                                           dev::AbstractDevice=CPU()) where {T}
    return _unpack_close_pair_candidates(
        _device_admissible_close_segment_buffer(contours, δ, domain, dev))
end

function _device_reconnect!(contours::Vector{PVContour{T}},
                            close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                            dev::AbstractDevice=CPU()) where {T}
    selected_pairs = _device_select_reconnection_pair_buffer(contours, close_pairs, dev)
    length(selected_pairs.ci) == 0 && return false
    rewritten = _device_rewrite_contours(contours, selected_pairs, dev)
    empty!(contours)
    append!(contours, rewritten)
    return true
end

function _device_reconnect!(contours::Vector{PVContour{T}},
                            close_pairs::DeviceClosePairCandidates,
                            dev::AbstractDevice=CPU()) where {T}
    selected_pairs = _device_select_reconnection_pair_buffer(contours, close_pairs, dev)
    length(selected_pairs.ci) == 0 && return false
    rewritten = _device_rewrite_contours(contours, selected_pairs, dev)
    empty!(contours)
    append!(contours, rewritten)
    return true
end

function _device_reconnect!(state::DeviceContourState{T},
                            close_pairs::DeviceClosePairCandidates,
                            dev::AbstractDevice=CPU()) where {T}
    selected_pairs = _device_select_reconnection_pair_buffer(state, close_pairs, dev)
    length(selected_pairs.ci) == 0 && return false
    _device_rewrite_state!(state, selected_pairs, dev)
    return true
end

function _device_reconnect!(state::DeviceContourState{T},
                            close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                            dev::AbstractDevice=CPU()) where {T}
    return _device_reconnect!(state, _pack_close_pair_candidates(close_pairs, dev), dev)
end

function _device_reconnect_once!(contours::Vector{PVContour{T}}, δ,
                                 domain::UnboundedDomain,
                                 dev::AbstractDevice=CPU()) where {T}
    close_pairs = _device_admissible_close_segment_buffer(contours, δ, domain, dev)
    length(close_pairs.ci) == 0 && return false
    return _device_reconnect!(contours, close_pairs, dev)
end

function _device_reconnect_once!(state::DeviceContourState{T}, δ,
                                 domain::UnboundedDomain,
                                 dev::AbstractDevice=CPU()) where {T}
    close_pairs = _device_admissible_close_segment_buffer(state, δ, domain, dev)
    length(close_pairs.ci) == 0 && return false
    return _device_reconnect!(state, close_pairs, dev)
end

function _unpack_rewrite_outputs(outputs::DeviceRewriteOutputs{T}) where {T}
    x = to_cpu(outputs.x)
    y = to_cpu(outputs.y)
    pv = to_cpu(outputs.pv)
    wrapx = to_cpu(outputs.wrapx)
    wrapy = to_cpu(outputs.wrapy)
    offsets = to_cpu(outputs.offsets)
    lengths = to_cpu(outputs.lengths)
    corners = to_cpu(outputs.corners)

    out = PVContour{T}[]
    @inbounds for ci in eachindex(lengths)
        off = offsets[ci]
        len = lengths[ci]
        nodes = Vector{SVector{2,T}}(undef, len)
        corner_flags = Vector{Bool}(undef, len)
        for li in 1:len
            g = off + li - 1
            nodes[li] = SVector{2,T}(x[g], y[g])
            corner_flags[li] = !iszero(corners[g])
        end
        push!(out, PVContour(nodes, pv[ci], SVector{2,T}(wrapx[ci], wrapy[ci]), corner_flags))
    end
    return out
end

function _device_surgery_reconnect_loop!(contours::Vector{PVContour{T}},
                                         params::SurgeryParams,
                                         domain::UnboundedDomain,
                                         dev::AbstractDevice,
                                         cleanup_reconnect_artifacts!) where {T}
    reconnected = false
    max_reconnect_iter = 100
    stall_warning_pairs = 100
    prev_n_pairs = typemax(Int)
    min_n_pairs = typemax(Int)
    stall_count = 0
    no_improve_count = 0

    for iter in 1:max_reconnect_iter
        close_pairs = _device_admissible_close_segment_buffer(contours, params.δ, domain, dev)
        length(close_pairs.ci) == 0 && break
        n_pairs = length(close_pairs.ci)

        if n_pairs > prev_n_pairs
            stall_count += 1
        else
            stall_count = 0
        end
        if n_pairs < min_n_pairs
            min_n_pairs = n_pairs
            no_improve_count = 0
        else
            no_improve_count += 1
        end

        if stall_count >= 3 || no_improve_count >= 6
            if reconnected
                cleanup_reconnect_artifacts!()
                _device_remove_filaments!(contours, params, dev)
                close_pairs = _device_admissible_close_segment_buffer(contours, params.δ, domain, dev)
                length(close_pairs.ci) == 0 && break
                remeshed_n_pairs = length(close_pairs.ci)
                if remeshed_n_pairs < n_pairs
                    prev_n_pairs = remeshed_n_pairs
                    min_n_pairs = min(min_n_pairs, remeshed_n_pairs)
                    stall_count = 0
                    no_improve_count = 0
                    continue
                end
                n_pairs = remeshed_n_pairs
            end
            if n_pairs >= stall_warning_pairs
                @warn "surgery!: device reconnection stalled ($n_pairs close pairs, min seen: $min_n_pairs) — stopping early"
            end
            break
        end

        prev_n_pairs = n_pairs
        _device_reconnect!(contours, close_pairs, dev) || break
        reconnected = true
        _device_remove_filaments!(contours, params, dev)
        cleanup_reconnect_artifacts!()
        _device_remove_filaments!(contours, params, dev)
        if iter == max_reconnect_iter
            @warn "surgery!: device reconnection iteration limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
        end
    end
    return reconnected
end

function _device_surgery_reconnect_loop!(state::DeviceContourState{T},
                                         params::SurgeryParams,
                                         domain::UnboundedDomain,
                                         dev::AbstractDevice,
                                         cleanup_reconnect_artifacts!) where {T}
    reconnected = false
    max_reconnect_iter = 100
    stall_warning_pairs = 100
    prev_n_pairs = typemax(Int)
    min_n_pairs = typemax(Int)
    stall_count = 0
    no_improve_count = 0

    for iter in 1:max_reconnect_iter
        close_pairs = _device_admissible_close_segment_buffer(state, params.δ, domain, dev)
        length(close_pairs.ci) == 0 && break
        n_pairs = length(close_pairs.ci)

        if n_pairs > prev_n_pairs
            stall_count += 1
        else
            stall_count = 0
        end
        if n_pairs < min_n_pairs
            min_n_pairs = n_pairs
            no_improve_count = 0
        else
            no_improve_count += 1
        end

        if stall_count >= 3 || no_improve_count >= 6
            if reconnected
                cleanup_reconnect_artifacts!()
                _device_remove_filaments!(state, params, dev)
                close_pairs = _device_admissible_close_segment_buffer(state, params.δ, domain, dev)
                length(close_pairs.ci) == 0 && break
                remeshed_n_pairs = length(close_pairs.ci)
                if remeshed_n_pairs < n_pairs
                    prev_n_pairs = remeshed_n_pairs
                    min_n_pairs = min(min_n_pairs, remeshed_n_pairs)
                    stall_count = 0
                    no_improve_count = 0
                    continue
                end
                n_pairs = remeshed_n_pairs
            end
            if n_pairs >= stall_warning_pairs
                @warn "surgery!: device reconnection stalled ($n_pairs close pairs, min seen: $min_n_pairs) — stopping early"
            end
            break
        end

        prev_n_pairs = n_pairs
        _device_reconnect!(state, close_pairs, dev) || break
        reconnected = true
        _device_remove_filaments!(state, params, dev)
        cleanup_reconnect_artifacts!()
        _device_remove_filaments!(state, params, dev)
        if iter == max_reconnect_iter
            @warn "surgery!: device reconnection iteration limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
        end
    end
    return reconnected
end

function _remesh_contours_after_surgery!(contours::Vector{PVContour{T}},
                                         params::SurgeryParams,
                                         remesh_buf::Vector{SVector{2,T}},
                                         arc_buf::Vector{T},
                                         vnodes_buf::Vector{SVector{2,T}}) where {T}
    density_sources = copy(contours)
    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params;
                             _buf=remesh_buf,
                             _arc_buf=arc_buf,
                             _vnodes_buf=vnodes_buf,
                             _density_sources=density_sources)
    end
    _demote_obtuse_corners!(contours)
    return contours
end

function _device_remesh_contours_after_surgery!(contours::Vector{PVContour{T}},
                                                params::SurgeryParams,
                                                dev::AbstractDevice,
                                                remesh_buf::Vector{SVector{2,T}},
                                                arc_buf::Vector{T},
                                                vnodes_buf::Vector{SVector{2,T}}) where {T}
    remeshed = _device_remesh_contours(contours, params, dev)
    if remeshed !== nothing
        empty!(contours)
        append!(contours, remeshed)
        _demote_obtuse_corners!(contours)
        return contours
    end

    return _remesh_contours_after_surgery!(contours, params, remesh_buf,
                                           arc_buf, vnodes_buf)
end

function _device_remesh_contours_after_surgery!(state::DeviceContourState{T},
                                                params::SurgeryParams,
                                                dev::AbstractDevice) where {T}
    _device_remesh_state!(state, params, dev)
    _demote_obtuse_corners!(state, dev)
    return state
end

@kernel function _spanning_proximity_flags_kernel!(flags, x, y, wrapx, wrapy,
                                                   offsets, lengths,
                                                   contour_of_node,
                                                   total_nodes, ncontours, δ2)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        if iszero(wrapx[ci]) && iszero(wrapy[ci])
            gx = x[g]
            gy = y[g]
            close = false
            @inbounds for cj in 1:ncontours
                (iszero(wrapx[cj]) && iszero(wrapy[cj])) && continue
                off = offsets[cj]
                n = lengths[cj]
                for li in 1:n
                    h = off + li - 1
                    dx = gx - x[h]
                    dy = gy - y[h]
                    if dx * dx + dy * dy < δ2
                        close = true
                        break
                    end
                end
                close && break
            end
            flags[g] = close ? UInt8(1) : UInt8(0)
        else
            flags[g] = UInt8(0)
        end
    end
end

function _check_spanning_proximity(state::DeviceContourState{T}, δ,
                                   ::UnboundedDomain,
                                   dev::AbstractDevice=CPU()) where {T}
    total_nodes = length(state.x)
    ncontours = length(state.lengths)
    (total_nodes == 0 || ncontours == 0) && return nothing
    flags = device_zeros(dev, UInt8, total_nodes)
    @_ka_launch dev total_nodes _spanning_proximity_flags_kernel!(
        flags, state.x, state.y, state.wrapx, state.wrapy, state.offsets,
        state.lengths, state.contour_of_node, total_nodes, ncontours, T(δ)^2)
    slots = device_zeros(dev, Int, total_nodes)
    count_store = device_zeros(dev, Int, 1)
    @_ka_launch dev total_nodes _prefix_u8_kernel!(slots, count_store, flags, total_nodes)
    nclose = to_cpu(count_store)[1]
    if nclose > 0
        @warn "surgery!: closed contour node within δ of spanning contour — this cannot be resolved by reconnection" δ maxlog=1
    end
    return nothing
end

"""
    _device_surgery_pipeline!(state, params, domain, dev)

Device-resident surgery pipeline for one `DeviceContourState`: filament removal,
corner demotion/promotion, remesh, reconnection loop with artifact cleanup, final
filament sweep, and spanning-proximity check.
"""
function _device_surgery_pipeline!(state::DeviceContourState, params::SurgeryParams,
                                   domain::UnboundedDomain, dev::AbstractDevice)
    _device_remove_filaments!(state, params, dev)
    _demote_obtuse_corners!(state, dev)
    _promote_high_curvature_corners!(state, params.δ, dev)
    _device_remesh_contours_after_surgery!(state, params, dev)

    cleanup_reconnect_artifacts!() =
        _device_remesh_contours_after_surgery!(state, params, dev)

    reconnected = _device_surgery_reconnect_loop!(state, params, domain, dev,
                                                  cleanup_reconnect_artifacts!)
    reconnected && cleanup_reconnect_artifacts!()

    _device_remove_filaments!(state, params, dev)
    _check_spanning_proximity(state, params.δ, domain, dev)
    return state
end

function surgery!(prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},
                                       UnboundedDomain, T, GPU},
                  params::SurgeryParams) where {T}
    _device_surgery_pipeline!(prob.device_state, params, prob.domain, prob.dev)
    return prob
end

function surgery!(::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},
                                   <:PeriodicDomain, T, GPU},
                  ::SurgeryParams) where {T}
    throw(ArgumentError(
        "GPU surgery is device-resident only for unbounded single-layer problems. " *
        "Use dev=CPU() for periodic surgery, or disable surgery for this GPU() problem."))
end

function surgery!(::ContourProblem{K, D, T, GPU}, ::SurgeryParams) where {K, D, T}
    throw(ArgumentError(
        "GPU surgery is not implemented for $(K) on $(D). " *
        "Use dev=CPU() or a supported unbounded single-layer Euler/QG/SQG problem."))
end

"""
    _device_multilayer_surgery!(states, params, domain, dev)

Run the device-resident surgery pipeline independently on each layer's state.
Layers never reconnect across layer boundaries (the CPU multi-layer surgery has
the same per-layer structure).
"""
function _device_multilayer_surgery!(states::NTuple{N, <:DeviceContourState},
                                     params::SurgeryParams,
                                     domain::UnboundedDomain,
                                     dev::AbstractDevice) where {N}
    for ℓ in 1:N
        _device_surgery_pipeline!(states[ℓ], params, domain, dev)
    end
    return states
end

function surgery!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, UnboundedDomain, T, GPU},
                  params::SurgeryParams) where {N, T}
    _device_multilayer_surgery!(prob.device_state, params, prob.domain, prob.dev)
    return prob
end

function surgery!(::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, <:PeriodicDomain, T, GPU},
                  ::SurgeryParams) where {N, T}
    throw(ArgumentError(
        "GPU surgery is device-resident only for unbounded multi-layer problems. " *
        "Use dev=CPU() for periodic surgery, or disable surgery for this GPU() problem."))
end
