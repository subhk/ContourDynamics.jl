# Close-pair detection and reconnection planning: segment-distance predicates,
# candidate generation, the interior-vorticity admissibility test, and
# independent split/merge pair selection.

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

@inline function _flat_wrap_coord(x, L)
    L2 = 2 * L
    return x - floor((x + L) / L2) * L2
end

@inline function _flat_shift_segment_to_image(ax, ay, bx, by, refx, refy,
                                              periodic, Lx, Ly)
    periodic || return ax, ay, bx, by
    midx = (ax + bx) / 2
    midy = (ay + by) / 2
    shiftx = round((refx - midx) / (2 * Lx)) * (2 * Lx)
    shifty = round((refy - midy) / (2 * Ly)) * (2 * Ly)
    return ax + shiftx, ay + shifty, bx + shiftx, by + shifty
end

@inline _flat_surgery_domain(::UnboundedDomain, ::Type{T}) where {T} =
    (false, zero(T), zero(T))
@inline _flat_surgery_domain(domain::PeriodicDomain, ::Type{T}) where {T} =
    (true, T(domain.Lx), T(domain.Ly))

@kernel function _eligible_surgery_segment_flags_kernel!(flags, wrapx, wrapy,
                                                         lengths, active,
                                                         contour_of_node,
                                                         total_nodes)
    g = @index(Global)
    if g <= total_nodes
        ci = contour_of_node[g]
        keep = !iszero(active[ci]) && lengths[ci] >= 3 &&
               iszero(wrapx[ci]) && iszero(wrapy[ci])
        flags[g] = keep ? UInt8(1) : UInt8(0)
    end
end

@kernel function _compact_eligible_surgery_segments_kernel!(eligible, slots,
                                                            flags, total_nodes)
    g = @index(Global)
    if g <= total_nodes && !iszero(flags[g])
        eligible[slots[g]] = g
    end
end

function _device_eligible_surgery_segment_indices(flat::FlatContourTopology,
                                                  dev::AbstractDevice=CPU())
    total_nodes = _flat_nnodes(flat)
    total_nodes == 0 && return device_zeros(dev, Int, 0)

    flags = device_zeros(dev, UInt8, total_nodes)
    @_ka_launch dev total_nodes _eligible_surgery_segment_flags_kernel!(
        flags, flat.wrapx, flat.wrapy, flat.lengths, flat.active,
        flat.contour_of_node, total_nodes)
    slots = device_zeros(dev, Int, total_nodes)
    count_store = device_zeros(dev, Int, 1)
    _device_compact_scan!(slots, count_store, flags, total_nodes, dev)
    neligible = to_cpu(count_store)[1]
    eligible = device_zeros(dev, Int, neligible)
    if neligible > 0
        @_ka_launch dev total_nodes _compact_eligible_surgery_segments_kernel!(
            eligible, slots, flags, total_nodes)
    end
    return eligible
end

@kernel function _close_pair_candidate_kernel!(valid, eligible, x, y, pv, wrapx, wrapy, offsets,
                                               lengths, contour_of_node, local_index,
                                               corners, periodic, Lx, Ly, δ2,
                                               neligible)
    pair_idx = @index(Global)
    npairs = neligible * neligible
    if pair_idx <= npairs
        is_valid = false
        g1 = eligible[((pair_idx - 1) % neligible) + 1]
        g2 = eligible[((pair_idx - 1) ÷ neligible) + 1]
        if g1 < g2
            ci = contour_of_node[g1]
            cj = contour_of_node[g2]
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
                    tol = sqrt(eps(one(δ2))) *
                          max(one(δ2), abs(pv[ci]), abs(pv[cj]))
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

                    if periodic
                        refx = _flat_wrap_coord((ax1 + bx1) / 2, Lx)
                        refy = _flat_wrap_coord((ay1 + by1) / 2, Ly)
                        ax1, ay1, bx1, by1 = _flat_shift_segment_to_image(
                            ax1, ay1, bx1, by1, refx, refy, periodic, Lx, Ly)
                        ax2, ay2, bx2, by2 = _flat_shift_segment_to_image(
                            ax2, ay2, bx2, by2, refx, refy, periodic, Lx, Ly)
                    end

                    d2 = _flat_surgery_contact_distance2(ax1, ay1, bx1, by1,
                                                         ax2, ay2, bx2, by2)
                    is_valid = d2 < δ2
                end
            end
        end
        valid[pair_idx] = is_valid ? UInt8(1) : UInt8(0)
    end
end

@kernel function _compact_close_pair_candidates_kernel!(pair_ci, pair_i,
                                                        pair_cj, pair_j,
                                                        slots, valid,
                                                        eligible,
                                                        contour_of_node,
                                                        local_index,
                                                        neligible, npairs)
    pair_idx = @index(Global)
    if pair_idx <= npairs && !iszero(valid[pair_idx])
        slot = slots[pair_idx]
        g1 = eligible[((pair_idx - 1) % neligible) + 1]
        g2 = eligible[((pair_idx - 1) ÷ neligible) + 1]
        pair_ci[slot] = contour_of_node[g1]
        pair_i[slot] = local_index[g1]
        pair_cj[slot] = contour_of_node[g2]
        pair_j[slot] = local_index[g2]
    end
end

function _device_close_pair_candidate_buffer(flat::FlatContourTopology{T}, δ,
                                             domain::AbstractDomain,
                                             dev::AbstractDevice=CPU()) where {T}
    total_nodes = _flat_nnodes(flat)
    if total_nodes == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    eligible = _device_eligible_surgery_segment_indices(flat, dev)
    neligible = length(eligible)
    if neligible == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    npairs = neligible * neligible
    valid = device_zeros(dev, UInt8, npairs)
    periodic, Lx, Ly = _flat_surgery_domain(domain, T)
    @_ka_launch dev npairs _close_pair_candidate_kernel!(
        valid, eligible, flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy, flat.offsets,
        flat.lengths, flat.contour_of_node, flat.local_index, flat.corners,
        periodic, Lx, Ly, T(δ)^2, neligible)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    _device_compact_scan!(slots, count_store, valid, npairs, dev)
    ncandidates = to_cpu(count_store)[1]

    pair_ci = device_zeros(dev, Int, ncandidates)
    pair_i = device_zeros(dev, Int, ncandidates)
    pair_cj = device_zeros(dev, Int, ncandidates)
    pair_j = device_zeros(dev, Int, ncandidates)
    if ncandidates > 0
        @_ka_launch dev npairs _compact_close_pair_candidates_kernel!(
            pair_ci, pair_i, pair_cj, pair_j, slots, valid,
            eligible, flat.contour_of_node, flat.local_index, neligible, npairs)
    end

    return DeviceClosePairCandidates(pair_ci, pair_i, pair_cj, pair_j)
end

function _device_close_pair_candidate_buffer(flat::FlatContourTopology{T}, δ,
                                             dev::AbstractDevice=CPU()) where {T}
    return _device_close_pair_candidate_buffer(flat, δ, UnboundedDomain(), dev)
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

function _device_close_pair_candidate_buffer(state::DeviceContourState{T}, δ,
                                             domain::AbstractDomain,
                                             dev::AbstractDevice=CPU()) where {T}
    return _device_close_pair_candidate_buffer(_flat_topology(state, dev),
                                               δ, domain, dev)
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
                                              lengths, ci, i, δ,
                                              periodic, Lx, Ly)
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
        px = (ax + bx) / 2
        py = (ay + by) / 2
        return periodic ? (_flat_wrap_coord(px, Lx), _flat_wrap_coord(py, Ly)) : (px, py)
    end

    area2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
    left_x = -sy / seg_len
    left_y = sx / seg_len
    inward_x = area2 >= zero(area2) ? left_x : -left_x
    inward_y = area2 >= zero(area2) ? left_y : -left_y
    probe_distance = max(δ / 10,
                         eps(δ) * (one(δ) + abs(ax) + abs(ay) + seg_len))
    px = (ax + bx) / 2 + probe_distance * inward_x
    py = (ay + by) / 2 + probe_distance * inward_y
    return periodic ? (_flat_wrap_coord(px, Lx), _flat_wrap_coord(py, Ly)) : (px, py)
end

@inline function _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                               offsets, lengths, ci,
                                               periodic, Lx, Ly)
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
        ax, ay, bx, by = _flat_shift_segment_to_image(
            ax, ay, bx, by, px, py, periodic, Lx, Ly)
        inside = inside != _flat_ray_crosses_segment(px, py, ax, ay, bx, by)
    end
    return inside
end

@inline function _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, ci, i, δ,
                                                ncontours, periodic, Lx, Ly)
    px, py = _flat_segment_interior_probe(x, y, wrapx, wrapy, offsets,
                                          lengths, ci, i, δ, periodic, Lx, Ly)
    q = zero(δ)
    @inbounds for ck in 1:ncontours
        if _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                         offsets, lengths, ck,
                                         periodic, Lx, Ly)
            q += pv[ck]
        end
    end
    return q
end

@kernel function _admissible_close_pair_kernel!(valid, pair_ci, pair_i,
                                                pair_cj, pair_j, x, y, pv,
                                                wrapx, wrapy, offsets, lengths,
                                                δ, ncontours, periodic, Lx, Ly,
                                                npairs)
    k = @index(Global)
    if k <= npairs
        ci = pair_ci[k]
        cj = pair_cj[k]
        ok = ci == cj
        if !ok
            qi = _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, ci,
                                                pair_i[k], δ, ncontours,
                                                periodic, Lx, Ly)
            qj = _flat_local_interior_vorticity(x, y, pv, wrapx, wrapy,
                                                offsets, lengths, cj,
                                                pair_j[k], δ, ncontours,
                                                periodic, Lx, Ly)
            tol = sqrt(eps(one(δ))) * max(one(δ), abs(qi), abs(qj))
            ok = abs(qi - qj) <= tol
        end
        valid[k] = ok ? UInt8(1) : UInt8(0)
    end
end

function _device_admissible_close_segment_buffer(flat::FlatContourTopology{T}, δ,
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    candidates = _device_close_pair_candidate_buffer(flat, δ, domain, dev)
    npairs = length(candidates.ci)
    npairs == 0 && return candidates

    ncontours = _flat_ncontours(flat)
    valid = device_zeros(dev, UInt8, npairs)
    periodic, Lx, Ly = _flat_surgery_domain(domain, T)
    @_ka_launch dev npairs _admissible_close_pair_kernel!(
        valid, candidates.ci, candidates.i, candidates.cj, candidates.j,
        flat.x, flat.y, flat.pv, flat.wrapx, flat.wrapy, flat.offsets,
        flat.lengths, T(δ), ncontours, periodic, Lx, Ly, npairs)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    _device_compact_scan!(slots, count_store, valid, npairs, dev)
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
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_admissible_close_segment_buffer(
        _pack_flat_topology(contours, dev), δ, domain, dev)
end

function _device_admissible_close_segment_buffer(state::DeviceContourState{T}, δ,
                                                 domain::AbstractDomain,
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
                                             wrapy, offsets, lengths,
                                             periodic, Lx, Ly, npairs)
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

        if periodic
            refx = _flat_wrap_coord((ax1 + bx1) / 2, Lx)
            refy = _flat_wrap_coord((ay1 + by1) / 2, Ly)
            ax1, ay1, bx1, by1 = _flat_shift_segment_to_image(
                ax1, ay1, bx1, by1, refx, refy, periodic, Lx, Ly)
            ax2, ay2, bx2, by2 = _flat_shift_segment_to_image(
                ax2, ay2, bx2, by2, refx, refy, periodic, Lx, Ly)
        end

        distance2[k] = _flat_surgery_contact_distance2(ax1, ay1, bx1, by1,
                                                       ax2, ay2, bx2, by2)
        op[k] = ci == cj ? UInt8(1) : UInt8(2)
    end
end

function _device_reconnection_plan_from_vectors(flat::FlatContourTopology{T},
                                                pair_ci, pair_i, pair_cj, pair_j,
                                                domain::AbstractDomain,
                                                dev::AbstractDevice=CPU()) where {T}
    npairs = length(pair_ci)
    distance2 = device_zeros(dev, T, npairs)
    op = device_zeros(dev, UInt8, npairs)
    selected = device_zeros(dev, UInt8, npairs)
    if npairs > 0
        periodic, Lx, Ly = _flat_surgery_domain(domain, T)
        @_ka_launch dev npairs _pair_distance_plan_kernel!(
            distance2, op, pair_ci, pair_i, pair_cj, pair_j,
            flat.x, flat.y, flat.wrapx, flat.wrapy, flat.offsets,
            flat.lengths, periodic, Lx, Ly, npairs)
    end
    return DeviceReconnectionPlan(pair_ci, pair_i, pair_cj, pair_j,
                                  distance2, op, selected)
end

function _device_reconnection_plan_from_vectors(flat::FlatContourTopology{T},
                                                pair_ci, pair_i, pair_cj, pair_j,
                                                dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(
        flat, pair_ci, pair_i, pair_cj, pair_j, UnboundedDomain(), dev)
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

function _device_reconnection_plan(state::DeviceContourState{T},
                                   candidates::DeviceClosePairCandidates,
                                   domain::AbstractDomain,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(
        _flat_topology(state, dev), candidates.ci, candidates.i,
        candidates.cj, candidates.j, domain, dev)
end

function _device_reconnection_plan(flat::FlatContourTopology{T},
                                   candidates::DeviceClosePairCandidates,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(flat, candidates.ci,
                                                  candidates.i, candidates.cj,
                                                  candidates.j, dev)
end

function _device_reconnection_plan(flat::FlatContourTopology{T},
                                   candidates::DeviceClosePairCandidates,
                                   domain::AbstractDomain,
                                   dev::AbstractDevice=CPU()) where {T}
    return _device_reconnection_plan_from_vectors(
        flat, candidates.ci, candidates.i, candidates.cj, candidates.j,
        domain, dev)
end

function _device_select_reconnection_pairs_from_plan(contours::Vector{PVContour{T}},
                                                     close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                     plan::DeviceReconnectionPlan{T}) where {T}
    distance2 = to_cpu(plan.distance2)

    ranked = Vector{Tuple{T,Tuple{Int,Int,Int,Int}}}(undef, length(close_pairs))
    @inbounds for k in eachindex(close_pairs)
        ranked[k] = (distance2[k], close_pairs[k])
    end
    sort!(ranked)

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
                                                   distance2, pair_ci, pair_i,
                                                   pair_cj, pair_j, npairs)
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
                tied_before = false
                if best != 0 && d2 == best_d2
                    best_ci = pair_ci[best]
                    best_i = pair_i[best]
                    best_cj = pair_cj[best]
                    best_j = pair_j[best]
                    tied_before = ci < best_ci ||
                        (ci == best_ci && pair_i[k] < best_i) ||
                        (ci == best_ci && pair_i[k] == best_i && cj < best_cj) ||
                        (ci == best_ci && pair_i[k] == best_i && cj == best_cj &&
                         pair_j[k] < best_j)
                end
                if best == 0 || d2 < best_d2 || tied_before
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
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    npairs = length(candidates.ci)
    if npairs == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    plan = _device_reconnection_plan(flat, candidates, domain, dev)
    used_contours = device_zeros(dev, UInt8, _flat_ncontours(flat))
    @_ka_launch dev 1 _select_independent_pairs_kernel!(
        plan.selected, used_contours, plan.distance2, plan.ci, plan.i,
        plan.cj, plan.j, npairs)

    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    _device_compact_scan!(slots, count_store, plan.selected, npairs, dev)
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

function _device_select_reconnection_pair_buffer(flat::FlatContourTopology{T},
                                                 candidates::DeviceClosePairCandidates,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_select_reconnection_pair_buffer(
        flat, candidates, UnboundedDomain(), dev)
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

function _device_select_reconnection_pair_buffer(state::DeviceContourState{T},
                                                 candidates::DeviceClosePairCandidates,
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice=CPU()) where {T}
    return _device_select_reconnection_pair_buffer(
        _flat_topology(state, dev), candidates, domain, dev)
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
