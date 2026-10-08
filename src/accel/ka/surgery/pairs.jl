# Close-pair detection and reconnection planning: segment-distance predicates,
# candidate generation, the interior-vorticity admissibility test, and
# independent split/merge pair selection.


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

# Whether `_flat_shift_segment_to_image` moves both segments toward
# (refx, refy) by the same whole number of periods.
@inline function _flat_same_image(ax1, ay1, bx1, by1, ax2, ay2, bx2, by2,
                                  refx, refy, Lx, Ly)
    return round((refx - (ax1 + bx1) / 2) / (2 * Lx)) ==
           round((refx - (ax2 + bx2) / 2) / (2 * Lx)) &&
           round((refy - (ay1 + by1) / 2) / (2 * Ly)) ==
           round((refy - (ay2 + by2) / 2) / (2 * Ly))
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

# Evaluates pair indices `offset+1 : offset+nlocal` of the flattened
# eligible×eligible pair space, writing chunk-local validity flags, so the
# caller can sweep an O(neligible²) pair space with O(chunk) scratch.
@kernel function _close_pair_candidate_kernel!(valid, offset, nlocal, eligible,
                                               x, y, pv, wrapx, wrapy, offsets,
                                               lengths, contour_of_node, local_index,
                                               corners, periodic, Lx, Ly, δ2,
                                               neligible)
    local_idx = @index(Global)
    if local_idx <= nlocal
        pair_idx = local_idx + offset
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
                    admissible = _same_surgery_pv(pv[ci], pv[cj])
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

                    same_image = true
                    if periodic
                        refx = _flat_wrap_coord((ax1 + bx1) / 2, Lx)
                        refy = _flat_wrap_coord((ay1 + by1) / 2, Ly)
                        # A closed contour touching its own periodic image
                        # would reconnect into spanning contours, which are
                        # exempt from surgery (CPU `find_close_segments`).
                        same_image = ci != cj ||
                            _flat_same_image(ax1, ay1, bx1, by1,
                                             ax2, ay2, bx2, by2,
                                             refx, refy, Lx, Ly)
                        ax1, ay1, bx1, by1 = _flat_shift_segment_to_image(
                            ax1, ay1, bx1, by1, refx, refy, periodic, Lx, Ly)
                        ax2, ay2, bx2, by2 = _flat_shift_segment_to_image(
                            ax2, ay2, bx2, by2, refx, refy, periodic, Lx, Ly)
                    end

                    d2 = _flat_surgery_contact_distance2(ax1, ay1, bx1, by1,
                                                         ax2, ay2, bx2, by2)
                    is_valid = same_image && d2 < δ2
                end
            end
        end
        valid[local_idx] = is_valid ? UInt8(1) : UInt8(0)
    end
end

@kernel function _compact_close_pair_candidates_kernel!(pair_ci, pair_i,
                                                        pair_cj, pair_j,
                                                        slots, valid, offset,
                                                        eligible,
                                                        contour_of_node,
                                                        local_index,
                                                        neligible, nlocal)
    local_idx = @index(Global)
    if local_idx <= nlocal && !iszero(valid[local_idx])
        slot = slots[local_idx]
        pair_idx = local_idx + offset
        g1 = eligible[((pair_idx - 1) % neligible) + 1]
        g2 = eligible[((pair_idx - 1) ÷ neligible) + 1]
        pair_ci[slot] = contour_of_node[g1]
        pair_i[slot] = local_index[g1]
        pair_cj[slot] = contour_of_node[g2]
        pair_j[slot] = local_index[g2]
    end
end

# Pair-index block size per launch when sweeping the eligible×eligible pair
# space. Bounds the validity/scan scratch to ~33 MB regardless of node count.
const _PAIR_SCAN_CHUNK = 1 << 20

function _device_close_pair_candidate_buffer(flat::FlatContourTopology{T}, δ,
                                             domain::AbstractDomain,
                                             dev::AbstractDevice) where {T}
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

    # The pair space is neligible², so materializing per-pair scratch for all
    # of it at once would need ~24 bytes per pair (~10 GB for 20k eligible
    # segments). Sweep it in fixed-size chunks instead: scratch stays
    # O(_PAIR_SCAN_CHUNK) while the compacted candidate output — which is
    # small in practice — is concatenated across chunks in pair-index order,
    # preserving the ordering of the previous all-at-once implementation.
    npairs = neligible * neligible
    chunk = min(npairs, _PAIR_SCAN_CHUNK)
    valid = device_zeros(dev, UInt8, chunk)
    slots = device_zeros(dev, Int, chunk)
    scan_a = device_zeros(dev, Int, chunk)
    scan_b = device_zeros(dev, Int, chunk)
    count_store = device_zeros(dev, Int, 1)
    periodic, Lx, Ly = _flat_surgery_domain(domain, T)
    δ2 = T(δ)^2

    V = typeof(slots)
    parts = Tuple{V,V,V,V}[]
    total = 0
    lo = 0
    while lo < npairs
        len = min(chunk, npairs - lo)
        @_ka_launch dev len _close_pair_candidate_kernel!(
            valid, lo, len, eligible, flat.x, flat.y, flat.pv, flat.wrapx,
            flat.wrapy, flat.offsets, flat.lengths, flat.contour_of_node,
            flat.local_index, flat.corners, periodic, Lx, Ly, δ2, neligible)
        _device_compact_scan!(slots, count_store, valid, len, dev, scan_a, scan_b)
        c = to_cpu(count_store)[1]
        if c > 0
            p_ci = device_zeros(dev, Int, c)
            p_i = device_zeros(dev, Int, c)
            p_cj = device_zeros(dev, Int, c)
            p_j = device_zeros(dev, Int, c)
            @_ka_launch dev len _compact_close_pair_candidates_kernel!(
                p_ci, p_i, p_cj, p_j, slots, valid, lo, eligible,
                flat.contour_of_node, flat.local_index, neligible, len)
            push!(parts, (p_ci, p_i, p_cj, p_j))
            total += c
        end
        lo += len
    end

    if total == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end
    length(parts) == 1 &&
        return DeviceClosePairCandidates(parts[1][1], parts[1][2], parts[1][3], parts[1][4])

    pair_ci = device_zeros(dev, Int, total)
    pair_i = device_zeros(dev, Int, total)
    pair_cj = device_zeros(dev, Int, total)
    pair_j = device_zeros(dev, Int, total)
    off = 1
    for (p_ci, p_i, p_cj, p_j) in parts
        n = length(p_ci)
        copyto!(pair_ci, off, p_ci, 1, n)
        copyto!(pair_i, off, p_i, 1, n)
        copyto!(pair_cj, off, p_cj, 1, n)
        copyto!(pair_j, off, p_j, 1, n)
        off += n
    end
    return DeviceClosePairCandidates(pair_ci, pair_i, pair_cj, pair_j)
end

# Adapters: any contour container, domain defaults to unbounded, dev to CPU.
function _device_close_pair_candidate_buffer(input::_UnflatContourInput, δ,
                                             domain::AbstractDomain=UnboundedDomain(),
                                             dev::AbstractDevice=CPU())
    return _device_close_pair_candidate_buffer(_as_flat(input, dev), δ, domain, dev)
end
_device_close_pair_candidate_buffer(input::_UnflatContourInput, δ, dev::AbstractDevice) =
    _device_close_pair_candidate_buffer(input, δ, UnboundedDomain(), dev)

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


@inline function _flat_segment_endpoints(x, y, wrapx, wrapy, offsets, lengths, ci, i)
    off = offsets[ci]
    n = lengths[ci]
    g = off + i - 1
    ax = x[g]
    ay = y[g]
    bx = i < n ? x[g + 1] : x[off] + wrapx[ci]
    by = i < n ? y[g + 1] : y[off] + wrapy[ci]
    return ax, ay, bx, by
end

# Device twin of `_segment_side_probe`: a point just off segment i on its left
# (`side = +1`) or right (`side = -1`).
@inline function _flat_segment_side_probe(x, y, wrapx, wrapy, offsets,
                                          lengths, ci, i, side, δ,
                                          periodic, Lx, Ly)
    ax, ay, bx, by = _flat_segment_endpoints(x, y, wrapx, wrapy, offsets,
                                             lengths, ci, i)
    sx = bx - ax
    sy = by - ay
    seg_len = sqrt(sx * sx + sy * sy)
    if iszero(seg_len)
        px = (ax + bx) / 2
        py = (ay + by) / 2
        return periodic ? (_flat_wrap_coord(px, Lx), _flat_wrap_coord(py, Ly)) : (px, py)
    end

    left_x = -sy / seg_len
    left_y = sx / seg_len
    probe_distance = max(δ / 10,
                         eps(typeof(δ)) * (one(δ) + abs(ax) + abs(ay) + seg_len))
    px = (ax + bx) / 2 + probe_distance * side * left_x
    py = (ay + by) / 2 + probe_distance * side * left_y
    return periodic ? (_flat_wrap_coord(px, Lx), _flat_wrap_coord(py, Ly)) : (px, py)
end

@inline function _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                               offsets, lengths, ci,
                                               periodic, Lx, Ly)
    (!iszero(wrapx[ci]) || !iszero(wrapy[ci])) && return false
    off = offsets[ci]
    n = lengths[ci]
    getnode = i -> (x[off + i - 1], y[off + i - 1])
    if periodic
        return _periodic_point_in_polygon(px, py, n, getnode, Lx, Ly)
    end
    return _point_in_polygon(px, py, n, getnode)
end

# Device twin of `_local_side_vorticity`: the physical PV level just off one
# side of segment i (clockwise contours bound -pv regions) and the magnitude
# of the summed jumps.
@inline function _flat_local_side_vorticity(x, y, pv, wrapx, wrapy,
                                            offsets, lengths, ci, i, side, δ,
                                            ncontours, periodic, Lx, Ly)
    px, py = _flat_segment_side_probe(x, y, wrapx, wrapy, offsets,
                                      lengths, ci, i, side, δ, periodic, Lx, Ly)
    q = zero(δ)
    scale = zero(δ)
    @inbounds for ck in 1:ncontours
        if _flat_point_in_closed_contour(px, py, x, y, wrapx, wrapy,
                                         offsets, lengths, ck,
                                         periodic, Lx, Ly)
            area2 = _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ck)
            q += sign(area2) * pv[ck]
            scale += abs(pv[ck])
        end
    end
    return q, scale
end

# Device twin of the far-side admissibility test in `find_close_segments`.
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
            i = pair_i[k]
            j = pair_j[k]
            ax1, ay1, bx1, by1 = _flat_segment_endpoints(x, y, wrapx, wrapy,
                                                         offsets, lengths, ci, i)
            ax2, ay2, bx2, by2 = _flat_segment_endpoints(x, y, wrapx, wrapy,
                                                         offsets, lengths, cj, j)
            if periodic
                refx = _flat_wrap_coord((ax1 + bx1) / 2, Lx)
                refy = _flat_wrap_coord((ay1 + by1) / 2, Ly)
                ax1, ay1, bx1, by1 = _flat_shift_segment_to_image(
                    ax1, ay1, bx1, by1, refx, refy, periodic, Lx, Ly)
                ax2, ay2, bx2, by2 = _flat_shift_segment_to_image(
                    ax2, ay2, bx2, by2, refx, refy, periodic, Lx, Ly)
            end
            side_i = _flat_far_side(ax1, ay1, bx1, by1,
                                    (ax2 + bx2) / 2, (ay2 + by2) / 2)
            side_j = _flat_far_side(ax2, ay2, bx2, by2,
                                    (ax1 + bx1) / 2, (ay1 + by1) / 2)
            qi, scale_i = _flat_local_side_vorticity(x, y, pv, wrapx, wrapy,
                                                     offsets, lengths, ci, i,
                                                     side_i, δ, ncontours,
                                                     periodic, Lx, Ly)
            qj, scale_j = _flat_local_side_vorticity(x, y, pv, wrapx, wrapy,
                                                     offsets, lengths, cj, j,
                                                     side_j, δ, ncontours,
                                                     periodic, Lx, Ly)
            ok = _same_surgery_pv(qi, qj, max(scale_i, scale_j))
        end
        valid[k] = ok ? UInt8(1) : UInt8(0)
    end
end

function _device_admissible_close_segment_buffer(flat::FlatContourTopology{T}, δ,
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice) where {T}
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

# Adapter: any contour container; the admissibility test always takes an
# explicit domain.
function _device_admissible_close_segment_buffer(input::_UnflatContourInput, δ,
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice=CPU())
    return _device_admissible_close_segment_buffer(_as_flat(input, dev), δ, domain, dev)
end

# Host-side test seam: unpacked candidate tuples.
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

# Normalize a `_DevicePairList` to packed device candidates.
_as_candidates(candidates::DeviceClosePairCandidates, ::AbstractDevice) = candidates
_as_candidates(pairs::Vector{Tuple{Int,Int,Int,Int}}, dev::AbstractDevice) =
    _pack_close_pair_candidates(pairs, dev)

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
                                                dev::AbstractDevice) where {T}
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

# Adapters: any contour container and pair list, domain defaults to unbounded.
function _device_reconnection_plan(input::_DeviceContourInput, pairs::_DevicePairList,
                                   domain::AbstractDomain=UnboundedDomain(),
                                   dev::AbstractDevice=CPU())
    c = _as_candidates(pairs, dev)
    return _device_reconnection_plan_from_vectors(
        _as_flat(input, dev), c.ci, c.i, c.cj, c.j, domain, dev)
end
_device_reconnection_plan(input::_DeviceContourInput, pairs::_DevicePairList,
                          dev::AbstractDevice) =
    _device_reconnection_plan(input, pairs, UnboundedDomain(), dev)

# Serial greedy planner: repeatedly pick the closest still-admissible pair whose
# contours are unused. Each pick is recorded both as a flag (`selected`) and as
# its 1-based pick order (`order`), so the compacted pair buffer can be laid
# out in `(distance2, (ci, i, cj, j))` order exactly like the CPU
# `_select_reconnection_pairs`. Split daughters are appended in that order on
# both backends, so the resulting contour vectors match element for element.
@kernel function _select_independent_pairs_kernel!(selected, order, count_store,
                                                   used_contours,
                                                   distance2, pair_ci, pair_i,
                                                   pair_cj, pair_j, npairs)
    worker = @index(Global)
    if worker == 1
        nselected = 0
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
            nselected += 1
            selected[best] = UInt8(1)
            order[best] = nselected
            used_contours[ci] = UInt8(1)
            used_contours[cj] = UInt8(1)
        end
        count_store[1] = nselected
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

function _device_select_reconnection_pair_buffer(flat::FlatContourTopology,
                                                 candidates::DeviceClosePairCandidates,
                                                 domain::AbstractDomain,
                                                 dev::AbstractDevice)
    npairs = length(candidates.ci)
    if npairs == 0
        empty_ints = device_zeros(dev, Int, 0)
        return DeviceClosePairCandidates(empty_ints, empty_ints, empty_ints, empty_ints)
    end

    plan = _device_reconnection_plan(flat, candidates, domain, dev)
    used_contours = device_zeros(dev, UInt8, _flat_ncontours(flat))
    # `slots` holds each selected pair's pick order, which doubles as its
    # compaction slot: the output buffer is sorted by proximity, not by
    # candidate-buffer position.
    slots = device_zeros(dev, Int, npairs)
    count_store = device_zeros(dev, Int, 1)
    @_ka_launch dev 1 _select_independent_pairs_kernel!(
        plan.selected, slots, count_store, used_contours, plan.distance2,
        plan.ci, plan.i, plan.cj, plan.j, npairs)
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

# Adapters: any contour container and pair list, domain defaults to unbounded.
function _device_select_reconnection_pair_buffer(input::_UnflatContourInput,
                                                 pairs::_DevicePairList,
                                                 domain::AbstractDomain=UnboundedDomain(),
                                                 dev::AbstractDevice=CPU())
    return _device_select_reconnection_pair_buffer(
        _as_flat(input, dev), _as_candidates(pairs, dev), domain, dev)
end
_device_select_reconnection_pair_buffer(input::_UnflatContourInput, pairs::_DevicePairList,
                                        dev::AbstractDevice) =
    _device_select_reconnection_pair_buffer(input, pairs, UnboundedDomain(), dev)
