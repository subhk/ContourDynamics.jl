# Topology rewrite: signed-area helpers, the split/merge size plan, output
# layout (serial reference and parallel scan), node materialization, and
# rebuilding a device state from the rewritten outputs.

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

# Stream-compaction prefix sum over a 0/1 `flags` array, replacing an earlier
# kernel where every workitem summed flags[1..idx-1] (O(n) per item → O(n²) total,
# and O(N⁴) when launched over the N² close-pair candidate buffer). This uses a
# ping-pong Hillis–Steele inclusive scan: O(n log n) work, O(log n) launches, and
# — because each pass reads `in` and writes a separate `out` — there is no
# intra-pass aliasing, so the result is independent of workitem execution order.
#
# Output matches the previous kernel exactly: for a kept item (flag ≠ 0),
# `slots[i]` is its 1-based compacted position (= number of kept items in 1..i);
# dropped items get 0; `total[1]` is the total kept count.
@kernel function _scan_init_kernel!(out, flags, n)
    i = @index(Global)
    if i <= n
        @inbounds out[i] = iszero(flags[i]) ? 0 : 1
    end
end

@kernel function _scan_step_kernel!(out, in, offset, n)
    i = @index(Global)
    if i <= n
        @inbounds out[i] = i > offset ? in[i] + in[i - offset] : in[i]
    end
end

@kernel function _scan_compact_finalize_kernel!(slots, total, scan, flags, n)
    i = @index(Global)
    if i <= n
        @inbounds begin
            incl = scan[i]                     # inclusive prefix count at i
            slots[i] = iszero(flags[i]) ? 0 : incl
            if i == n
                total[1] = incl
            end
        end
    end
end

# Host driver for the compaction scan. `slots` and `total` are caller-allocated;
# `total` must be zero-initialized so the n == 0 case leaves a 0 count.
function _device_compact_scan!(slots, total, flags, n::Int, dev::AbstractDevice)
    n == 0 && return slots
    a = device_zeros(dev, Int, n)
    b = device_zeros(dev, Int, n)
    @_ka_launch dev n _scan_init_kernel!(a, flags, n)
    cur, other = a, b
    offset = 1
    while offset < n
        @_ka_launch dev n _scan_step_kernel!(other, cur, offset, n)
        cur, other = other, cur
        offset *= 2
    end
    @_ka_launch dev n _scan_compact_finalize_kernel!(slots, total, cur, flags, n)
    return slots
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
                              outputs.wrapy,
                              device_zeros(dev, T, ncontours),
                              device_zeros(dev, T, ncontours),
                              outputs.offsets, outputs.lengths,
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
    state.shiftx = replacement.shiftx
    state.shifty = replacement.shifty
    state.offsets = replacement.offsets
    state.lengths = replacement.lengths
    state.corners = replacement.corners
    state.contour_of_node = replacement.contour_of_node
    state.local_index = replacement.local_index
    return state
end

