# Topology rewrite: signed-area helpers, the split/merge size plan, output
# layout (serial reference and parallel scan), node materialization, and
# rebuilding a device state from the rewritten outputs.

@inline function _flat_closed_area2(x, y, wrapx, wrapy, offsets, lengths, ci)
    n = lengths[ci]
    n < 3 && return zero(eltype(x))
    off = offsets[ci]
    ox = x[off]
    oy = y[off]
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
        px = x[g] - ox
        py = y[g] - oy
        next_x = nx - ox
        next_y = ny - oy
        area2 += px * next_y - next_x * py
    end
    return area2
end

@inline function _flat_point_segment_dist2_in_domain(px, py, ax, ay, bx, by,
                                                     periodic, Lx, Ly)
    if periodic
        px = _flat_wrap_coord(px, Lx)
        py = _flat_wrap_coord(py, Ly)
        ax, ay, bx, by = _flat_shift_segment_to_image(
            ax, ay, bx, by, px, py, periodic, Lx, Ly)
    end
    d2, _, _ = _flat_point_segment_closest(px, py, ax, ay, bx, by)
    return d2
end

@kernel function _topology_rewrite_size_kernel!(op, valid, node_from_first,
                                                node_idx, seg_idx, inserted_idx,
                                                stitch_x, stitch_y,
                                                merge_shift_x, merge_shift_y,
                                                out_count,
                                                out_len1, out_len2, pair_ci,
                                                pair_i, pair_cj, pair_j, x, y,
                                                wrapx, wrapy, offsets, lengths,
                                                periodic, Lx, Ly, npairs)
    k = @index(Global)
    if k <= npairs
        ci = pair_ci[k]
        i = pair_i[k]
        cj = pair_cj[k]
        j = pair_j[k]
        n1 = lengths[ci]
        n2 = lengths[cj]
        op_k = ci == cj ? UInt8(1) : UInt8(2)

        # Neither splits nor merges reverse a contour: see the CPU
        # `_reconnect_split!` and `_reconnect_merge!`.
        g1 = offsets[ci] + i - 1
        ax1 = x[g1]
        ay1 = y[g1]
        i_end = i < n1 ? i + 1 : 1
        g1_end = offsets[ci] + i_end - 1
        bx1 = i < n1 ? x[g1 + 1] : x[offsets[ci]] + wrapx[ci]
        by1 = i < n1 ? y[g1 + 1] : y[offsets[ci]] + wrapy[ci]

        j_eff = j
        j_end = j_eff < n2 ? j_eff + 1 : 1
        ax2 = x[offsets[cj] + j_eff - 1]
        ay2 = y[offsets[cj] + j_eff - 1]
        bx2 = j_eff < n2 ? x[offsets[cj] + j_eff] : x[offsets[cj]] + wrapx[cj]
        by2 = j_eff < n2 ? y[offsets[cj] + j_eff] : y[offsets[cj]] + wrapy[cj]

        shift_x = zero(eltype(x))
        shift_y = zero(eltype(y))
        if op_k == UInt8(2) && periodic
            raw_x = ax1 - ax2
            raw_y = ay1 - ay2
            shift_x = round(raw_x / (2 * Lx)) * (2 * Lx)
            shift_y = round(raw_y / (2 * Ly)) * (2 * Ly)
            ax2 += shift_x
            ay2 += shift_y
            bx2 += shift_x
            by2 += shift_y
        end

        best_d2 = _flat_point_segment_dist2_in_domain(
            ax1, ay1, ax2, ay2, bx2, by2, periodic, Lx, Ly)
        best_x = ax1
        best_y = ay1
        best_node_from_first = UInt8(1)
        best_node_idx = i
        best_seg_idx = j_eff

        x1_end = x[g1_end]
        y1_end = y[g1_end]
        d2 = _flat_point_segment_dist2_in_domain(
            x1_end, y1_end, ax2, ay2, bx2, by2, periodic, Lx, Ly)
        if d2 < best_d2
            best_d2 = d2
            best_x = x1_end
            best_y = y1_end
            best_node_from_first = UInt8(1)
            best_node_idx = i_end
            best_seg_idx = j_eff
        end

        d2 = _flat_point_segment_dist2_in_domain(
            ax2, ay2, ax1, ay1, bx1, by1, periodic, Lx, Ly)
        if d2 < best_d2
            best_d2 = d2
            best_x = ax2
            best_y = ay2
            best_node_from_first = UInt8(0)
            best_node_idx = j_eff
            best_seg_idx = i
        end

        d2 = _flat_point_segment_dist2_in_domain(
            bx2, by2, ax1, ay1, bx1, by1, periodic, Lx, Ly)
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
        if op_k == UInt8(1)
            adjusted_node = best_seg_idx < best_node_idx ? best_node_idx + 1 : best_node_idx
            lo = min(adjusted_node, inserted)
            hi = max(adjusted_node, inserted)
            len1 = hi - lo
            len2 = n1 + 1 - len1
            if len1 >= 3 && len2 >= 3
                count_k = 2
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
        stitch_x[k] = best_x
        stitch_y[k] = best_y
        merge_shift_x[k] = shift_x
        merge_shift_y[k] = shift_y
        out_count[k] = count_k
        out_len1[k] = len1
        out_len2[k] = len2
    end
end

function _device_topology_rewrite_plan_from_vectors(flat::FlatContourTopology{T},
                                                    pair_ci, pair_i, pair_cj, pair_j,
                                                    domain::AbstractDomain,
                                                    dev::AbstractDevice) where {T}
    npairs = length(pair_ci)
    op = device_zeros(dev, UInt8, npairs)
    valid = device_zeros(dev, UInt8, npairs)
    node_from_first = device_zeros(dev, UInt8, npairs)
    node_idx = device_zeros(dev, Int, npairs)
    seg_idx = device_zeros(dev, Int, npairs)
    inserted_idx = device_zeros(dev, Int, npairs)
    stitch_x = device_zeros(dev, T, npairs)
    stitch_y = device_zeros(dev, T, npairs)
    merge_shift_x = device_zeros(dev, T, npairs)
    merge_shift_y = device_zeros(dev, T, npairs)
    out_count = device_zeros(dev, Int, npairs)
    out_len1 = device_zeros(dev, Int, npairs)
    out_len2 = device_zeros(dev, Int, npairs)

    if npairs > 0
        periodic, Lx, Ly = _flat_surgery_domain(domain, T)
        @_ka_launch dev npairs _topology_rewrite_size_kernel!(
            op, valid, node_from_first, node_idx, seg_idx, inserted_idx,
            stitch_x, stitch_y, merge_shift_x, merge_shift_y,
            out_count, out_len1, out_len2, pair_ci, pair_i,
            pair_cj, pair_j, flat.x, flat.y, flat.wrapx, flat.wrapy,
            flat.offsets, flat.lengths, periodic, Lx, Ly, npairs)
    end

    return DeviceTopologyRewritePlan(pair_ci, pair_i, pair_cj, pair_j, op,
                                     valid, node_from_first, node_idx, seg_idx,
                                     inserted_idx, stitch_x, stitch_y,
                                     merge_shift_x, merge_shift_y,
                                     out_count, out_len1, out_len2)
end

# Adapters: any contour container and pair list, domain defaults to unbounded.
function _device_topology_rewrite_plan(input::_DeviceContourInput,
                                       selected_pairs::_DevicePairList,
                                       domain::AbstractDomain=UnboundedDomain(),
                                       dev::AbstractDevice=CPU())
    c = _as_candidates(selected_pairs, dev)
    return _device_topology_rewrite_plan_from_vectors(
        _as_flat(input, dev), c.ci, c.i, c.cj, c.j, domain, dev)
end
_device_topology_rewrite_plan(input::_DeviceContourInput, selected_pairs::_DevicePairList,
                              dev::AbstractDevice) =
    _device_topology_rewrite_plan(input, selected_pairs, UnboundedDomain(), dev)

@inline function _inserted_contour_node(x, y, corners, offsets, ci, inserted_idx,
                                        stitch_x, stitch_y, local_idx)
    if inserted_idx > 0 && local_idx == inserted_idx
        return stitch_x, stitch_y, UInt8(0)
    end
    original_idx = inserted_idx > 0 && local_idx > inserted_idx ? local_idx - 1 : local_idx
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
                                                       stitch_x, stitch_y,
                                                       merge_shift_x, merge_shift_y,
                                                       in_x, in_y,
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
                source_local = 1
                if part == 1
                    source_local = lo + out_local - 1
                else
                    first_span = nc - hi + 1
                    source_local = out_local <= first_span ? hi + out_local - 1 :
                                                         out_local - first_span
                end
                ox, oy, corner = _inserted_contour_node(in_x, in_y, in_corners,
                                                        in_offsets, ci, inserted,
                                                        stitch_x[op_idx],
                                                        stitch_y[op_idx],
                                                        source_local)
                out_local == 1 && (corner = UInt8(1))
            else
                n1 = in_lengths[pair_ci[op_idx]]
                n2 = in_lengths[pair_cj[op_idx]]
                from_first = !iszero(node_from_first[op_idx])
                c1_inserted = from_first ? 0 : inserted_idx[op_idx]
                c2_inserted = from_first ? inserted_idx[op_idx] : 0
                c1_len = n1 + (from_first ? 0 : 1)
                c2_len = n2 + (from_first ? 1 : 0)
                c1_start = from_first ? node_idx[op_idx] : inserted_idx[op_idx]
                c2_start = from_first ? inserted_idx[op_idx] : node_idx[op_idx]

                if out_local <= c1_len
                    source_local = c1_start + out_local - 1
                    source_local = source_local > c1_len ? source_local - c1_len : source_local
                    ox, oy, corner = _inserted_contour_node(
                        in_x, in_y, in_corners, in_offsets, ci, c1_inserted,
                        stitch_x[op_idx], stitch_y[op_idx], source_local)
                    out_local == 1 && (corner = UInt8(1))
                else
                    local2 = out_local - c1_len
                    source_local = c2_start + local2 - 1
                    source_local = source_local > c2_len ? source_local - c2_len : source_local
                    ox, oy, corner = _inserted_contour_node(
                        in_x, in_y, in_corners, in_offsets, cj, c2_inserted,
                        stitch_x[op_idx] - merge_shift_x[op_idx],
                        stitch_y[op_idx] - merge_shift_y[op_idx],
                        source_local)
                    ox += merge_shift_x[op_idx]
                    oy += merge_shift_y[op_idx]
                    local2 == 1 && (corner = UInt8(1))
                end
            end
        end

        out_x[g] = ox
        out_y[g] = oy
        out_corners[g] = corner
    end
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
    return _device_compact_scan!(slots, total, flags, n, dev, a, b)
end

# Scratch-buffer overload for hot paths whose topology size is stable. Callers
# own `a` and `b`, allowing scans to be repeated without two O(n) allocations.
function _device_compact_scan!(slots, total, flags, n::Int, dev::AbstractDevice,
                               a, b)
    n == 0 && return slots
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

# Backend-level launch (same as `@_ka_launch`, for helpers that only hold a
# KernelAbstractions backend rather than an `AbstractDevice`).
@inline function _ka_run(backend, n, builder, args...)
    builder(backend)(args...; ndrange=n)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Segmented inclusive Hillis–Steele scan step over a tuple of arrays. `seg[i]`
# is the segment key of item i (segments are contiguous, e.g. `contour_of_node`),
# so item i combines with item i-offset only when both lie in the same segment.
# `ops` holds one associative operator per array (`+` for sums, `max` for
# maxima). Each pass reads `ins` and writes `outs`, so the result is independent
# of workitem order; floating-point sums therefore reproduce run to run, though
# their association differs from a serial loop.
@inline _segscan_write!(::Tuple{}, ::Tuple{}, ::Tuple{}, i, j, take) = nothing
@inline function _segscan_write!(outs::Tuple, ins::Tuple, ops::Tuple, i, j, take)
    out = first(outs)
    in = first(ins)
    op = first(ops)
    @inbounds out[i] = take ? op(in[i], in[j]) : in[i]
    return _segscan_write!(Base.tail(outs), Base.tail(ins), Base.tail(ops), i, j, take)
end

@kernel function _segmented_scan_step_kernel!(outs, ins, ops, seg, offset, n)
    i = @index(Global)
    if i <= n
        j = i - offset
        @inbounds take = j >= 1 && seg[j] == seg[i]
        _segscan_write!(outs, ins, ops, i, j, take)
    end
end

# Inclusive segmented scan of each array in `vals` (left intact), keyed by
# `seg`. Returns the tuple of arrays holding the result: `vals` itself when
# n <= 1, otherwise one of the caller-owned scratch tuples `a`/`b` (one
# same-sized array per entry of `vals`). O(n log n) work, O(log n) launches.
function _device_segmented_scan(vals::Tuple, a::Tuple, b::Tuple, seg, n::Int,
                                ops::Tuple, dev::AbstractDevice)
    return _device_segmented_scan(vals, a, b, seg, n, ops, _ka_backend(dev))
end

function _device_segmented_scan(vals::Tuple, a::Tuple, b::Tuple, seg, n::Int,
                                ops::Tuple, backend)
    n <= 1 && return vals
    cur, other = vals, a
    offset = 1
    while offset < n
        _ka_run(backend, n, _segmented_scan_step_kernel!, other, cur, ops, seg, offset, n)
        cur, other = other, (other === a ? b : a)
        offset *= 2
    end
    return cur
end

# Convenience: allocate the scratch tuples for `_device_segmented_scan`.
function _device_segmented_scan(vals::Tuple, seg, n::Int, ops::Tuple, dev::AbstractDevice)
    backend = _ka_backend(dev)
    a = map(v -> KernelAbstractions.zeros(backend, eltype(v), n), vals)
    b = map(v -> KernelAbstractions.zeros(backend, eltype(v), n), vals)
    return _device_segmented_scan(vals, a, b, seg, n, ops, backend)
end

# Per-segment totals: the inclusive scan value at the last item of each
# segment (`zero` for an empty segment).
@kernel function _segment_last_kernel!(totals, scan, offsets, lengths, nseg)
    ci = @index(Global)
    if ci <= nseg
        @inbounds begin
            n = lengths[ci]
            totals[ci] = n > 0 ? scan[offsets[ci] + n - 1] : zero(eltype(totals))
        end
    end
end

@kernel function _prefix_lengths_finalize_kernel!(offsets, total_nodes, scan, lengths, n)
    i = @index(Global)
    if i <= n
        @inbounds begin
            incl = scan[i]
            offsets[i] = incl - lengths[i] + 1
            if i == n
                total_nodes[1] = incl
            end
        end
    end
end

# Exclusive prefix sum of `lengths` (1-based offsets) and the node total, via
# an inclusive Hillis–Steele scan (O(n log n) work, O(log n) launches) in place
# of the earlier O(n) per-item loop. Integer arithmetic, so the output matches
# the serial version exactly.
#
# `_prefix_lengths_kernel!` keeps its kernel-style call sites: `builder(backend)`
# returns a launcher whose call `(offsets, total, lengths, n; ndrange)` runs the
# scan, so `@_ka_launch dev n _prefix_lengths_kernel!(offsets, total, lengths, n)`
# is unchanged. The `ndrange` is ignored in favor of `n`.
struct _PrefixLengthsLauncher{B}
    backend::B
end
_prefix_lengths_kernel!(backend) = _PrefixLengthsLauncher(backend)

function (k::_PrefixLengthsLauncher)(offsets, total_nodes, lengths, n; ndrange=n)
    n == 0 && return nothing
    backend = k.backend
    a = KernelAbstractions.zeros(backend, Int, n)
    b = KernelAbstractions.zeros(backend, Int, n)
    copyto!(a, 1, lengths, 1, n)
    cur, other = a, b
    offset = 1
    while offset < n
        _ka_run(backend, n, _scan_step_kernel!, other, cur, offset, n)
        cur, other = other, cur
        offset *= 2
    end
    _ka_run(backend, n, _prefix_lengths_finalize_kernel!, offsets, total_nodes, cur, lengths, n)
    return nothing
end

# Node-parallel contour lookup: binary search over the monotone `offsets` for
# the last contour whose first node is <= g. An empty contour shares its offset
# with its successor, which is later and therefore wins the "last" search.
@kernel function _node_contour_search_kernel!(out_node_contour, offsets, nout,
                                              total_nodes)
    g = @index(Global)
    if g <= total_nodes
        @inbounds begin
            lo = 1
            hi = nout
            while lo < hi
                mid = (lo + hi + 1) >> 1
                if offsets[mid] <= g
                    lo = mid
                else
                    hi = mid - 1
                end
            end
            out_node_contour[g] = lo
        end
    end
end

# Same launcher pattern as `_prefix_lengths_kernel!`: the call sites stay
# `@_ka_launch dev nout _out_node_contour_kernel!(out_node_contour, offsets, lengths, nout)`
# while the work is distributed over nodes rather than contours.
struct _OutNodeContourLauncher{B}
    backend::B
end
_out_node_contour_kernel!(backend) = _OutNodeContourLauncher(backend)

function (k::_OutNodeContourLauncher)(out_node_contour, offsets, lengths, nout; ndrange=nout)
    total_nodes = length(out_node_contour)
    (nout == 0 || total_nodes == 0) && return nothing
    _ka_run(k.backend, total_nodes, _node_contour_search_kernel!,
            out_node_contour, offsets, nout, total_nodes)
    return nothing
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
