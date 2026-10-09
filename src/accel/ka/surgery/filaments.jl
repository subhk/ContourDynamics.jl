# Filament removal: flag contours below the area/aspect thresholds and
# stream-compact the survivors.

# Node-parallel partials for the filament test: the signed-area cross term and
# segment length of node g's outgoing segment (both relative to the contour's
# first node) and its corner flag, reduced per contour by a segmented scan.
@kernel function _filament_node_partials_kernel!(area_part, perim_part, corner_part,
                                                 x, y, wrapx, wrapy, offsets, lengths,
                                                 contour_of_node, local_index,
                                                 corners, total_nodes)
    g = @index(Global)
    if g <= total_nodes
        @inbounds begin
            ci = contour_of_node[g]
            li = local_index[g]
            nc = lengths[ci]
            off = offsets[ci]
            ox = x[off]
            oy = y[off]
            nx = li < nc ? x[g + 1] : x[off] + wrapx[ci]
            ny = li < nc ? y[g + 1] : y[off] + wrapy[ci]
            px = x[g] - ox
            py = y[g] - oy
            next_x = nx - ox
            next_y = ny - oy
            area_part[g] = px * next_y - next_x * py
            dx = nx - x[g]
            dy = ny - y[g]
            perim_part[g] = sqrt(dx * dx + dy * dy)
            corner_part[g] = iszero(corners[g]) ? zero(eltype(corner_part)) :
                             one(eltype(corner_part))
        end
    end
end

@kernel function _mark_filament_contours_kernel!(remove, area2_scan, perim_scan,
                                                 corner_scan, wrapx, wrapy,
                                                 offsets, lengths, area_min, μ,
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
            last = offsets[ci] + nc - 1
            @inbounds area2 = area2_scan[last]
            @inbounds perimeter = perim_scan[last]
            @inbounds has_corner = !iszero(corner_scan[last])

            area = abs(area2) / 2
            drop = area < area_min
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
    total_nodes = _flat_nnodes(flat)
    area_part = device_zeros(dev, T, total_nodes)
    perim_part = device_zeros(dev, T, total_nodes)
    corner_part = device_zeros(dev, T, total_nodes)
    if total_nodes > 0
        @_ka_launch dev total_nodes _filament_node_partials_kernel!(
            area_part, perim_part, corner_part, flat.x, flat.y, flat.wrapx,
            flat.wrapy, flat.offsets, flat.lengths, flat.contour_of_node,
            flat.local_index, flat.corners, total_nodes)
    end
    area2_scan, perim_scan, corner_scan = _device_segmented_scan(
        (area_part, perim_part, corner_part), flat.contour_of_node, total_nodes,
        (+, +, max), dev)
    @_ka_launch dev ncontours _mark_filament_contours_kernel!(
        remove, area2_scan, perim_scan, corner_scan, flat.wrapx, flat.wrapy,
        flat.offsets, flat.lengths, T(params.area_min), T(params.μ), ncontours)
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
        _device_compact_scan!(keep_slots, count_store, keep, ncontours, dev)
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
    # Nothing flagged: the state is already the compacted result.
    count(!iszero, to_cpu(remove)) == 0 && return state
    keep = device_zeros(dev, UInt8, ncontours)
    @_ka_launch dev ncontours _invert_remove_flags_kernel!(keep, remove, ncontours)
    outputs = _device_compact_kept_contours_outputs(flat, keep, dev)
    return _replace_device_state!(state, outputs, dev)
end
