# Filament removal: flag contours below the area/aspect thresholds and
# stream-compact the survivors.

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
    keep = device_zeros(dev, UInt8, ncontours)
    @_ka_launch dev ncontours _invert_remove_flags_kernel!(keep, remove, ncontours)
    outputs = _device_compact_kept_contours_outputs(flat, keep, dev)
    return _replace_device_state!(state, outputs, dev)
end

