# Surgery drivers and public dispatch: rewrite/remesh entry points, the
# reconnect loop, the full surgery pipeline, and the `surgery!` methods.

function _device_full_rewrite_output_layout(flat::FlatContourTopology{T},
                                            plan::DeviceTopologyRewritePlan,
                                            dev::AbstractDevice) where {T}
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
        _device_compact_scan!(main_slot, main_count, main_keep, ncontours, dev)
    end
    if npairs > 0
        _device_compact_scan!(extra_slot, extra_count, extra_keep, npairs, dev)
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

# Adapter: any contour container, dev defaults to CPU.
_device_full_rewrite_output_layout(input::_UnflatContourInput,
                                   plan::DeviceTopologyRewritePlan,
                                   dev::AbstractDevice=CPU()) =
    _device_full_rewrite_output_layout(_as_flat(input, dev), plan, dev)

function _materialize_rewrite_outputs(flat::FlatContourTopology{T},
                                      plan::DeviceTopologyRewritePlan,
                                      layout,
                                      dev::AbstractDevice) where {T}
    out_x = device_zeros(dev, T, layout.total_nodes)
    out_y = device_zeros(dev, T, layout.total_nodes)
    out_corners = device_zeros(dev, UInt8, layout.total_nodes)

    if layout.total_nodes > 0
        @_ka_launch dev layout.total_nodes _materialize_rewrite_outputs_kernel!(
            out_x, out_y, out_corners, layout.offsets, layout.lengths,
            layout.out_node_contour, layout.op_index,
            layout.source_contour, layout.part, plan.ci, plan.cj,
            plan.op, plan.valid, plan.node_from_first, plan.node_idx,
            plan.seg_idx, plan.inserted_idx, plan.stitch_x, plan.stitch_y, plan.merge_shift_x, plan.merge_shift_y,
            flat.x, flat.y, flat.corners, flat.offsets,
            flat.lengths, layout.total_nodes)
    end

    return DeviceRewriteOutputs(out_x, out_y, layout.pv, layout.wrapx,
                                layout.wrapy, layout.offsets, layout.lengths,
                                out_corners)
end

# Adapter: any contour container, dev defaults to CPU.
_materialize_rewrite_outputs(input::_UnflatContourInput,
                             plan::DeviceTopologyRewritePlan,
                             layout,
                             dev::AbstractDevice=CPU()) =
    _materialize_rewrite_outputs(_as_flat(input, dev), plan, layout, dev)

# Plan + layout + materialize for any contour container and pair list; the
# input is flattened once and shared by all three stages. Domain defaults to
# unbounded.
function _device_materialize_full_rewrite_outputs(input::_DeviceContourInput,
                                                  selected_pairs::_DevicePairList,
                                                  domain::AbstractDomain=UnboundedDomain(),
                                                  dev::AbstractDevice=CPU())
    flat = _as_flat(input, dev)
    plan = _device_topology_rewrite_plan(flat, selected_pairs, domain, dev)
    layout = _device_full_rewrite_output_layout(flat, plan, dev)
    return _materialize_rewrite_outputs(flat, plan, layout, dev)
end
_device_materialize_full_rewrite_outputs(input::_DeviceContourInput,
                                         selected_pairs::_DevicePairList,
                                         dev::AbstractDevice) =
    _device_materialize_full_rewrite_outputs(input, selected_pairs, UnboundedDomain(), dev)

function _device_rewrite_contours(contours::Vector{<:PVContour},
                                  selected_pairs::_DevicePairList,
                                  domain::AbstractDomain=UnboundedDomain(),
                                  dev::AbstractDevice=CPU())
    return _unpack_rewrite_outputs(
        _device_materialize_full_rewrite_outputs(contours, selected_pairs, domain, dev))
end
_device_rewrite_contours(contours::Vector{<:PVContour}, selected_pairs::_DevicePairList,
                         dev::AbstractDevice) =
    _device_rewrite_contours(contours, selected_pairs, UnboundedDomain(), dev)

function _device_rewrite_state!(state::DeviceContourState,
                                selected_pairs::_DevicePairList,
                                domain::AbstractDomain=UnboundedDomain(),
                                dev::AbstractDevice=CPU())
    outputs = _device_materialize_full_rewrite_outputs(state, selected_pairs, domain, dev)
    return _replace_device_state!(state, outputs, dev)
end
_device_rewrite_state!(state::DeviceContourState, selected_pairs::_DevicePairList,
                       dev::AbstractDevice) =
    _device_rewrite_state!(state, selected_pairs, UnboundedDomain(), dev)

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

# Select independent pairs and rewrite in place. Host contour vectors are
# replaced wholesale; device states are rewritten on the device. Domain
# defaults to unbounded. Returns whether anything was reconnected.
const _DeviceReconnectTarget = Union{Vector{<:PVContour}, DeviceContourState}

function _device_reconnect!(contours::Vector{<:PVContour},
                            close_pairs::_DevicePairList,
                            domain::AbstractDomain=UnboundedDomain(),
                            dev::AbstractDevice=CPU())
    selected_pairs = _device_select_reconnection_pair_buffer(contours, close_pairs, domain, dev)
    length(selected_pairs.ci) == 0 && return false
    rewritten = _device_rewrite_contours(contours, selected_pairs, domain, dev)
    empty!(contours)
    append!(contours, rewritten)
    return true
end

function _device_reconnect!(state::DeviceContourState,
                            close_pairs::_DevicePairList,
                            domain::AbstractDomain=UnboundedDomain(),
                            dev::AbstractDevice=CPU())
    selected_pairs = _device_select_reconnection_pair_buffer(state, close_pairs, domain, dev)
    length(selected_pairs.ci) == 0 && return false
    _device_rewrite_state!(state, selected_pairs, domain, dev)
    return true
end

_device_reconnect!(target::_DeviceReconnectTarget, close_pairs::_DevicePairList,
                   dev::AbstractDevice) =
    _device_reconnect!(target, close_pairs, UnboundedDomain(), dev)

# Test seam: one admissible-pair search followed by one reconnect pass.
function _device_reconnect_once!(target::_DeviceReconnectTarget, δ,
                                 domain::UnboundedDomain,
                                 dev::AbstractDevice=CPU())
    close_pairs = _device_admissible_close_segment_buffer(target, δ, domain, dev)
    length(close_pairs.ci) == 0 && return false
    return _device_reconnect!(target, close_pairs, domain, dev)
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

# Device reconnection loop on the shared stall policy
# (`_reconnect_until_exhausted!` in core/surgery.jl): find admissible close
# pairs on the device, reconnect, and clean stitch artifacts, until the pair
# count stops improving.
function _device_surgery_reconnect_loop!(state::DeviceContourState{T},
                                         params::SurgeryParams,
                                         domain::AbstractDomain,
                                         dev::AbstractDevice,
                                         cleanup_reconnect_artifacts!;
                                         layer_label::AbstractString="") where {T}
    return _reconnect_until_exhausted!(
        () -> _device_admissible_close_segment_buffer(state, params.δ, domain, dev),
        pairs -> length(pairs.ci),
        pairs -> begin
            _device_reconnect!(state, pairs, domain, dev) || return false
            _device_remove_filaments!(state, params, dev)
            cleanup_reconnect_artifacts!()
            _device_remove_filaments!(state, params, dev)
            true
        end,
        () -> begin
            cleanup_reconnect_artifacts!()
            _device_remove_filaments!(state, params, dev)
        end,
        layer_label)
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
                                                   total_nodes, ncontours,
                                                   periodic, Lx, Ly, δ2)
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
                    if periodic
                        dx -= round(dx / (2 * Lx)) * (2 * Lx)
                        dy -= round(dy / (2 * Ly)) * (2 * Ly)
                    end
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
                                   domain::AbstractDomain,
                                   dev::AbstractDevice=CPU()) where {T}
    total_nodes = length(state.x)
    ncontours = length(state.lengths)
    (total_nodes == 0 || ncontours == 0) && return nothing
    flags = device_zeros(dev, UInt8, total_nodes)
    periodic, Lx, Ly = _flat_surgery_domain(domain, T)
    @_ka_launch dev total_nodes _spanning_proximity_flags_kernel!(
        flags, state.x, state.y, state.wrapx, state.wrapy, state.offsets,
        state.lengths, state.contour_of_node, total_nodes, ncontours,
        periodic, Lx, Ly, T(δ)^2)
    slots = device_zeros(dev, Int, total_nodes)
    count_store = device_zeros(dev, Int, 1)
    _device_compact_scan!(slots, count_store, flags, total_nodes, dev)
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
                                   domain::AbstractDomain, dev::AbstractDevice;
                                   layer_label::AbstractString="")
    _device_remove_filaments!(state, params, dev)
    _demote_obtuse_corners!(state, dev)
    _promote_high_curvature_corners!(state, params.δ, dev)
    _device_remesh_contours_after_surgery!(state, params, dev)

    cleanup_reconnect_artifacts!() =
        _device_remesh_contours_after_surgery!(state, params, dev)

    reconnected = _device_surgery_reconnect_loop!(state, params, domain, dev,
                                                  cleanup_reconnect_artifacts!;
                                                  layer_label)
    reconnected && cleanup_reconnect_artifacts!()

    _device_remove_filaments!(state, params, dev)
    _check_spanning_proximity(state, params.δ, domain, dev)
    return state
end

# Which kernel/domain combinations may exist on the GPU is decided once, in the
# `ContourProblem` constructor (`_check_gpu_support`); every GPU problem that
# reaches here runs the same device pipeline.
function surgery!(prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, GPU},
                  params::SurgeryParams) where {T}
    _device_surgery_pipeline!(_device_state(prob), params, prob.domain, prob.dev)
    return prob
end

"""
    _device_multilayer_surgery!(states, params, domain, dev)

Run the device-resident surgery pipeline independently on each layer's state.
Layers never reconnect across layer boundaries (the CPU multi-layer surgery has
the same per-layer structure).
"""
function _device_multilayer_surgery!(states::NTuple{N, <:DeviceContourState},
                                     params::SurgeryParams,
                                     domain::AbstractDomain,
                                     dev::AbstractDevice) where {N}
    for ℓ in 1:N
        # Same per-layer label as the CPU multi-layer `surgery!`, so stall
        # warnings identify the layer on either backend.
        _device_surgery_pipeline!(states[ℓ], params, domain, dev;
                                  layer_label=" layer $ℓ")
    end
    return states
end

function surgery!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, <:AbstractDomain, T, GPU},
                  params::SurgeryParams) where {N, T}
    _device_multilayer_surgery!(_device_state(prob), params, prob.domain, prob.dev)
    return prob
end
