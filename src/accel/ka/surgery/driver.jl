# Surgery drivers and public dispatch: rewrite/remesh entry points, the
# reconnect loop, the full surgery pipeline, and the `surgery!` methods.

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
    density_source_data = _prepare_density_sources(density_sources)
    for i in eachindex(contours)
        contours[i] = remesh(contours[i], params;
                             _buf=remesh_buf,
                             _arc_buf=arc_buf,
                             _vnodes_buf=vnodes_buf,
                             _density_sources=density_sources,
                             _density_source_data=density_source_data)
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

"""
    _host_boundary_surgery!(state, domain, params, dev; layer_label="")

Run one CPU surgery pass on materialized contours and reload the device state
in place. Periodic surgery involves cross-seam topology (minimum-image
proximity scans, merge frame shifts) that lives on the battle-tested CPU path;
running it at the host boundary keeps GPU periodic evolution fully supported
while the device-resident scan pipeline remains unbounded-only. The transfer
cost is amortized over `n_surgery` steps of device-resident stepping.
"""
function _host_boundary_surgery!(state::DeviceContourState{T},
                                 domain::AbstractDomain,
                                 params::SurgeryParams,
                                 dev::AbstractDevice;
                                 layer_label::AbstractString="") where {T}
    contours = materialize_contours(state)
    remesh_buf = SVector{2, T}[]
    arc_buf = T[]
    vnodes_buf = SVector{2, T}[]
    _surgery_pass!(contours, domain, params, remesh_buf, arc_buf, vnodes_buf;
                   layer_label=layer_label)
    _reload_state!(state, contours, dev)
    return state
end

function surgery!(prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel},
                                       <:PeriodicDomain, T, GPU},
                  params::SurgeryParams) where {T}
    _host_boundary_surgery!(prob.device_state, prob.domain, params, prob.dev)
    return prob
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

function surgery!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, <:PeriodicDomain, T, GPU},
                  params::SurgeryParams) where {N, T}
    # Same per-layer structure as CPU multi-layer surgery; layers never
    # reconnect across layer boundaries.
    for ℓ in 1:N
        _host_boundary_surgery!(prob.device_state[ℓ], prob.domain, params, prob.dev;
                                layer_label=" layer $ℓ")
    end
    return prob
end
