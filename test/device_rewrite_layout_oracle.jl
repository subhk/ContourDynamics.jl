# Serial reference implementations of the parallel device surgery pipeline.
#
# `ContourDynamics._device_full_rewrite_output_layout` computes this same
# layout with a Hillis-Steele scan across several kernels. This straightforward
# host version is the oracle it is checked against
# (`_assert_device_layout_matches_host`), so it lives with the tests rather
# than in `src` — nothing in the package calls it. The same goes for the
# host-side pair selection and replaced-contours-only rewrite entry points
# further below: they are serial references for the device kernels, reachable
# only from `test_device.jl`.

function _full_rewrite_output_layout(contours::Vector{PVContour{T}},
                                     plan::ContourDynamics.DeviceTopologyRewritePlan,
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

# Serial host reference for `_select_independent_pairs_kernel!`: greedy
# distance-ranked matching with each contour used at most once.
function _device_select_reconnection_pairs_from_plan(contours::Vector{PVContour{T}},
                                                     close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                                     plan::ContourDynamics.DeviceReconnectionPlan{T}) where {T}
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

function _device_select_reconnection_pairs(contours::Vector{PVContour{T}},
                                           close_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                           dev::AbstractDevice=CPU()) where {T}
    isempty(close_pairs) && return Tuple{Int,Int,Int,Int}[]
    plan = ContourDynamics._device_reconnection_plan(contours, close_pairs, dev)
    return _device_select_reconnection_pairs_from_plan(contours, close_pairs, plan)
end

function _device_select_reconnection_pairs(contours::Vector{PVContour{T}},
                                           candidates::ContourDynamics.DeviceClosePairCandidates,
                                           dev::AbstractDevice=CPU()) where {T}
    return ContourDynamics._unpack_close_pair_candidates(
        ContourDynamics._device_select_reconnection_pair_buffer(contours, candidates, dev))
end

# Serial reference layout containing only the rewritten contours (parts 1/2),
# unlike the full layout above which also carries untouched contours.
function _rewrite_output_layout(contours::Vector{PVContour{T}},
                                plan::ContourDynamics.DeviceTopologyRewritePlan,
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

function _device_materialize_rewrite_outputs(contours::Vector{PVContour{T}},
                                             selected_pairs::Vector{Tuple{Int,Int,Int,Int}},
                                             dev::AbstractDevice=CPU()) where {T}
    plan = ContourDynamics._device_topology_rewrite_plan(contours, selected_pairs, dev)
    layout = _rewrite_output_layout(contours, plan, dev)
    return ContourDynamics._materialize_rewrite_outputs(contours, plan, layout, dev)
end
