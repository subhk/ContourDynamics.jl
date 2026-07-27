# Serial reference implementation of the parallel device rewrite layout.
#
# `ContourDynamics._device_full_rewrite_output_layout` computes this same
# layout with a Hillis-Steele scan across several kernels. This straightforward
# host version is the oracle it is checked against
# (`_assert_device_layout_matches_host`), so it lives with the tests rather
# than in `src` — nothing in the package calls it.

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
