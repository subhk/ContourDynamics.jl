# Flat device representation of a contour set, plus packing helpers.
#
# Device-side data layout and kernels for Dritschel-style topology surgery are
# split across this directory: types (here), filaments, pairs, rewrite, remesh,
# driver. The CPU implementation remains the reference contour model; these
# helpers mirror its Table III surgery rules on flat device arrays. The public
# GPU dispatch currently applies this backend to unbounded single-layer
# problems.

struct FlatContourTopology{T<:AbstractFloat,
                           FA<:AbstractVector{T},
                           IA<:AbstractVector{Int},
                           BA<:AbstractVector{UInt8}}
    # Flat node coordinates; contour membership is recovered through
    # `offsets`, `lengths`, `contour_of_node`, and `local_index`.
    x::FA
    y::FA
    # Per-contour metadata. `wrapx/wrapy` preserve spanning contour topology.
    pv::FA
    wrapx::FA
    wrapy::FA
    offsets::IA
    lengths::IA
    contour_of_node::IA
    local_index::IA
    corners::BA
    active::BA
end

struct DeviceReconnectionPlan{T<:AbstractFloat,
                              IA<:AbstractVector{Int},
                              FA<:AbstractVector{T},
                              BA<:AbstractVector{UInt8}}
    # Candidate contact is segment i on contour ci against segment j on contour cj.
    ci::IA
    i::IA
    cj::IA
    j::IA
    distance2::FA
    op::BA        # 1 = split, 2 = merge
    selected::BA  # 1 = selected by independent-pair planner
end

struct DeviceClosePairCandidates{IA<:AbstractVector{Int}}
    ci::IA
    i::IA
    cj::IA
    j::IA
end

struct DeviceTopologyRewritePlan{T<:AbstractFloat,
                                 IA<:AbstractVector{Int},
                                 FA<:AbstractVector{T},
                                 BA<:AbstractVector{UInt8}}
    ci::IA
    i::IA
    cj::IA
    j::IA
    op::BA                # 1 = split, 2 = merge
    valid::BA             # 1 = rewrite can be applied, 0 = leave unchanged
    node_from_first::BA   # best stitch node came from ci/i segment
    node_idx::IA          # local index of the chosen stitch node
    seg_idx::IA           # local segment index receiving the inserted stitch node
    inserted_idx::IA      # local index of the inserted stitch node after insertion
    split_reverse1::BA
    split_reverse2::BA
    merge_reverse_second::BA
    stitch_x::FA
    stitch_y::FA
    out_count::IA         # number of output contours for this operation
    out_len1::IA
    out_len2::IA
end

struct DeviceRewriteOutputs{T<:AbstractFloat,
                            FA<:AbstractVector{T},
                            IA<:AbstractVector{Int},
                            BA<:AbstractVector{UInt8}}
    x::FA
    y::FA
    pv::FA
    wrapx::FA
    wrapy::FA
    offsets::IA
    lengths::IA
    corners::BA
end

_flat_ncontours(flat::FlatContourTopology) = length(flat.lengths)
_flat_nnodes(flat::FlatContourTopology) = length(flat.x)

@kernel function _fill_u8_kernel!(out, value, n)
    i = @index(Global)
    i <= n && (out[i] = value)
end

function _device_u8_filled(dev::AbstractDevice, n::Int, value::UInt8)
    out = device_zeros(dev, UInt8, n)
    n > 0 && @_ka_launch dev n _fill_u8_kernel!(out, value, n)
    return out
end

function _flat_topology(state::DeviceContourState{T},
                        dev::AbstractDevice=CPU()) where {T}
    return FlatContourTopology(state.x, state.y, state.pv, state.wrapx,
                               state.wrapy, state.offsets, state.lengths,
                               state.contour_of_node, state.local_index,
                               state.corners,
                               _device_u8_filled(dev, length(state.lengths), UInt8(1)))
end

function _pack_flat_topology(contours::Vector{PVContour{T}}, dev::AbstractDevice=CPU()) where {T}
    # Convert ragged contour vectors into dense buffers. Device kernels cannot
    # follow Julia Vector{Vector}-style topology, so all per-node and per-contour
    # data is flattened with offset/length lookup tables.
    ncontours = length(contours)
    total = sum(nnodes, contours; init=0)
    x = Vector{T}(undef, total)
    y = Vector{T}(undef, total)
    pv = Vector{T}(undef, ncontours)
    wrapx = Vector{T}(undef, ncontours)
    wrapy = Vector{T}(undef, ncontours)
    offsets = Vector{Int}(undef, ncontours)
    lengths = Vector{Int}(undef, ncontours)
    contour_of_node = Vector{Int}(undef, total)
    local_index = Vector{Int}(undef, total)
    corners = Vector{UInt8}(undef, total)
    active = fill(UInt8(1), ncontours)

    cursor = 1
    @inbounds for (ci, c) in pairs(contours)
        offsets[ci] = cursor
        nc = nnodes(c)
        lengths[ci] = nc
        pv[ci] = c.pv
        wrapx[ci] = c.wrap[1]
        wrapy[ci] = c.wrap[2]
        for li in 1:nc
            node = c.nodes[li]
            x[cursor] = node[1]
            y[cursor] = node[2]
            contour_of_node[cursor] = ci
            local_index[cursor] = li
            corners[cursor] = c.corners[li] ? UInt8(1) : UInt8(0)
            cursor += 1
        end
    end

    return FlatContourTopology(to_device(dev, x),
                               to_device(dev, y),
                               to_device(dev, pv),
                               to_device(dev, wrapx),
                               to_device(dev, wrapy),
                               to_device(dev, offsets),
                               to_device(dev, lengths),
                               to_device(dev, contour_of_node),
                               to_device(dev, local_index),
                               to_device(dev, corners),
                               to_device(dev, active))
end

