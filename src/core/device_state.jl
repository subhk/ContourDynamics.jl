# Flat device-resident contour topology used by GPU problem state.

"""
    DeviceContourState{T}

Structure-of-arrays representation of a vector of [`PVContour`](@ref)s on a
target device. Coordinates, PV jumps, wrap vectors, corner flags, and local
topology indices are stored in flat arrays so KernelAbstractions kernels can
traverse contour geometry without chasing host-side Julia objects.

Use [`materialize_contours`](@ref) to copy the state back to ordinary
`Vector{PVContour}` form for output, plotting, or inspection.
"""
mutable struct DeviceContourState{T<:AbstractFloat,
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
    contour_of_node::IA
    local_index::IA
end

"""
    DeviceContourState(contours, dev=CPU())

Pack CPU contours into the flat device-resident layout used by GPU-tagged
problems. Passing `CPU()` is mainly useful for tests and for sharing the same
topology code path without device transfer.
"""
function DeviceContourState(contours::Vector{PVContour{T}},
                            dev::AbstractDevice=CPU()) where {T}
    ncontours = length(contours)
    total = sum(nnodes, contours; init=0)

    # Build ordinary host arrays first, then transfer once per field. This keeps
    # packing deterministic and avoids many small device writes during setup.
    x = Vector{T}(undef, total)
    y = Vector{T}(undef, total)
    pv = Vector{T}(undef, ncontours)
    wrapx = Vector{T}(undef, ncontours)
    wrapy = Vector{T}(undef, ncontours)
    offsets = Vector{Int}(undef, ncontours)
    lengths = Vector{Int}(undef, ncontours)
    corners = Vector{UInt8}(undef, total)
    contour_of_node = Vector{Int}(undef, total)
    local_index = Vector{Int}(undef, total)

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
            corners[cursor] = c.corners[li] ? UInt8(1) : UInt8(0)
            contour_of_node[cursor] = ci
            local_index[cursor] = li
            cursor += 1
        end
    end

    return DeviceContourState(to_device(dev, x),
                              to_device(dev, y),
                              to_device(dev, pv),
                              to_device(dev, wrapx),
                              to_device(dev, wrapy),
                              to_device(dev, offsets),
                              to_device(dev, lengths),
                              to_device(dev, corners),
                              to_device(dev, contour_of_node),
                              to_device(dev, local_index))
end

@inline _device_state_nnodes(state::DeviceContourState) = length(state.x)

"""
    _layer_state_ranges(states)

Flat index range of each layer's nodes when layers are concatenated in order.
"""
function _layer_state_ranges(states::NTuple{N, <:DeviceContourState}) where {N}
    ranges = Vector{UnitRange{Int}}(undef, N)
    idx = 1
    for ℓ in 1:N
        n = _device_state_nnodes(states[ℓ])
        ranges[ℓ] = idx:(idx + n - 1)
        idx += n
    end
    return ranges
end

"""
    materialize_contours(state::DeviceContourState)

Copy a flat device contour state back to a CPU `Vector{PVContour}` while
preserving PV jumps, periodic wrap vectors, and surgery corner flags.
"""
function materialize_contours(state::DeviceContourState{T}) where {T}
    # Bring every field back before rebuilding PVContour objects. Reconstructing
    # directly from device arrays would force scalar indexing on GPU backends.
    x = to_cpu(state.x)
    y = to_cpu(state.y)
    pv = to_cpu(state.pv)
    wrapx = to_cpu(state.wrapx)
    wrapy = to_cpu(state.wrapy)
    offsets = to_cpu(state.offsets)
    lengths = to_cpu(state.lengths)
    corners = to_cpu(state.corners)

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

_build_device_state(contours::Vector{PVContour{T}}, ::CPU) where {T} = nothing
_build_device_state(contours::Vector{PVContour{T}}, dev::GPU) where {T} =
    DeviceContourState(contours, dev)

_build_layer_device_state(layers::NTuple{N, Vector{PVContour{T}}}, ::CPU) where {N, T} =
    nothing
_build_layer_device_state(layers::NTuple{N, Vector{PVContour{T}}}, dev::GPU) where {N, T} =
    ntuple(i -> DeviceContourState(layers[i], dev), N)
