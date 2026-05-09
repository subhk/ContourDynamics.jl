# ── Contours ─────────────────────────────────────────────

"""
    PVContour{T}(nodes, pv[, wrap[, corners]])

A node-based contour carrying a potential-vorticity jump `pv`.
`nodes` is the ordered list of material contour nodes. For contours that span a
periodic domain, `wrap` gives the shift vector that closes the final segment
back to `nodes[1]`; it defaults to zero for ordinary closed contours.

The optional `corners` vector marks Dritschel surgery corner nodes.  Corners are
fixed during corner-aware remeshing and are introduced when surgery merges or
splits contours.
"""
struct PVContour{T<:AbstractFloat}
    nodes::Vector{SVector{2, T}}
    pv::T
    wrap::SVector{2, T}
    corners::BitVector
end

function PVContour(nodes::Vector{SVector{2, T}}, pv::Real,
                   wrap::SVector{2, T},
                   corners::AbstractVector{Bool}) where {T<:AbstractFloat}
    length(corners) == length(nodes) ||
        throw(DimensionMismatch("corners length ($(length(corners))) must match nodes length ($(length(nodes)))"))
    return PVContour{T}(nodes, T(pv), wrap, BitVector(corners))
end

PVContour{T}(nodes::Vector{SVector{2, T}}, pv::Real,
             wrap::SVector{2, T},
             corners::AbstractVector{Bool}) where {T<:AbstractFloat} =
    PVContour(nodes, pv, wrap, corners)

PVContour(nodes::Vector{SVector{2, T}}, pv::Real,
          wrap::SVector{2, T}) where {T<:AbstractFloat} =
    PVContour(nodes, pv, wrap, falses(length(nodes)))

PVContour{T}(nodes::Vector{SVector{2, T}}, pv::Real,
             wrap::SVector{2, T}) where {T<:AbstractFloat} =
    PVContour(nodes, pv, wrap)

PVContour(nodes::Vector{SVector{2, T}}, pv::Real) where {T<:AbstractFloat} =
    PVContour(nodes, pv, zero(SVector{2, T}))

PVContour{T}(nodes::Vector{SVector{2, T}}, pv::Real) where {T<:AbstractFloat} =
    PVContour(nodes, pv)

"""
    nnodes(c::PVContour)

Number of nodes (vertices) in contour `c`.
"""
nnodes(c::PVContour) = length(c.nodes)

"""
    is_corner(c::PVContour, i::Int) -> Bool

Return `true` if node `i` is labelled as a fixed surgery corner.
"""
@inline function is_corner(c::PVContour, i::Int)
    @boundscheck (1 <= i <= length(c.corners) || throw(BoundsError(c.corners, i)))
    return c.corners[i]
end

"""Return the indices of all fixed surgery corner nodes on contour `c`."""
corner_indices(c::PVContour) = findall(c.corners)

"""
    is_spanning(c::PVContour) -> Bool

Return `true` if contour `c` spans the periodic domain (i.e. has a non-zero wrap vector).
"""
is_spanning(c::PVContour) = any(!iszero, c.wrap)

"""Get the next node after index `j`, handling periodic wrap for spanning contours."""
@inline function next_node(c::PVContour{T}, j::Int) where {T}
    @boundscheck (1 <= j <= length(c.nodes) || throw(BoundsError(c.nodes, j)))
    j < length(c.nodes) ? c.nodes[j + 1] : c.nodes[1] + c.wrap
end
