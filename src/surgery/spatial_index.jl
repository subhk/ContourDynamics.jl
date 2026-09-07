# surgery/spatial_index.jl — CPU surgery stage.

# ── Periodic Helpers ────────────────────────────────────

"""Wrap a scalar coordinate to the canonical interval [-L, L) using floor."""
@inline function _wrap_coord(x::T, L::T) where {T}
    L2 = 2 * L
    return x - floor((x + L) / L2) * L2
end

"""
Minimum-image displacement vector for a periodic domain using round.

Note: uses round (nearest-integer) rather than floor, giving the interval (-L, L]
instead of [-L, L). The difference matters only at exact boundary values, which
are astronomically unlikely with floating-point arithmetic.
"""
@inline function _min_image(r::SVector{2,Tr}, domain::PeriodicDomain{Td}) where {Tr, Td}
    T = promote_type(Tr, Td)
    Lx2 = 2 * T(domain.Lx)
    Ly2 = 2 * T(domain.Ly)
    SVector{2,T}(T(r[1]) - round(T(r[1]) / Lx2) * Lx2, T(r[2]) - round(T(r[2]) / Ly2) * Ly2)
end
@inline _min_image(r::SVector{2}, ::UnboundedDomain) = r

# ── Spatial Index ────────────────────────────────────────

struct SpatialIndex{T<:AbstractFloat}
    bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}  # (bin_x, bin_y) => [(contour_idx, node_idx)]
    bin_size::T
end

"""
Build a spatial index for all contour segments, binned by grid of size `bin_size`.
Each segment is binned at evenly spaced points no farther than `bin_size` apart
so that long segments whose interiors cross bin boundaries remain discoverable
via neighbour-bin queries.

For `PeriodicDomain`, coordinates are wrapped to the canonical domain and ghost
entries are inserted near boundaries so that segments physically close across
the periodic seam appear in adjacent bins.
"""
function build_spatial_index(contours::Vector{PVContour{T}}, bin_size,
                             domain::AbstractDomain=UnboundedDomain()) where {T}
    bin_size = T(bin_size)
    bins = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()

    for (ci, c) in enumerate(contours)
        is_spanning(c) && continue  # spanning contours never participate in reconnection
        nc = nnodes(c)
        nc < 3 && continue  # degenerate contours cannot participate in surgery
        for ni in 1:nc
            a = c.nodes[ni]
            b = next_node(c, ni)
            seg = b - a
            seg_len = sqrt(seg[1]^2 + seg[2]^2)
            # Bin at evenly spaced points along the segment, spaced at most bin_size
            # apart. This ensures every point on the segment is within bin_size/2 of
            # a binned point, so the 3×3 neighbourhood query in find_close_segments
            # can discover close pairs even when Δ_max >> δ.
            n_samples = max(2, ceil(Int, seg_len / bin_size) + 1)
            for k in 0:(n_samples - 1)
                t = T(k) / T(n_samples - 1)
                pt = a + t * seg
                _insert_bin!(bins, pt, ci, ni, bin_size, domain)
            end
        end
    end

    return SpatialIndex(bins, bin_size)
end

@inline function _push_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                            key::Tuple{Int,Int}, ci::Int, ni::Int)
    entry = (ci, ni)
    if !haskey(bins, key)
        bins[key] = [entry]
    else
        # Deduplicate: same segment can be sampled at multiple points in the
        # same bin.  Only need one entry per segment per bin; duplicates cause
        # redundant distance computations in find_close_segments.
        vec = bins[key]
        entry ∈ vec || push!(vec, entry)
    end
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                    pt::SVector{2,T}, ci::Int, ni::Int, δ::T,
                    ::UnboundedDomain) where {T}

    bx = floor(Int, pt[1] / δ)
    by = floor(Int, pt[2] / δ)
    _push_bin!(bins, (bx, by), ci, ni)
end

function _insert_bin!(bins::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}},
                      pt::SVector{2,T}, ci::Int, ni::Int, δ::T,
                      domain::PeriodicDomain{T}) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    x_w = _wrap_coord(pt[1], Lx)
    y_w = _wrap_coord(pt[2], Ly)

    bx = floor(Int, x_w / δ)
    by = floor(Int, y_w / δ)
    _push_bin!(bins, (bx, by), ci, ni)

    # Ghost entries near periodic boundaries so that the 3×3 neighbour query
    # in find_close_segments discovers segments close across the seam.
    # Use 2*δ threshold: the query extends ±1 bin, so a segment up to
    # 2*δ from the seam can have its periodic image within δ of a
    # query on the opposite side.
    near_xhi = x_w > Lx  - 2δ
    near_xlo = x_w < -Lx + 2δ
    near_yhi = y_w > Ly  - 2δ
    near_ylo = y_w < -Ly + 2δ

    # Edge ghosts
    near_xhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), by), ci, ni)
    near_xlo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), by), ci, ni)
    near_yhi && _push_bin!(bins, (bx, floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_ylo && _push_bin!(bins, (bx, floor(Int, (y_w + 2Ly) / δ)), ci, ni)

    # Corner ghosts
    near_xhi && near_yhi && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_xhi && near_ylo && _push_bin!(bins, (floor(Int, (x_w - 2Lx) / δ), floor(Int, (y_w + 2Ly) / δ)), ci, ni)
    near_xlo && near_yhi && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), floor(Int, (y_w - 2Ly) / δ)), ci, ni)
    near_xlo && near_ylo && _push_bin!(bins, (floor(Int, (x_w + 2Lx) / δ), floor(Int, (y_w + 2Ly) / δ)), ci, ni)
end
