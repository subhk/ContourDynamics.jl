# surgery/cleanup.jl — CPU surgery stage.

# ── Filament Removal ─────────────────────────────────────

function _contour_perimeter(c::PVContour{T}) where {T}
    # Perimeter is used as a cheap thickness proxy during filament cleanup.
    n = nnodes(c)
    n < 2 && return zero(T)
    perimeter = zero(T)
    @inbounds for i in 1:n
        d = next_node(c, i) - c.nodes[i]
        perimeter += sqrt(d[1]^2 + d[2]^2)
    end
    return perimeter
end

"""Remove closed contours below the retained area or perimeter scale."""
function _is_corner_filament(c::PVContour{T}, area::T, perimeter::T, μ::T) where {T}
    any(c.corners) || return false
    # Dritschel (1988) removes contours with too few nodes (four or fewer) as
    # unresolved surgical debris.  Restricting this rule to labelled-corner
    # contours preserves coarse user-provided input contours until remeshing has
    # a chance to resolve them.
    nnodes(c) <= 4 && return true
    μ > zero(T) || return false
    perimeter <= eps(T) && return true
    width = T(2) * area / perimeter
    return area <= μ^2 || width < μ
end

"""
    remove_filaments!(contours, area_min[, μ])

Remove unresolved closed contours from `contours` in place.

Closed contours with area below `area_min` are discarded. When `μ` is supplied,
very short or thin labelled-corner fragments produced by reconnection are also
removed. Spanning contours are retained.
"""
function remove_filaments!(contours::Vector{PVContour{T}}, area_min, μ=nothing) where {T}
    # Remove unresolved closed debris. The optional μ cutoff also rejects very
    # short or thin labelled-corner fragments produced by reconnection.
    amin = T(area_min)
    μ_cut = μ === nothing ? zero(T) : T(μ)
    min_perimeter = μ === nothing ? zero(T) : T(4) * μ_cut
    filter!(contours) do c
        is_spanning(c) && return true
        nnodes(c) >= 3 || return false
        area = abs(vortex_area(c))
        perimeter = _contour_perimeter(c)
        area >= amin &&
            perimeter >= min_perimeter && !_is_corner_filament(c, area, perimeter, μ_cut)
    end
end

function _demote_obtuse_corners!(contours::Vector{PVContour{T}}) where {T}
    # Batch wrapper used after every remesh stage, when fixed corners may have
    # relaxed into ordinary smooth contour points.
    for i in eachindex(contours)
        contours[i] = _demote_obtuse_corners(contours[i])
    end
    return contours
end

# ── Top-Level Surgery ────────────────────────────────────

"""
    _check_spanning_proximity(contours, δ[, domain])

Warn if any closed contour node is within `δ` of a spanning contour node.
This situation cannot be resolved by surgery (spanning contours are exempt from
reconnection) and may indicate insufficient resolution or an overly large δ.

For `PeriodicDomain`, minimum-image distances are used.
"""
function _check_spanning_proximity(contours::Vector{PVContour{T}}, δ,
                                   domain::AbstractDomain=UnboundedDomain()) where {T}
    δ = T(δ)
    δ2 = δ^2
    # Bin spanning nodes for O(1) proximity lookup instead of O(N_spanning) per query.
    # Wrap coordinates before binning so that spanning nodes (which may have drifted
    # outside [-Lx,Lx) since wrap_nodes! skips them) land in the same bin space as
    # the wrapped closed-contour nodes.
    spanning_bins = Dict{Tuple{Int,Int}, Vector{SVector{2,T}}}()
    has_spanning = false
    _push_spanning_bin!(bins, key, sn) = (v = get!(bins, key, SVector{2,T}[]); push!(v, sn))

    for c in contours
        is_spanning(c) || continue
        has_spanning = true
        for sn in c.nodes
            sn_w = _wrap_query_pt(sn, domain)
            bx = floor(Int, sn_w[1] / δ)
            by = floor(Int, sn_w[2] / δ)
            _push_spanning_bin!(spanning_bins, (bx, by), sn)
            # Ghost entries near periodic boundaries (same 2*δ threshold
            # as _insert_bin!) so 3×3 queries find spanning nodes across seams.
            if domain isa PeriodicDomain
                Lx, Ly = domain.Lx, domain.Ly
                two_δ = 2 * δ
                near_xhi = sn_w[1] > Lx - two_δ
                near_xlo = sn_w[1] < -Lx + two_δ
                near_yhi = sn_w[2] > Ly - two_δ
                near_ylo = sn_w[2] < -Ly + two_δ
                near_xhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), by), sn)
                near_xlo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), by), sn)
                near_yhi && _push_spanning_bin!(spanning_bins, (bx, floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_ylo && _push_spanning_bin!(spanning_bins, (bx, floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
                near_xhi && near_yhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_xhi && near_ylo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] - 2Lx) / δ), floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
                near_xlo && near_yhi && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), floor(Int, (sn_w[2] - 2Ly) / δ)), sn)
                near_xlo && near_ylo && _push_spanning_bin!(spanning_bins, (floor(Int, (sn_w[1] + 2Lx) / δ), floor(Int, (sn_w[2] + 2Ly) / δ)), sn)
            end
        end
    end
    has_spanning || return

    for c in contours
        is_spanning(c) && continue
        for node in c.nodes
            node_w = _wrap_query_pt(node, domain)
            bx = floor(Int, node_w[1] / δ)
            by = floor(Int, node_w[2] / δ)
            for dbx in -1:1, dby in -1:1
                key = (bx + dbx, by + dby)
                haskey(spanning_bins, key) || continue
                for sn in spanning_bins[key]
                    r = _min_image(node - sn, domain)
                    d2 = r[1]^2 + r[2]^2
                    if d2 < δ2
                        @warn "surgery!: closed contour node within δ of spanning contour — this cannot be resolved by reconnection" distance=sqrt(d2) δ maxlog=1
                        return
                    end
                end
            end
        end
    end
end
