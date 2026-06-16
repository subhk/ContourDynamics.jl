# Domain helper functions

"""
    PeriodicDomain(L)

Construct a square periodic domain with half-width `L`, equivalent to
`PeriodicDomain(L, L)`.
"""
PeriodicDomain(L::T) where {T<:AbstractFloat} = PeriodicDomain(L, L)

"""Identity wrap for unbounded domains — returns the point unchanged."""
@inline wrap_node(p::SVector{2,T}, ::UnboundedDomain) where {T} = p

"""
    wrap_node(p, domain::PeriodicDomain)

Wrap point `p` into the fundamental domain `[-Lx, Lx) × [-Ly, Ly)`.
"""
@inline function wrap_node(p::SVector{2,Tp}, domain::PeriodicDomain{Td}) where {Tp, Td}
    T = promote_type(Tp, Td)
    Lx2, Ly2 = 2 * domain.Lx, 2 * domain.Ly
    x = p[1] - Lx2 * floor((p[1] + domain.Lx) / Lx2)
    y = p[2] - Ly2 * floor((p[2] + domain.Ly) / Ly2)
    SVector{2,T}(x, y)
end

"""
    _periodic_reference_point(nodes, domain::PeriodicDomain)

Area-weighted centroid of `nodes` computed on minimum-image-unwrapped
coordinates (anchored at the first node), falling back to the node mean for
degenerate (near-zero-area) contours.

Unwrapping first makes this robust to contours stored straddling a periodic
seam, where a raw centroid is meaningless (e.g. nodes at x = -L+ε and x = L-ε
would average to ~0). For a contour stored contiguously within one period the
unwrap is the identity, so this reproduces `centroid` exactly.

Assumes every node lies within half a period (`Lx`/`Ly`) of the first node, so
the single-anchor minimum-image unwrap places it in the right image — true for
non-spanning contours smaller than the domain half-width. A non-spanning contour
wider than that would have far nodes folded to the wrong image; the returned
shift is still an exact lattice translation (geometry/area preserved), only its
choice of period image may be suboptimal.
"""
@inline function _periodic_reference_point(nodes::AbstractVector{SVector{2,T}},
                                           domain::PeriodicDomain) where {T}
    n = length(nodes)
    p0 = nodes[1]
    Lx2 = T(2) * T(domain.Lx)
    Ly2 = T(2) * T(domain.Ly)
    unwrap(p) = begin
        dx = p[1] - p0[1]
        dy = p[2] - p0[2]
        SVector{2,T}(p0[1] + dx - Lx2 * round(dx / Lx2),
                     p0[2] + dy - Ly2 * round(dy / Ly2))
    end

    area2 = zero(T)
    cx = zero(T)
    cy = zero(T)
    sx = zero(T)
    sy = zero(T)
    @inbounds for i in 1:n
        ui = unwrap(nodes[i])
        uj = unwrap(nodes[i < n ? i + 1 : 1])
        cross = ui[1] * uj[2] - uj[1] * ui[2]
        area2 += cross
        cx += (ui[1] + uj[1]) * cross
        cy += (ui[2] + uj[2]) * cross
        sx += ui[1]
        sy += ui[2]
    end
    if n < 3 || abs(area2) <= T(2) * eps(T)
        return SVector{2,T}(sx / n, sy / n)
    end
    inv3A2 = one(T) / (T(3) * area2)
    return SVector{2,T}(cx * inv3A2, cy * inv3A2)
end

"""
    contour_periodic_shift(c, domain::PeriodicDomain)

Return the uniform lattice translation that moves a non-spanning contour's
centroid into the fundamental domain. Applying one shift to the whole contour
preserves its geometry across periodic seams; wrapping nodes independently does not.
"""
@inline function contour_periodic_shift(c::PVContour, domain::PeriodicDomain)
    isempty(c.nodes) && return zero(eltype(c.nodes))
    ref = _periodic_reference_point(c.nodes, domain)
    return wrap_node(ref, domain) - ref
end

"""
    wrap_nodes!(prob::ContourProblem{K, PeriodicDomain{T}})

Wrap all non-spanning contour nodes into the fundamental domain.
Spanning contours are left untouched since their positions encode
the cross-domain topology via the wrap vector.
"""
function wrap_nodes!(prob::ContourProblem{K, PeriodicDomain{T}}) where {K, T}
    domain = prob.domain
    for c in prob.contours
        is_spanning(c) && continue
        shift = contour_periodic_shift(c, domain)
        iszero(shift) && continue
        @inbounds for i in eachindex(c.nodes)
            c.nodes[i] += shift
        end
    end
    return prob
end

"""No-op for unbounded domains; nodes do not need periodic wrapping."""
wrap_nodes!(prob::ContourProblem{<:AbstractKernel, UnboundedDomain}) = prob
wrap_nodes!(prob::MultiLayerContourProblem{<:Any, <:Any, UnboundedDomain}) = prob

"""
    wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, CPU})

Wrap all non-spanning contours in every layer of a periodic multi-layer problem
into the fundamental domain. Spanning contours are left untouched.
"""
function wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, CPU}) where {N, K, T}
    domain = prob.domain
    for layer in prob.layers
        for c in layer
            is_spanning(c) && continue
            shift = contour_periodic_shift(c, domain)
            iszero(shift) && continue
            @inbounds for i in eachindex(c.nodes)
                c.nodes[i] += shift
            end
        end
    end
    return prob
end
