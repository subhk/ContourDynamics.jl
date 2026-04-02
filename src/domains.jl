# Domain helper functions

"""Square periodic domain with half-width L."""
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
    contour_periodic_shift(c, domain::PeriodicDomain)

Return the uniform lattice translation that moves a non-spanning contour's
centroid into the fundamental domain. Applying one shift to the whole contour
preserves its geometry across periodic seams; wrapping nodes independently does not.
"""
@inline function contour_periodic_shift(c::PVContour, domain::PeriodicDomain)
    ref = centroid(c)
    # centroid returns zero for degenerate (near-zero-area) contours;
    # fall back to minimum-image mean relative to the first node.
    # A naive arithmetic mean fails for contours straddling a periodic
    # boundary (e.g., nodes at x = -L+ε and x = L-ε average to ~0).
    if iszero(ref) && !isempty(c.nodes)
        p0 = c.nodes[1]
        Lx2, Ly2 = 2 * domain.Lx, 2 * domain.Ly
        # Accumulate minimum-image displacements from p0, then add mean back to p0.
        sum_dx = zero(eltype(p0))
        sum_dy = zero(eltype(p0))
        for k in 2:length(c.nodes)
            d = c.nodes[k] - p0
            sum_dx += d[1] - Lx2 * round(d[1] / Lx2)
            sum_dy += d[2] - Ly2 * round(d[2] / Ly2)
        end
        n = length(c.nodes)
        ref = p0 + typeof(p0)(sum_dx / n, sum_dy / n)
    end
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

"""No-op for unbounded domains — nodes don't need wrapping."""
wrap_nodes!(prob::ContourProblem{<:AbstractKernel, UnboundedDomain}) = prob
wrap_nodes!(prob::MultiLayerContourProblem{<:Any, <:Any, UnboundedDomain}) = prob

function wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}}) where {N, K, T}
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
