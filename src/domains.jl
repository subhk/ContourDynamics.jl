# Domain helper functions

"""Square periodic domain with half-width L."""
PeriodicDomain(L::T) where {T<:AbstractFloat} = PeriodicDomain(L, L)

"""Identity wrap for unbounded domains — returns the point unchanged."""
@inline wrap_node(p::SVector{2,T}, ::UnboundedDomain) where {T} = p

"""
    wrap_node(p, domain::PeriodicDomain)

Wrap point `p` into the fundamental domain `[-Lx, Lx) × [-Ly, Ly)`.
"""
@inline function wrap_node(p::SVector{2,T}, domain::PeriodicDomain{T}) where {T}
    Lx2, Ly2 = 2 * domain.Lx, 2 * domain.Ly
    x = p[1] - Lx2 * floor((p[1] + domain.Lx) / Lx2)
    y = p[2] - Ly2 * floor((p[2] + domain.Ly) / Ly2)
    SVector{2,T}(x, y)
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
        @inbounds for i in eachindex(c.nodes)
            c.nodes[i] = wrap_node(c.nodes[i], domain)
        end
    end
    return prob
end

function wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}}) where {N, K, T}
    domain = prob.domain
    for layer in prob.layers
        for c in layer
            is_spanning(c) && continue
            @inbounds for i in eachindex(c.nodes)
                c.nodes[i] = wrap_node(c.nodes[i], domain)
            end
        end
    end
    return prob
end
