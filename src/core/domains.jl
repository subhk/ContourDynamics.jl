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
    Lx2 = T(2) * T(domain.Lx)
    Ly2 = T(2) * T(domain.Ly)
    rx, ry = _unwrapped_centroid_core(i -> (nodes[i][1], nodes[i][2]), n, Lx2, Ly2)
    return SVector{2,T}(rx, ry)
end

# Scalar core of the unwrapped area-weighted centroid, shared verbatim by the
# CPU `_periodic_reference_point` above and the flat-array device kernel in
# evolution_buffers.jl (`_compute_state_shifts_ka!`). Both paths must produce
# bit-identical reference points, so the accumulation order lives in exactly
# one place. `getnode(i)` returns the i-th node as an `(x, y)` tuple.
@inline function _unwrapped_centroid_core(getnode::F, n::Int,
                                          Lx2::T, Ly2::T) where {F, T}
    p0x, p0y = getnode(1)
    area2 = zero(T)
    cx = zero(T)
    cy = zero(T)
    sx = zero(T)
    sy = zero(T)
    @inbounds for i in 1:n
        xi, yi = getnode(i)
        xj, yj = getnode(i < n ? i + 1 : 1)
        dxi = xi - p0x
        dyi = yi - p0y
        uix = p0x + dxi - Lx2 * round(dxi / Lx2)
        uiy = p0y + dyi - Ly2 * round(dyi / Ly2)
        dxj = xj - p0x
        dyj = yj - p0y
        ujx = p0x + dxj - Lx2 * round(dxj / Lx2)
        ujy = p0y + dyj - Ly2 * round(dyj / Ly2)
        cross = uix * ujy - ujx * uiy
        area2 += cross
        cx += (uix + ujx) * cross
        cy += (uiy + ujy) * cross
        sx += uix
        sy += uiy
    end
    if n < 3 || abs(area2) <= T(2) * eps(T)
        return sx / n, sy / n
    end
    inv3A2 = one(T) / (T(3) * area2)
    return cx * inv3A2, cy * inv3A2
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

# Shared body of the periodic wrap_nodes! methods: shift each non-spanning
# contour by one uniform lattice translation.
function _wrap_contours!(contours, domain::PeriodicDomain)
    for c in contours
        is_spanning(c) && continue
        shift = contour_periodic_shift(c, domain)
        iszero(shift) && continue
        @inbounds for i in eachindex(c.nodes)
            c.nodes[i] += shift
        end
    end
    return contours
end

"""
    wrap_nodes!(prob::ContourProblem{K, PeriodicDomain{T}})

Wrap all non-spanning contour nodes into the fundamental domain.
Spanning contours are left untouched since their positions encode
the cross-domain topology via the wrap vector.
"""
function wrap_nodes!(prob::ContourProblem{K, PeriodicDomain{T}}) where {K, T}
    _wrap_contours!(prob.contours, prob.domain)
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
    for layer in prob.layers
        _wrap_contours!(layer, prob.domain)
    end
    return prob
end
