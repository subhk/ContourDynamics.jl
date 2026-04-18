# Minimum segment count to enable threading in energy pair loops.
# Below this threshold, thread spawn overhead dominates the computation.
const _THREADING_THRESHOLD = 64

"""
    @_maybe_threads cond for-loop

Apply `Threads.@threads` to `for-loop` only when `cond` is true at runtime.
Falls back to a plain serial loop otherwise, avoiding thread-spawn overhead.
"""
macro _maybe_threads(cond, loop)
    @assert loop.head === :for
    # Inject @inbounds into both paths for consistent semantics.
    # Tasks don't inherit @inbounds from the caller scope, so the
    # threaded path needs it explicitly. The serial path also gets
    # @inbounds so that both paths behave identically.
    inbounds_loop = Expr(:for, loop.args[1], Expr(:macrocall, Symbol("@inbounds"), nothing, loop.args[2]))
    threaded = esc(:(Threads.@threads $inbounds_loop))
    serial = esc(inbounds_loop)
    quote
        if $(esc(cond))
            $threaded
        else
            $serial
        end
    end
end

"""
    vortex_area(c::PVContour)

Signed area enclosed by contour `c` using the shoelace formula.
"""
function vortex_area(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    is_spanning(c) && return zero(T)  # area undefined for spanning contours
    A = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        A += nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
    end
    return A / 2
end

"""
    centroid(c::PVContour)

Centroid of the region enclosed by contour `c`, via Green's theorem.
"""
function centroid(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(SVector{2, T})
    A = vortex_area(c)
    abs(A) < eps(T) && return sum(nodes) / n

    cx = zero(T)
    cy = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        cross = nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
        cx += (nodes[i][1] + nxt[1]) * cross
        cy += (nodes[i][2] + nxt[2]) * cross
    end
    inv6A = one(T) / (6 * A)
    return SVector{2, T}(cx * inv6A, cy * inv6A)
end

"""
    ellipse_moments(c::PVContour)

Second moments → (aspect_ratio, orientation_angle).
"""
function ellipse_moments(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    A = vortex_area(c)

    # Guard against degenerate contours (zero or near-zero area)
    if abs(A) < eps(T) || n < 3
        return (one(T), zero(T))
    end

    ctr = centroid(c)

    Jxx = zero(T)
    Jyy = zero(T)
    Jxy = zero(T)

    @inbounds for i in 1:n
        nxt = next_node(c, i)
        xi, yi = nodes[i][1] - ctr[1], nodes[i][2] - ctr[2]
        xj, yj = nxt[1] - ctr[1], nxt[2] - ctr[2]
        cross = xi * yj - xj * yi
        Jxx += (xi^2 + xi * xj + xj^2) * cross
        Jyy += (yi^2 + yi * yj + yj^2) * cross
        Jxy += (xi * yj + 2 * xi * yi + 2 * xj * yj + xj * yi) * cross
    end

    Jxx /= 12
    Jyy /= 12
    Jxy /= 24

    # Use signed area so that CW contours (A < 0) produce positive moments
    Jxx /= A
    Jyy /= A
    Jxy /= A

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy^2
    disc = sqrt(max(zero(T), trace^2 / 4 - det))
    lambda1 = trace / 2 + disc
    lambda2 = trace / 2 - disc

    # Guard against near-degenerate contours: if the trace (sum of eigenvalues)
    # is negligible, the contour is essentially a point — return unit aspect ratio.
    if trace < eps(T)
        return (one(T), zero(T))
    end
    lambda2_safe = max(lambda2, trace * eps(T) * T(100))
    aspect_ratio = sqrt(max(one(T), lambda1 / lambda2_safe))
    angle = T(0.5) * atan(2 * Jxy, Jxx - Jyy)

    return (aspect_ratio, angle)
end

"""
    circulation(prob)

Total circulation `Γ = ∑ qᵢ Aᵢ` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).

!!! warning
    Spanning contours have undefined area and are silently excluded.
    If your problem uses spanning contours (e.g. from [`beta_staircase`](@ref)),
    the returned circulation only reflects closed contours.
"""
function circulation(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * vortex_area(c)
    end
    return s
end

vortex_area(prob::ContourProblem) = vortex_area.(prob.contours)

function vortex_area(prob::MultiLayerContourProblem{N}) where {N}
    ntuple(i -> vortex_area.(prob.layers[i]), Val(N))
end

"""
    enstrophy(prob)

Enstrophy `½ ∑ qᵢ² Aᵢ` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).

Uses signed area `Aᵢ` from the shoelace formula (positive for CCW, negative
for CW contours), so contributions from inner boundaries are subtracted.
Spanning contours are excluded (see [`circulation`](@ref)).

!!! warning
    This diagnostic is exact when contours encode disjoint PV regions through
    their signed areas, but it does not reconstruct the fully squared piecewise
    PV field for arbitrary nested multi-jump contour sets. In those cases the
    missing cross-terms make the result only approximate.
"""
function enstrophy(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv^2 * vortex_area(c)
    end
    return s / 2
end

"""
    angular_momentum(prob)

Angular momentum `∑ qᵢ ∫ r² dA` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
function angular_momentum(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * _second_moment_r2(c)
    end
    return s
end

function _second_moment_r2(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    is_spanning(c) && return zero(T)  # moment undefined for spanning contours
    s = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        xi, yi = nodes[i][1], nodes[i][2]
        xj, yj = nxt[1], nxt[2]
        cross = xi * yj - xj * yi
        s += (xi^2 + xi * xj + xj^2) * cross
        s += (yi^2 + yi * yj + yj^2) * cross
    end
    return s / 12
end

# Fallback for unsupported kernel/domain combinations
function energy(prob::ContourProblem)
    throw(ArgumentError(
        "energy is not implemented for $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Supported: EulerKernel/QGKernel/SQGKernel on UnboundedDomain, EulerKernel/QGKernel on PeriodicDomain."))
end

function circulation(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv * vortex_area(c)
        end
    end
    return s
end

function enstrophy(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv^2 * vortex_area(c)
        end
    end
    return s / 2
end

function angular_momentum(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv * _second_moment_r2(c)
        end
    end
    return s
end
