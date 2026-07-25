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
    @_energy_segment_loop partial workspace n for-loop

Reset `partial[1:n]`, run `for-loop` with the diagnostics threading policy,
and return `sum(partial[1:n])`.
"""
macro _energy_segment_loop(partial, workspace, n, loop)
    @assert loop.head === :for
    inbounds_loop = Expr(:for, loop.args[1], Expr(:macrocall, Symbol("@inbounds"), nothing, loop.args[2]))
    threaded = esc(:(Threads.@threads $inbounds_loop))
    serial = esc(inbounds_loop)
    quote
        local $(esc(partial)) = $(esc(workspace))
        @inbounds for k in 1:$(esc(n))
            $(esc(partial))[k] = zero(eltype($(esc(partial))))
        end
        # Match the velocity threading policy: only spawn tasks when more than one
        # thread is available. `Threads.@threads` allocates task/partition
        # machinery even at nthreads()==1, which is pure overhead repeated on
        # every contour-pair term of the O(C²) energy double sum.
        if Threads.nthreads() > 1 && $(esc(n)) >= _THREADING_THRESHOLD
            $threaded
        else
            $serial
        end
        # Reduce partial[1:n] without materializing a SubArray — `sum(@view ...)`
        # allocated ~600 B per term on the hot energy path.
        local _energy_acc = zero(eltype($(esc(partial))))
        @inbounds for k in 1:$(esc(n))
            _energy_acc += $(esc(partial))[k]
        end
        _energy_acc
    end
end

@inline _valid_energy_contour(c) = nnodes(c) >= 3 && !is_spanning(c)

@inline _period_lengths(Lx::T, Ly::T) where {T} = (T(2) * Lx, T(2) * Ly)

function _max_valid_energy_nnodes(contours)
    return maximum((nnodes(c) for c in contours if _valid_energy_contour(c)); init=0)
end

"""
    @_valid_contour_pairs ci cj partial contours scratch begin ... end

Loop over valid closed contour pairs, reusing the caller-owned `scratch` vector
as the per-pair workspace `partial`, resized to the largest valid contour.
"""
macro _valid_contour_pairs(ci, cj, partial, contours, scratch, body)
    contours_var = gensym(:contours)
    quote
        local $contours_var = $(esc(contours))
        # Reuse the caller's scratch buffer instead of allocating a fresh
        # workspace on every energy() call.
        local $(esc(partial)) = $(esc(scratch))
        resize!($(esc(partial)), _max_valid_energy_nnodes($contours_var))
        for $(esc(ci)) in $contours_var
            _valid_energy_contour($(esc(ci))) || continue
            for $(esc(cj)) in $contours_var
                _valid_energy_contour($(esc(cj))) || continue
                $(esc(body))
            end
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

Total circulation `Γ = ∑ qᵢ Aᵢ` of a `ContourProblem` or
`MultiLayerContourProblem`.

!!! warning
    Spanning contours have undefined area and are silently excluded.
    If your problem uses spanning contours (e.g. from `beta_staircase`),
    the returned circulation only reflects closed contours.
"""
function circulation(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * vortex_area(c)
    end
    return s
end

"""Return signed areas for every contour in a single-layer problem."""
vortex_area(prob::ContourProblem) = vortex_area.(prob.contours)

"""Return one vector of signed contour areas per layer."""
function vortex_area(prob::MultiLayerContourProblem{N}) where {N}
    ntuple(i -> vortex_area.(prob.layers[i]), Val(N))
end

"""
    enstrophy(prob)

Enstrophy `½ ∑ qᵢ² Aᵢ` of a `ContourProblem` or
`MultiLayerContourProblem`.

Uses signed area `Aᵢ` from the shoelace formula (positive for CCW, negative
for CW contours), so contributions from inner boundaries are subtracted.
Spanning contours are excluded (see `circulation`).

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

Angular momentum `∑ qᵢ ∫ r² dA` of a `ContourProblem` or
`MultiLayerContourProblem`.
"""
function angular_momentum(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * _second_moment_r2(c)
    end
    return s
end

function _second_moment_r2(c::PVContour{T}) where {T}
    # Green's-theorem polygon formula for ∫(x²+y²)dA. Inner boundaries keep
    # their signed orientation, matching the area convention used elsewhere.
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

@kernel function _state_area_moment_kernel!(area, moment, x, y, wrapx, wrapy,
                                            offsets, lengths, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        T = eltype(area)
        n = lengths[ci]
        if n < 3 || !iszero(wrapx[ci]) || !iszero(wrapy[ci])
            area[ci] = zero(T)
            moment[ci] = zero(T)
        else
            off = offsets[ci]
            area2 = zero(T)
            m = zero(T)
            @inbounds for li in 1:n
                g = off + li - 1
                xi = x[g]
                yi = y[g]
                if li < n
                    xj = x[g + 1]
                    yj = y[g + 1]
                else
                    xj = x[off] + wrapx[ci]
                    yj = y[off] + wrapy[ci]
                end
                cross = xi * yj - xj * yi
                area2 += cross
                m += (xi * xi + xi * xj + xj * xj) * cross
                m += (yi * yi + yi * yj + yj * yj) * cross
            end
            area[ci] = area2 / T(2)
            moment[ci] = m / T(12)
        end
    end
end

function _state_area_moment(state::DeviceContourState{T},
                            dev::AbstractDevice=CPU()) where {T}
    ncontours = length(state.lengths)
    area = device_zeros(dev, T, ncontours)
    moment = device_zeros(dev, T, ncontours)
    if ncontours > 0
        @_ka_launch dev ncontours _state_area_moment_kernel!(
            area, moment, state.x, state.y, state.wrapx, state.wrapy,
            state.offsets, state.lengths, ncontours)
    end
    return area, moment
end

_state_vortex_area(state::DeviceContourState, dev::AbstractDevice=CPU()) =
    first(_state_area_moment(state, dev))

@kernel function _state_weighted_diagnostic_kernel!(partial, area, moment, pv,
                                                    mode, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        q = pv[ci]
        if mode == UInt8(1)
            partial[ci] = q * area[ci]
        elseif mode == UInt8(2)
            partial[ci] = q * q * area[ci] / eltype(partial)(2)
        else
            partial[ci] = q * moment[ci]
        end
    end
end

function _state_weighted_diagnostic(state::DeviceContourState{T}, mode::UInt8,
                                    dev::AbstractDevice=CPU()) where {T}
    area, moment = _state_area_moment(state, dev)
    ncontours = length(state.lengths)
    partial = device_zeros(dev, T, ncontours)
    if ncontours > 0
        @_ka_launch dev ncontours _state_weighted_diagnostic_kernel!(
            partial, area, moment, state.pv, mode, ncontours)
    end
    return sum(to_cpu(partial))
end

_state_circulation(state::DeviceContourState, dev::AbstractDevice=CPU()) =
    _state_weighted_diagnostic(state, UInt8(1), dev)
_state_enstrophy(state::DeviceContourState, dev::AbstractDevice=CPU()) =
    _state_weighted_diagnostic(state, UInt8(2), dev)
_state_angular_momentum(state::DeviceContourState, dev::AbstractDevice=CPU()) =
    _state_weighted_diagnostic(state, UInt8(3), dev)

circulation(prob::ContourProblem{K, D, T, GPU}) where {K, D, T} =
    _state_circulation(prob.device_state, prob.dev)

vortex_area(prob::ContourProblem{K, D, T, GPU}) where {K, D, T} =
    to_cpu(_state_vortex_area(prob.device_state, prob.dev))

enstrophy(prob::ContourProblem{K, D, T, GPU}) where {K, D, T} =
    _state_enstrophy(prob.device_state, prob.dev)

angular_momentum(prob::ContourProblem{K, D, T, GPU}) where {K, D, T} =
    _state_angular_momentum(prob.device_state, prob.dev)

function circulation(prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        s += _state_circulation(prob.device_state[i], prob.dev)
    end
    return s
end

function vortex_area(prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    ntuple(i -> to_cpu(_state_vortex_area(prob.device_state[i], prob.dev)), Val(N))
end

function enstrophy(prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        s += _state_enstrophy(prob.device_state[i], prob.dev)
    end
    return s
end

function angular_momentum(prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        s += _state_angular_momentum(prob.device_state[i], prob.dev)
    end
    return s
end

# Fallback for unsupported kernel/domain combinations
"""
    energy(prob)

Energy diagnostic for a `ContourProblem` or
`MultiLayerContourProblem`.

Available implementations depend on the kernel/domain combination:

- Euler, QG, and SQG on [`UnboundedDomain`](@ref)
- Euler, QG, and SQG on [`PeriodicDomain`](@ref)
- multi-layer QG on unbounded and periodic domains

Unsupported combinations throw an `ArgumentError`.
"""
function energy(prob::ContourProblem)
    throw(ArgumentError(
        "energy is not implemented for $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Supported: EulerKernel/QGKernel/SQGKernel on UnboundedDomain or PeriodicDomain."))
end

function energy(prob::MultiLayerContourProblem{N, K, D, T, GPU}) where {N, K, D, T}
    throw(ArgumentError(
        "GPU multi-layer energy is implemented for UnboundedDomain and PeriodicDomain; " *
        "got $(D). Use dev=CPU() for this diagnostic."))
end

function energy(prob::MultiLayerContourProblem{N, K, UnboundedDomain, T, GPU}) where {N, K, T}
    return _ka_multilayer_energy_from_states(prob.device_state, prob.kernel,
                                             prob.domain, prob.dev)
end

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, GPU}) where {N, K, T}
    return _ka_multilayer_energy_from_states(prob.device_state, prob.kernel,
                                             prob.domain, prob.dev)
end

function circulation(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    # Multi-layer scalar diagnostics are summed over all layers. Layer-resolved
    # values can be obtained by applying the single-contour diagnostics to
    # `prob.layers[i]` directly.
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
