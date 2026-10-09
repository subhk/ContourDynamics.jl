# Minimum segment count to enable threading in energy pair loops.
# Below this threshold, thread spawn overhead dominates the computation.
const _THREADING_THRESHOLD = 64

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

function _max_valid_energy_nnodes(contours)
    return maximum((nnodes(c) for c in contours if _valid_energy_contour(c)); init=0)
end

"""
    @_valid_contour_pairs ci cj mult partial contours scratch begin ... end

Loop over the unordered pairs of valid closed contours, each contour also
paired with itself, reusing the caller-owned `scratch` vector as the per-pair
workspace `partial`, resized to the largest valid contour. Pair integrands are
symmetric, so weighting each pair by `mult` (1 for a contour with itself, 2
otherwise) recovers the ordered double sum at half the cost.
"""
macro _valid_contour_pairs(ci, cj, mult, partial, contours, scratch, body)
    contours_var = gensym(:contours)
    a = gensym(:a)
    b = gensym(:b)
    quote
        local $contours_var = $(esc(contours))
        # Reuse the caller's scratch buffer instead of allocating a fresh
        # workspace on every energy() call.
        local $(esc(partial)) = $(esc(scratch))
        resize!($(esc(partial)), _max_valid_energy_nnodes($contours_var))
        for $a in eachindex($contours_var)
            local $(esc(ci)) = $contours_var[$a]
            _valid_energy_contour($(esc(ci))) || continue
            for $b in $a:lastindex($contours_var)
                local $(esc(cj)) = $contours_var[$b]
                _valid_energy_contour($(esc(cj))) || continue
                local $(esc(mult)) = $a == $b ? 1 : 2
                $(esc(body))
            end
        end
    end
end

"""
    _normalize_energy(raw)

Turn a raw double sum `Σ qᵢqⱼ ∮∮ Φ(r) ds·ds'` into an energy: `E = -(1/4π)·raw/2`.
The ½ is because the double sum counts both `(i,j)` and `(j,i)` for a symmetric
integrand. Shared by every kernel, domain, and backend — CPU and KA alike.
"""
@inline _normalize_energy(raw::T) where {T} = -(one(T) / (T(4) * T(π))) * raw / T(2)

# Contour potential for the unbounded Euler Hamiltonian.  If
# phi = r²(log(r) - 1)/4, then Δphi = log(r).  The shared energy
# normalization expects -2phi, which is the expression below.  Its r→0 limit
# is zero, so unlike the Green function itself it needs no singular self rule.
@inline function _euler_energy_potential_scalar(r2::T) where {T}
    r2 <= eps(T)^2 && return zero(T)
    return r2 * (T(2) - log(r2)) / T(4)
end

# Contour potential for the QG Hamiltonian. Distributionally,
# Δ[K₀(r/Ld) + log(r)] = K₀(r/Ld)/Ld²: the logarithm cancels the
# δ singularity in K₀, leaving a smooth function at the origin. The
# factor two matches the shared -raw/(8π) energy normalization.
# The additive constant 2Ld²(log(2Ld) - γ) is dropped: it cancels in the
# closed-contour double integral, but for large Ld its rounding swamps the
# O(r²) variation that carries the energy (in Float32 already at Ld ~ 10³).
@inline function _qg_energy_potential_scalar(r2::T, Ld::T) where {T}
    r2 <= eps(T)^2 && return zero(T)
    rr = sqrt(r2) / Ld
    # K₀(rr) + log(rr/2) + γ, without cancellation for small rr.
    smooth = rr <= 2 ? _besselk0_correction(rr) :
             _besselk0_scalar(rr) + log(rr / 2) + T(Base.MathConstants.eulergamma)
    return T(2) * Ld * Ld * smooth
end

"""
    _gl3_pair_quad(midi, half_dsi, midj, half_dsj, g_nodes, g_weights, Φ)

3×3 Gauss-Legendre tensor quadrature of `Φ` over the segment pair
`midi ± half_dsi` × `midj ± half_dsj`. `Φ` receives the separation vector
(not `r²`) because the periodic Ewald Green's functions depend on direction
through their Fourier phases.

`Φ` carries its own type parameter so each call site specializes and inlines
it — the energy path is allocation-tested, so a non-specialized (boxed) `Φ`
would show up as a regression in `test_allocations.jl`.
"""
@inline function _gl3_pair_quad(midi::SVector{2,T}, half_dsi::SVector{2,T},
                                midj::SVector{2,T}, half_dsj::SVector{2,T},
                                g_nodes, g_weights, Φ::F) where {T,F}
    quad = zero(T)
    for qi in 1:3
        pi_pt = midi + g_nodes[qi] * half_dsi
        for qj in 1:3
            pj_pt = midj + g_nodes[qj] * half_dsj
            r_vec = SVector{2,T}(pi_pt[1] - pj_pt[1], pi_pt[2] - pj_pt[2])
            quad += g_weights[qi] * g_weights[qj] * Φ(r_vec)
        end
    end
    return quad
end

# Midpoint, half vector, and vector of segment `i` of contour `c`.
@inline function _contour_energy_segment(c::PVContour, i::Int)
    a = c.nodes[i]
    b = next_node(c, i)
    ds = b - a
    return (mid=(a + b) / 2, half_ds=ds / 2, ds=ds)
end

# ∫∫ Φ(r) ds·ds' over one pair of straight segments.
@inline function _segment_pair_energy(si, sj, g_nodes, g_weights, Φ::F) where {F}
    quad = _gl3_pair_quad(si.mid, si.half_ds, sj.mid, sj.half_ds, g_nodes, g_weights, Φ)
    # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
    return quad / 4 * (si.ds[1] * sj.ds[1] + si.ds[2] * sj.ds[2])
end

"""
    _energy_contour_pair(ci, cj, Φ; _partial)

Double contour integral `∮∮ Φ(r) ds·ds'` over the segment pairs of `ci` and
`cj`, threaded over the outer segments. Every single-layer energy kernel is
this same O(N²) loop with a different integrand, so only `Φ` varies.
"""
function _energy_contour_pair(ci::PVContour{T}, cj::PVContour{T}, Φ::F;
                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T,F}
    ci === cj && return _energy_contour_self(ci, Φ, _partial)
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Thread over outer segments, each thread accumulates a partial sum.
    return @_energy_segment_loop partial _partial nci for i in 1:nci
        si = _contour_energy_segment(ci, i)
        local_s = zero(T)
        for j in 1:ncj
            local_s += _segment_pair_energy(
                si, _contour_energy_segment(cj, j), g_nodes, g_weights, Φ)
        end
        partial[i] = local_s
    end
end

# A contour with itself. The segment-pair integrand is symmetric, so row i
# takes its diagonal term, twice the terms at cyclic offsets 1…⌈n/2⌉-1, and,
# for even n, the opposite segment once: the ordered double sum at half the
# cost, with every row equally long so the threaded rows stay balanced.
function _energy_contour_self(c::PVContour{T}, Φ::F, _partial::Vector{T}) where {T,F}
    n = nnodes(c)
    doubled = cld(n, 2) - 1
    g_nodes, g_weights = _gl3_nodes_weights(T)
    return @_energy_segment_loop partial _partial n for i in 1:n
        si = _contour_energy_segment(c, i)
        local_s = _segment_pair_energy(si, si, g_nodes, g_weights, Φ)
        for d in 1:doubled
            j = i + d > n ? i + d - n : i + d
            local_s += 2 * _segment_pair_energy(
                si, _contour_energy_segment(c, j), g_nodes, g_weights, Φ)
        end
        if iseven(n)
            j = i + n ÷ 2 > n ? i - n ÷ 2 : i + n ÷ 2
            local_s += _segment_pair_energy(
                si, _contour_energy_segment(c, j), g_nodes, g_weights, Φ)
        end
        partial[i] = local_s
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
    return _raw_polygon_area(nodes)
end

"""
    centroid(c::PVContour)

Centroid of the region enclosed by contour `c`, via Green's theorem.
"""
function centroid(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(SVector{2, T})
    is_spanning(c) && return _raw_polygon_mean(nodes)
    return _raw_polygon_centroid(nodes)
end

"""
    ellipse_moments(c::PVContour)

Second moments → (aspect_ratio, orientation_angle).
"""
function ellipse_moments(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    A = vortex_area(c)

    # Guard against degenerate contours using the local-coordinate shoelace
    # rounding scale. An absolute eps floor would misclassify well-resolved but
    # physically small vortices.
    if n < 3 || abs(A) <= _raw_polygon_area_tolerance(nodes)
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
    # is negligible relative to the vortex area — both carry units of length² —
    # the contour is essentially a point; return unit aspect ratio. Comparing
    # against bare eps(T) would flag every vortex smaller than ~1e-8 in linear
    # scale as a circle regardless of its true shape.
    if trace <= eps(T) * abs(A)
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
    for c in _host_contours(prob)
        s += c.pv * vortex_area(c)
    end
    return s
end

"""Return signed areas for every contour in a single-layer problem."""
vortex_area(prob::ContourProblem) = vortex_area.(_host_contours(prob))

"""Return one vector of signed contour areas per layer."""
function vortex_area(prob::MultiLayerContourProblem{N}) where {N}
    ntuple(i -> vortex_area.(_host_contours(prob)[i]), Val(N))
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
    for c in _host_contours(prob)
        s += c.pv^2 * vortex_area(c)
    end
    return s / 2
end

"""
    angular_momentum(prob)

Angular momentum `∑ qᵢ ∫ r² dA` of a `ContourProblem`. For a
`MultiLayerContourProblem` each layer's moment is weighted by the kernel's
`layer_thicknesses` `Hₗ`, `∑ₗ Hₗ ∑ᵢ qᵢ ∫ r² dA`: layers exchange angular
momentum through the coupling, and only this depth-weighted sum is conserved.
"""
function angular_momentum(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in _host_contours(prob)
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
    origin = nodes[1]
    area2 = zero(T)
    first_moment_x6 = zero(T)
    first_moment_y6 = zero(T)
    local_moment12 = zero(T)
    @inbounds for i in 1:n
        point = nodes[i] - origin
        nxt = next_node(c, i) - origin
        xi, yi = point[1], point[2]
        xj, yj = nxt[1], nxt[2]
        cross = xi * yj - xj * yi
        area2 += cross
        first_moment_x6 += (xi + xj) * cross
        first_moment_y6 += (yi + yj) * cross
        local_moment12 += (xi^2 + xi * xj + xj^2) * cross
        local_moment12 += (yi^2 + yi * yj + yj^2) * cross
    end
    ox, oy = origin
    return local_moment12 / T(12) +
           (ox * first_moment_x6 + oy * first_moment_y6) / T(3) +
           (ox * ox + oy * oy) * area2 / T(2)
end

# Per-node partials of the polygon area and second moment, relative to the
# contour's first node; a segmented scan over `contour_of_node` reduces them.
@kernel function _state_area_moment_partials_kernel!(cross_part, fmx_part, fmy_part,
                                                     lm_part, x, y, wrapx, wrapy,
                                                     offsets, lengths,
                                                     contour_of_node, local_index,
                                                     total_nodes)
    g = @index(Global)
    if g <= total_nodes
        @inbounds begin
            ci = contour_of_node[g]
            li = local_index[g]
            n = lengths[ci]
            off = offsets[ci]
            ox = x[off]
            oy = y[off]
            xi = x[g] - ox
            yi = y[g] - oy
            xj = li < n ? x[g + 1] - ox : x[off] + wrapx[ci] - ox
            yj = li < n ? y[g + 1] - oy : y[off] + wrapy[ci] - oy
            cross = xi * yj - xj * yi
            cross_part[g] = cross
            fmx_part[g] = (xi + xj) * cross
            fmy_part[g] = (yi + yj) * cross
            lm_part[g] = (xi * xi + xi * xj + xj * xj) * cross +
                         (yi * yi + yi * yj + yj * yj) * cross
        end
    end
end

@kernel function _state_area_moment_kernel!(area, moment, cross_scan, fmx_scan,
                                            fmy_scan, lm_scan, x, y, wrapx, wrapy,
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
            last = off + n - 1
            @inbounds begin
                ox = x[off]
                oy = y[off]
                area2 = cross_scan[last]
                first_moment_x6 = fmx_scan[last]
                first_moment_y6 = fmy_scan[last]
                local_moment12 = lm_scan[last]
            end
            area[ci] = area2 / T(2)
            moment[ci] = local_moment12 / T(12) +
                         (ox * first_moment_x6 + oy * first_moment_y6) / T(3) +
                         (ox * ox + oy * oy) * area2 / T(2)
        end
    end
end

function _state_area_moment(state::DeviceContourState{T},
                            dev::AbstractDevice=CPU()) where {T}
    ncontours = length(state.lengths)
    area = device_zeros(dev, T, ncontours)
    moment = device_zeros(dev, T, ncontours)
    if ncontours > 0
        total_nodes = length(state.x)
        parts = ntuple(_ -> device_zeros(dev, T, total_nodes), 4)
        if total_nodes > 0
            @_ka_launch dev total_nodes _state_area_moment_partials_kernel!(
                parts..., state.x, state.y, state.wrapx, state.wrapy,
                state.offsets, state.lengths, state.contour_of_node,
                state.local_index, total_nodes)
        end
        scans = _device_segmented_scan(parts, state.contour_of_node, total_nodes,
                                       (+, +, +, +), dev)
        @_ka_launch dev ncontours _state_area_moment_kernel!(
            area, moment, scans..., state.x, state.y, state.wrapx, state.wrapy,
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

_state_circulation(state::DeviceContourState, dev::AbstractDevice=CPU()) = _state_weighted_diagnostic(state, UInt8(1), dev)

_state_enstrophy(state::DeviceContourState, dev::AbstractDevice=CPU()) = _state_weighted_diagnostic(state, UInt8(2), dev)

_state_angular_momentum(state::DeviceContourState, dev::AbstractDevice=CPU()) = _state_weighted_diagnostic(state, UInt8(3), dev)

circulation(prob::ContourProblem{K,D,T,GPU,S}) where {
    K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, S} = _state_circulation(_device_state(prob), prob.dev)

vortex_area(prob::ContourProblem{K,D,T,GPU,S}) where {
    K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, S} = to_cpu(_state_vortex_area(_device_state(prob), prob.dev))

enstrophy(prob::ContourProblem{K,D,T,GPU,S}) where {
    K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, S} = _state_enstrophy(_device_state(prob), prob.dev)

angular_momentum(prob::ContourProblem{K,D,T,GPU,S}) where {
    K<:AbstractKernel, D<:AbstractDomain, T<:AbstractFloat, S} = _state_angular_momentum(_device_state(prob), prob.dev)

function circulation(prob::MultiLayerContourProblem{N,K,D,T,GPU,S}) where {
    N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat,S}
    s = zero(T)
    for i in 1:N
        s += _state_circulation(_device_state(prob)[i], prob.dev)
    end
    return s
end

function vortex_area(prob::MultiLayerContourProblem{N,K,D,T,GPU,S}) where {
    N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat,S}
    ntuple(i -> to_cpu(_state_vortex_area(_device_state(prob)[i], prob.dev)), Val(N))
end

function enstrophy(prob::MultiLayerContourProblem{N,K,D,T,GPU,S}) where {
    N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat, S}
    s = zero(T)
    for i in 1:N
        s += _state_enstrophy(_device_state(prob)[i], prob.dev)
    end
    return s
end

function angular_momentum(prob::MultiLayerContourProblem{N,K,D,T,GPU,S}) where {
    N, K<:MultiLayerQGKernel{N}, D<:AbstractDomain, T<:AbstractFloat, S}
    H = prob.kernel.layer_thicknesses
    s = zero(T)
    for i in 1:N
        s += T(H[i]) * _state_angular_momentum(_device_state(prob)[i], prob.dev)
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
    return _ka_multilayer_energy_from_states(_device_state(prob), prob.kernel,
                                             prob.domain, prob.dev; workspace=execution_workspace(prob))
end

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, GPU}) where {N, K, T}
    return _ka_multilayer_energy_from_states(_device_state(prob), prob.kernel,
                                             prob.domain, prob.dev; workspace=execution_workspace(prob))
end

function circulation(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    # Multi-layer scalar diagnostics are summed over all layers. Layer-resolved
    # values can be obtained by applying the single-contour diagnostics to
    # `_host_contours(prob)[i]` directly.
    s = zero(T)
    for i in 1:N
        for c in _host_contours(prob)[i]
            s += c.pv * vortex_area(c)
        end
    end
    return s
end

function enstrophy(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in _host_contours(prob)[i]
            s += c.pv^2 * vortex_area(c)
        end
    end
    return s / 2
end

function angular_momentum(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    # Only the depth-weighted sum is invariant; see the docstring.
    H = prob.kernel.layer_thicknesses
    s = zero(T)
    for i in 1:N
        layer = zero(T)
        for c in _host_contours(prob)[i]
            layer += c.pv * _second_moment_r2(c)
        end
        s += T(H[i]) * layer
    end
    return s
end
