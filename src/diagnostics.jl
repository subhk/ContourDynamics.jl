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
    abs(A) < eps(T) && return zero(SVector{2, T})

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

    # Guard against near-degenerate contours: use a threshold relative to
    # the trace (sum of eigenvalues) so that the aspect ratio stays finite.
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

"""
    energy(prob)

Kinetic energy of a [`ContourProblem`](@ref) or [`MultiLayerContourProblem`](@ref),
computed via contour integrals of the appropriate Green's function.
"""
function energy(prob::ContourProblem{EulerKernel, UnboundedDomain, T}) where {T}
    contours = prob.contours
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    # Pre-allocate partial-sum buffer sized to the largest contour.
    max_n = maximum((nnodes(c) for c in contours if nnodes(c) >= 3 && !is_spanning(c)), init=0)
    _partial = zeros(T, max_n)
    for ci in contours
        nci = nnodes(ci)
        nci < 3 && continue
        is_spanning(ci) && continue
        for cj in contours
            ncj = nnodes(cj)
            ncj < 3 && continue
            is_spanning(cj) && continue
            E += ci.pv * cj.pv * _energy_contour_pair_euler(ci, cj; _partial=_partial)
        end
    end
    # Factor 1/2: the double sum counts both (i,j) and (j,i) for the symmetric integrand.
    return -inv4pi * E / 2
end

function _energy_contour_pair_euler(ci::PVContour{T}, cj::PVContour{T};
                                    _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes  # detect self-interaction
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral:
    # ∫₋₁¹∫₋₁¹ log(|s-t| * |half_ds|) ds dt = 4*log(2) - 6 + 4*log|half_ds|
    self_seg_const = 4 * log(T(2)) - T(6)  # precompute constant part
    # Thread over outer segments, each thread accumulates a partial sum
    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]

            if is_self && i == j
                # Self-segment: log singularity requires analytical integration.
                # ∫₋₁¹∫₋₁¹ log(r²)/2 ds dt where r = |s-t|*|half_ds|
                #   = ∫₋₁¹∫₋₁¹ (log|s-t| + log|half_ds|) ds dt
                #   = (4*log(2) - 6) + 4*log|half_ds|
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad = self_seg_const + 4 * log(half_ds_len)
                else
                    quad = zero(T)
                end
            else
                # 3×3 Gauss-Legendre quadrature over both segments.
                # Use max(r2, eps(T)) instead of skipping near-zero r2:
                # adjacent segments share a node where log(r²) diverges,
                # but the integral is finite (integrable singularity).
                # Clamping avoids log(0) while preserving the contribution.
                quad = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r2 = max(dx^2 + dy^2, eps(T))
                        quad += g_weights[qi] * g_weights[qj] * log(r2) / 2
                    end
                end
            end
            # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end

function energy(prob::ContourProblem{SQGKernel{T}, UnboundedDomain, T}) where {T}
    contours = prob.contours
    delta = prob.kernel.delta
    E = zero(T)
    max_n = maximum((nnodes(c) for c in contours if nnodes(c) >= 3 && !is_spanning(c)), init=0)
    _partial = zeros(T, max_n)
    for ci in contours
        nnodes(ci) < 3 && continue
        is_spanning(ci) && continue
        for cj in contours
            nnodes(cj) < 3 && continue
            is_spanning(cj) && continue
            E += ci.pv * cj.pv * _energy_contour_pair_sqg(ci, cj, delta; _partial=_partial)
        end
    end
    # E_SQG = (1/(4π)) × (1/2) × Σ q_i q_j ∮∮ √(r²+δ²) ds·ds'
    # Derived from ∫∫ (1/r) dA dA' = -∮∮ r ds·ds' via ∇'²r = 1/r
    inv4pi = one(T) / (4 * T(π))
    return inv4pi * E / 2
end

function _energy_contour_pair_sqg(ci::PVContour{T}, cj::PVContour{T}, delta::T;
                                   _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    delta_sq = delta^2
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            # 3×3 Gauss-Legendre quadrature — √(r²+δ²) is smooth everywhere
            quad = zero(T)
            for qi in 1:3
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:3
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    quad += g_weights[qi] * g_weights[qj] * sqrt(dx^2 + dy^2 + delta_sq)
                end
            end
            # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end

function energy(prob::ContourProblem{QGKernel{T}, UnboundedDomain, T}) where {T}
    contours = prob.contours
    Ld = prob.kernel.Ld
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    max_n = maximum((nnodes(c) for c in contours if nnodes(c) >= 3 && !is_spanning(c)), init=0)
    _partial = zeros(T, max_n)
    for ci in contours
        nnodes(ci) < 3 && continue
        is_spanning(ci) && continue
        for cj in contours
            nnodes(cj) < 3 && continue
            is_spanning(cj) && continue
            E += ci.pv * cj.pv * _energy_contour_pair_qg(ci, cj, Ld; _partial=_partial)
        end
    end
    return -inv4pi * E / 2
end

function _energy_contour_pair_qg(ci::PVContour{T}, cj::PVContour{T}, Ld::T;
                                  _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes  # detect self-interaction
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral for the log(r²)/2 singularity
    # (same formula as Euler self-segment)
    self_seg_const = 4 * log(T(2)) - T(6)
    # Smooth limit of K₀(r/Ld) + log(r) as r→0
    k0_smooth_at_zero = log(2 * Ld) - T(Base.MathConstants.eulergamma)
    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]

            if is_self && i == j
                # Self-segment: singular subtraction.
                # Decompose K₀(r/Ld) = [-log(r)] + [K₀(r/Ld) + log(r)]
                # 1) The -log(r) part has a known analytical integral:
                #    ∫₋₁¹∫₋₁¹ log(|s-t|·|half_ds|) ds dt = self_seg_const + 4·log|half_ds|
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad_log = self_seg_const + 4 * log(half_ds_len)
                else
                    quad_log = zero(T)
                end
                # 2) The smooth remainder K₀(r/Ld) + log(r) → log(2Ld) - γ at r=0
                #    is safe for GL quadrature.
                quad_smooth = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r2 = dx^2 + dy^2
                        if r2 < eps(T)^2
                            # qi == qj: use smooth limit
                            quad_smooth += g_weights[qi] * g_weights[qj] * k0_smooth_at_zero
                        else
                            r = sqrt(r2)
                            quad_smooth += g_weights[qi] * g_weights[qj] * (besselk(0, r / Ld) + log(r))
                        end
                    end
                end
                # Combined: K₀(r/Ld) = [-log(r)] + [K₀(r/Ld) + log(r)]
                # quad_log = ∫∫ log(r) ds dt (positive).
                # The -log(r) part contributes -quad_log.
                # The smooth part contributes +quad_smooth.
                quad = -quad_log + quad_smooth
            else
                # 3×3 Gauss-Legendre quadrature over both segments
                quad = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        dx = pi_pt[1] - pj_pt[1]
                        dy = pi_pt[2] - pj_pt[2]
                        r = sqrt(dx^2 + dy^2)
                        r < eps(T) * Ld && continue
                        quad += g_weights[qi] * g_weights[qj] * besselk(0, r / Ld)
                    end
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end

"""
    _eval_ewald_greens(r_vec, cache, domain)

Evaluate the periodic Green's function at separation `r_vec` using Ewald summation.
Returns `G_per(r)` = real-space Ewald sum + Fourier-space sum.

!!! note
    The central-image real-space term (E₁(α²r²)) diverges at `r = 0`.
    This function silently skips that term when `r² < eps(T)`, so the
    returned value is *not* valid at zero separation.  Callers that need
    the self-interaction limit must handle `r = 0` separately (see
    `_energy_contour_pair_euler_periodic` for an example).
"""
function _eval_ewald_greens(r_vec::SVector{2,T}, cache::EwaldCache{T},
                            domain::PeriodicDomain{T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    inv4pi = one(T) / (4 * T(π))
    G_val = zero(T)

    # Real-space sum
    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
            rv = r_vec - shift
            r2 = rv[1]^2 + rv[2]^2
            if r2 > eps(T)
                G_val += inv4pi * _expint_e1(alpha^2 * r2)
            end
        end
    end

    # Fourier-space sum
    for (mi, kxi) in enumerate(cache.kx)
        for (ni, kyi) in enumerate(cache.ky)
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            phase = kxi * r_vec[1] + kyi * r_vec[2]
            G_val += coeff * cos(phase)
        end
    end

    return G_val
end

function _energy_contour_pair_euler_periodic(ci::PVContour{T}, cj::PVContour{T},
                                              cache::EwaldCache{T},
                                              domain::PeriodicDomain{T};
                                              _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    is_self = ci.nodes === cj.nodes
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    # Analytical self-segment integral for the log(r²)/2 singularity
    self_seg_const = 4 * log(T(2)) - T(6)

    # Precompute the limit of [-2π G_per(r) - log(r²)/2] as r→0.
    # This is the smooth periodic correction at zero separation, needed for
    # self-segment GL points where both quadrature indices coincide (r=0).
    corr_at_zero = zero(T)
    if is_self
        alpha = cache.alpha
        Lx, Ly = domain.Lx, domain.Ly
        gamma_euler = T(Base.MathConstants.eulergamma)
        # Central image: lim_{r→0} [-(1/2) E₁(α²r²) - log(r²)/2] = (γ + 2ln(α))/2
        corr_at_zero = (gamma_euler + 2 * log(alpha)) / 2
        # Non-central real-space images evaluated at r=0
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                (px == 0 && py == 0) && continue
                shift_r2 = (2 * Lx * px)^2 + (2 * Ly * py)^2
                corr_at_zero -= _expint_e1(alpha^2 * shift_r2) / 2
            end
        end
        # Fourier-space sum at r=0 (cos(k·0) = 1)
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                corr_at_zero -= 2 * T(π) * coeff
            end
        end
    end

    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]

            if is_self && i == j
                # Self-segment: singular subtraction.
                # 1) Analytical integral of log(r²)/2 (same as unbounded)
                half_ds_len = sqrt(half_dsi[1]^2 + half_dsi[2]^2)
                if half_ds_len > eps(T)
                    quad_analytical = self_seg_const + 4 * log(half_ds_len)
                else
                    quad_analytical = zero(T)
                end
                # 2) Smooth periodic correction [-2π G_per(r) - log(r²)/2] via GL
                quad_corr = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        r_vec = SVector{2,T}(pi_pt[1] - pj_pt[1], pi_pt[2] - pj_pt[2])
                        r2 = r_vec[1]^2 + r_vec[2]^2
                        if r2 > eps(T)
                            G_per = _eval_ewald_greens(r_vec, cache, domain)
                            quad_corr += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_per - log(r2) / 2)
                        else
                            # qi == qj: use precomputed finite limit
                            quad_corr += g_weights[qi] * g_weights[qj] * corr_at_zero
                        end
                    end
                end
                quad = quad_analytical + quad_corr
            else
                quad = zero(T)
                for qi in 1:3
                    pi_pt = midi + g_nodes[qi] * half_dsi
                    for qj in 1:3
                        pj_pt = midj + g_nodes[qj] * half_dsj
                        r_vec = SVector{2,T}(pi_pt[1] - pj_pt[1], pi_pt[2] - pj_pt[2])
                        # Replace log(r²)/2 with the periodic equivalent: -2π * G_per
                        # since log(r²)/2 = -2π * G_∞ for unbounded Euler.
                        G_per = _eval_ewald_greens(r_vec, cache, domain)
                        quad += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_per)
                    end
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end

function energy(prob::ContourProblem{EulerKernel, PeriodicDomain{T}, T}) where {T}
    contours = prob.contours
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    max_n = maximum((nnodes(c) for c in contours if nnodes(c) >= 3 && !is_spanning(c)), init=0)
    _partial = zeros(T, max_n)
    for ci in contours
        nnodes(ci) < 3 && continue
        is_spanning(ci) && continue
        for cj in contours
            nnodes(cj) < 3 && continue
            is_spanning(cj) && continue
            E += ci.pv * cj.pv * _energy_contour_pair_euler_periodic(ci, cj, cache, prob.domain; _partial=_partial)
        end
    end
    return -inv4pi * E / 2
end

function energy(prob::ContourProblem{QGKernel{T}, PeriodicDomain{T}, T}) where {T}
    contours = prob.contours
    Ld = prob.kernel.Ld
    # Decompose: G_QG_per = G_Euler_per + G_correction
    # Use Euler periodic energy + QG correction via Fourier sum.
    euler_cache = _get_ewald_cache(prob.domain, EulerKernel())
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    kappa2 = one(T) / Ld^2
    area = 4 * prob.domain.Lx * prob.domain.Ly
    max_n = maximum((nnodes(c) for c in contours if nnodes(c) >= 3 && !is_spanning(c)), init=0)
    _partial = zeros(T, max_n)
    for ci in contours
        nnodes(ci) < 3 && continue
        is_spanning(ci) && continue
        for cj in contours
            nnodes(cj) < 3 && continue
            is_spanning(cj) && continue
            pair_E = _energy_contour_pair_euler_periodic(ci, cj, euler_cache, prob.domain; _partial=_partial)
            pair_E += _energy_contour_pair_qg_correction(ci, cj, euler_cache, kappa2, area; _partial=_partial)
            E += ci.pv * cj.pv * pair_E
        end
    end
    return -inv4pi * E / 2
end

"""QG-Euler correction for periodic energy: smooth Fourier series with -κ²/(k²(k²+κ²)) coefficients."""
function _energy_contour_pair_qg_correction(ci::PVContour{T}, cj::PVContour{T},
                                             euler_cache::EwaldCache{T},
                                             kappa2::T, area::T;
                                             _partial::Vector{T}=zeros(T, nnodes(ci))) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # 3-point Gauss-Legendre nodes/weights on [-1,1]
    g_nodes, g_weights = _gl3_nodes_weights(T)
    partial = _partial
    @inbounds for k in 1:nci; partial[k] = zero(T); end
    @_maybe_threads nci >= _THREADING_THRESHOLD for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            quad = zero(T)
            for qi in 1:3
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:3
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    G_corr = zero(T)
                    for kxi in euler_cache.kx
                        for kyi in euler_cache.ky
                            k2 = kxi^2 + kyi^2
                            k2 < eps(T) && continue
                            coeff = kappa2 / (k2 * (k2 + kappa2) * area)
                            phase = kxi * dx + kyi * dy
                            G_corr -= coeff * cos(phase)
                        end
                    end
                    quad += g_weights[qi] * g_weights[qj] * (-2 * T(π) * G_corr)
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(@view partial[1:nci])
end

# SQG on PeriodicDomain: velocity is supported but energy is not yet implemented.
function energy(prob::ContourProblem{SQGKernel{T}, PeriodicDomain{T}, T}) where {T}
    throw(ArgumentError(
        "energy is not yet implemented for SQGKernel on PeriodicDomain. " *
        "SQG periodic velocity works, but the energy diagnostic requires Ewald-split " *
        "double contour integrals of 1/r that are not yet available."))
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

function energy(prob::MultiLayerContourProblem{N, K, UnboundedDomain, T}) where {N, K, T}
    kernel = prob.kernel
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    E = zero(T)

    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        for li in 1:N
            wi = P_inv[mode, li]
            abs(wi) < eps(T) && continue
            for lj in 1:N
                wj = P_inv[mode, lj]
                abs(wj) < eps(T) && continue
                for ci in prob.layers[li]
                    nci = nnodes(ci)
                    nci < 3 && continue
                    is_spanning(ci) && continue
                    for cj in prob.layers[lj]
                        ncj = nnodes(cj)
                        ncj < 3 && continue
                        is_spanning(cj) && continue
                        if abs(lam) < eps(T) * 100
                            pair_E = _energy_contour_pair_euler(ci, cj; _partial=_partial)
                        else
                            Ld_mode = one(T) / sqrt(abs(lam))
                            pair_E = _energy_contour_pair_qg(ci, cj, Ld_mode; _partial=_partial)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end

function energy(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T}) where {N, K, T}
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    E = zero(T)

    euler_cache = _get_ewald_cache(domain, EulerKernel())
    area = 4 * domain.Lx * domain.Ly
    max_n = maximum(nnodes(c) for layer in prob.layers for c in layer if nnodes(c) >= 3 && !is_spanning(c); init=0)
    _partial = zeros(T, max_n)

    for mode in 1:N
        lam = evals[mode]
        for li in 1:N
            wi = P_inv[mode, li]
            abs(wi) < eps(T) && continue
            for lj in 1:N
                wj = P_inv[mode, lj]
                abs(wj) < eps(T) && continue
                for ci in prob.layers[li]
                    nci = nnodes(ci)
                    nci < 3 && continue
                    is_spanning(ci) && continue
                    for cj in prob.layers[lj]
                        ncj = nnodes(cj)
                        ncj < 3 && continue
                        is_spanning(cj) && continue
                        if abs(lam) < eps(T) * 100
                            pair_E = _energy_contour_pair_euler_periodic(ci, cj, euler_cache, domain; _partial=_partial)
                        else
                            kappa2 = abs(lam)
                            pair_E = _energy_contour_pair_euler_periodic(ci, cj, euler_cache, domain; _partial=_partial)
                            pair_E += _energy_contour_pair_qg_correction(ci, cj, euler_cache, kappa2, area; _partial=_partial)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end
