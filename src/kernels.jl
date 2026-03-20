# Green's function velocity implementations for unbounded domains.
#
# Sign convention: positive PV induces counterclockwise circulation.
# Euler: G(r) = -1/(2π) log(r), v = ∇⊥ψ = (-∂ψ/∂y, ∂ψ/∂x)
# The velocity induced by a contour segment is computed via the boundary integral
# of the Green's function along the contour.

"""
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex sheet segment from node `a` to node `b`
with unit PV jump, using the 2D Euler Green's function in an unbounded domain.

Uses the analytic integral of -1/(2π) ∇⊥log(r) along the segment.
"""
function segment_velocity(::EulerKernel, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ra = a - x
    rb = b - x
    ds = b - a

    ra_sq = ra[1]^2 + ra[2]^2
    rb_sq = rb[1]^2 + rb[2]^2

    eps2 = eps(T)^2
    if ra_sq < eps2 || rb_sq < eps2
        return zero(SVector{2,T})
    end

    cross_ab = ra[1]*rb[2] - ra[2]*rb[1]
    dot_ab = ra[1]*rb[1] + ra[2]*rb[2]

    theta = atan(cross_ab, dot_ab)
    log_ratio = T(0.5) * log(rb_sq / ra_sq)

    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    if ds_len < eps(T)
        return zero(SVector{2,T})
    end
    t_hat = ds / ds_len
    n_hat = SVector{2,T}(-t_hat[2], t_hat[1])

    inv2pi = one(T) / (2 * T(π))
    return -inv2pi * (theta * t_hat + log_ratio * n_hat)
end

"""
    velocity!(vel, prob::ContourProblem)

Compute velocity at every contour node of `prob`, storing results in `vel`.
"""
function velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    kernel = prob.kernel
    domain = prob.domain
    contours = prob.contours

    all_nodes = SVector{2,T}[]
    for c in contours
        append!(all_nodes, c.nodes)
    end

    N = length(all_nodes)
    @assert length(vel) == N "vel length ($(length(vel))) must equal total nodes ($N)"

    for i in 1:N
        vel[i] = zero(SVector{2,T})
    end

    for c in contours
        nc = nnodes(c)
        nc < 2 && continue
        pv = c.pv
        for j in 1:nc
            a = c.nodes[j]
            b = c.nodes[mod1(j + 1, nc)]
            @inbounds Threads.@threads for i in 1:N
                vel[i] = vel[i] + pv * segment_velocity(kernel, domain, all_nodes[i], a, b)
            end
        end
    end

    return vel
end

"""
    velocity(prob::ContourProblem, x::SVector{2,T})

Compute velocity at a single point `x` from all contours in `prob`.
"""
function velocity(prob::ContourProblem, x::SVector{2,T}) where {T}
    v = zero(SVector{2,T})
    for c in prob.contours
        nc = nnodes(c)
        nc < 2 && continue
        for j in 1:nc
            a = c.nodes[j]
            b = c.nodes[mod1(j + 1, nc)]
            v = v + c.pv * segment_velocity(prob.kernel, prob.domain, x, a, b)
        end
    end
    return v
end

"""
    segment_velocity(::QGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex sheet segment from `a` to `b`
using the QG Green's function G(r) = -1/(2π) K₀(r/Ld).

Uses singular subtraction: the 1/r singularity is handled analytically
(matching the Euler kernel), and the smooth remainder
[K₁(r/Ld)/Ld - 1/r] is integrated with 5-point Gauss-Legendre quadrature.
"""
function segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    Ld = kernel.Ld
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    # Analytic Euler contribution (handles the 1/r² = K₁(r/Ld)/Ld |_{Ld→∞} singularity)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b)

    # 5-point Gauss-Legendre on [-1, 1] for the smooth correction
    g5_n2 = sqrt(T(3) / T(7) - T(2) / T(7) * sqrt(T(6) / T(5)))
    g5_n3 = sqrt(T(3) / T(7) + T(2) / T(7) * sqrt(T(6) / T(5)))
    g5_w1 = T(128) / T(225)
    g5_w2 = (T(322) + T(13) * sqrt(T(70))) / T(900)
    g5_w3 = (T(322) - T(13) * sqrt(T(70))) / T(900)

    g_nodes = SVector{5,T}(-g5_n3, -g5_n2, zero(T), g5_n2, g5_n3)
    g_weights = SVector{5,T}(g5_w3, g5_w2, g5_w1, g5_w2, g5_w3)

    mid = (a + b) / 2
    half_ds = ds / 2

    v_corr = zero(SVector{2,T})
    inv2pi = one(T) / (2 * T(π))

    for q in 1:5
        s = mid + g_nodes[q] * half_ds
        r_vec = s - x
        r2 = r_vec[1]^2 + r_vec[2]^2

        if r2 < eps(T)^2
            continue
        end

        r = sqrt(r2)
        rr = r / Ld
        # Smooth correction: K₁(rr)/Ld - 1/r = (K₁(rr)/rr - 1) / r
        # For rr → 0: K₁(x)/x → 1/x² + (ln(x/2)+γ-1/2)/2 + O(x²), so K₁(x)/x - 1/x² is smooth
        K1_over_rr = besselk(1, rr) / rr   # = K₁(rr)/rr = K₁(r/Ld) * Ld/r
        # K₁(r/Ld)/Ld - 1/r = (K₁(rr)/rr - 1) / r
        correction_factor = (K1_over_rr - one(T)) / r
        perp = SVector{2,T}(r_vec[2], -r_vec[1])   # unnormalized perp (length r)

        v_corr = v_corr + g_weights[q] * inv2pi * correction_factor * perp
    end

    v_corr = v_corr * (ds_len / 2)
    return v_euler + v_corr
end
