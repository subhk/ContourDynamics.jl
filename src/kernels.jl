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

Uses 3-point Gauss-Legendre quadrature along the segment.
"""
function segment_velocity(kernel::QGKernel{T}, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    Ld = kernel.Ld
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    # 3-point Gauss-Legendre on [-1, 1]
    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))

    mid = (a + b) / 2
    half_ds = ds / 2

    v = zero(SVector{2,T})
    inv2pi = one(T) / (2 * T(π))

    for q in 1:3
        s = mid + g_nodes[q] * half_ds
        r_vec = s - x
        r = sqrt(r_vec[1]^2 + r_vec[2]^2)

        if r < eps(T) * Ld
            continue
        end

        rr = r / Ld
        K1_val = besselk(1, rr)
        perp = SVector{2,T}(r_vec[2], -r_vec[1]) / r

        v = v + g_weights[q] * (inv2pi / Ld) * K1_val * perp
    end

    return v * (ds_len / 2)
end
