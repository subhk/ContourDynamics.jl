# Contour dynamics velocity implementations for unbounded domains.
#
# Sign convention: positive PV induces counterclockwise circulation.
# For a vortex patch with uniform vorticity q bounded by contour C,
# the velocity is obtained by converting the area integral of the
# Green's function to a contour integral via Green's theorem:
#
#   u(x) = -(q/(4π)) ∮_C log|x-x'|² dy'
#   v(x) =  (q/(4π)) ∮_C log|x-x'|² dx'
#
# Each segment contribution is integrated analytically.

"""
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from node `a`
to node `b` with unit PV jump, using the 2D Euler Green's function in an
unbounded domain.

Computes the contour dynamics integral analytically:
  v_seg = (1/(4π)) * (-(by-ay), (bx-ax)) * ∫₀¹ log|x - a - t(b-a)|² dt
"""
function segment_velocity(::EulerKernel, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ds = b - a
    ds_len_sq = ds[1]^2 + ds[2]^2
    ds_len = sqrt(ds_len_sq)

    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    t_hat = ds / ds_len
    n_hat = SVector{2,T}(-t_hat[2], t_hat[1])

    r0 = x - a  # vector from a to x
    # Project onto segment coordinates
    u_a = r0[1] * t_hat[1] + r0[2] * t_hat[2]   # tangential component
    h   = r0[1] * n_hat[1] + r0[2] * n_hat[2]    # normal component
    u_b = u_a - ds_len

    h_sq = h * h
    eps2 = eps(T)^2

    # Antiderivative F(u) = u*log(u^2 + h^2) - 2u + 2|h|*atan(u/|h|)
    # Handle h ≈ 0 case (point on segment line)
    function F(u)
        r2 = u*u + h_sq
        if r2 < eps2
            return zero(T)
        end
        val = u * log(r2) - 2*u
        if abs(h) > eps(T)
            val += 2 * h * atan(u, h)
        else
            # h = 0: atan(u/h) = ±π/2 for u ≠ 0
            if u > zero(T)
                val += h * T(π)
            elseif u < zero(T)
                val -= h * T(π)
            end
        end
        return val
    end

    I = (F(u_a) - F(u_b)) / ds_len

    # v_seg = (1/(4π)) * (dx', dy') * I = (1/(4π)) * ds * I
    # = (1/(4π)) * t_hat * |ds| * I = (1/(4π)) * t_hat * (F(u_a) - F(u_b))
    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * t_hat * (F(u_a) - F(u_b))
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

Velocity at point `x` due to a vortex patch contour segment from `a` to `b`
using the QG Green's function G(r) = -1/(2π) K₀(r/Ld).

The contour dynamics velocity is:
  v_seg = (1/(2π)) ∫₀¹ K₀(|x-P(t)|/Ld) ds dt

Uses singular subtraction: the log singularity in K₀ is handled analytically
(matching the Euler kernel), and the smooth remainder [K₀(r/Ld) + log(r/Ld)]
is integrated with 5-point Gauss-Legendre quadrature.
"""
function segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    Ld = kernel.Ld
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    # Analytic Euler contribution (handles the log singularity)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b)

    # 5-point Gauss-Legendre on [-1, 1] for the smooth correction
    # Correction integrand: K₀(r/Ld) + log(r/Ld) which is smooth at r=0
    # (since K₀(s) ~ -log(s/2) - γ as s→0)
    # The full QG integral is:
    #   v_seg = (1/(2π)) ds ∫₀¹ K₀(r/Ld) dt
    #         = (1/(2π)) ds ∫₀¹ [-log(r) + (K₀(r/Ld) + log(r))] dt
    # The -log(r) part gives the Euler contribution (with appropriate factors).
    # The correction is: (1/(2π)) ds ∫₀¹ [K₀(r/Ld) + log(r/Ld)] dt
    # Note: the log(Ld) term integrates to a constant times ds, which sums to
    # zero around a closed contour.

    g5_n2 = sqrt(T(3) / T(7) - T(2) / T(7) * sqrt(T(6) / T(5)))
    g5_n3 = sqrt(T(3) / T(7) + T(2) / T(7) * sqrt(T(6) / T(5)))
    g5_w1 = T(128) / T(225)
    g5_w2 = (T(322) + T(13) * sqrt(T(70))) / T(900)
    g5_w3 = (T(322) - T(13) * sqrt(T(70))) / T(900)

    g_nodes = SVector{5,T}(-g5_n3, -g5_n2, zero(T), g5_n2, g5_n3)
    g_weights = SVector{5,T}(g5_w3, g5_w2, g5_w1, g5_w2, g5_w3)

    mid = (a + b) / 2
    half_ds = ds / 2

    corr_integral = zero(T)
    inv2pi = one(T) / (2 * T(π))

    for q in 1:5
        s = mid + g_nodes[q] * half_ds
        r_vec = s - x
        r2 = r_vec[1]^2 + r_vec[2]^2

        if r2 < eps(T)^2
            # K₀(s) + log(s) → log(2) - γ as s→0, finite
            corr_integral += g_weights[q] * (log(T(2)) - T(Base.MathConstants.eulergamma))
            continue
        end

        r = sqrt(r2)
        rr = r / Ld
        # K₀(rr) + log(rr) is smooth and bounded near rr=0
        corr_integral += g_weights[q] * (besselk(0, rr) + log(rr))
    end

    # Scale: the Gauss quadrature approximates ∫₋₁¹ f(t) dt, and our
    # parameterization maps [-1,1] to [0,1] via t → (1+t)/2, giving factor 1/2.
    # But the segment parameterization is already handled by the mid/half_ds.
    # So: ∫₀¹ correction dt ≈ (1/2) * sum(w_i * f(t_i))
    corr_integral *= T(0.5)  # [-1,1] to [0,1] Jacobian

    # v_corr = (1/(2π)) * ds * corr_integral
    # But we need the negative sign to match the Euler convention
    # v_seg_QG = v_Euler + v_corr where v_Euler already has the correct sign
    # From the derivation: v = (q/(2π)) oint K₀ (dx',dy') for CCW contour with positive PV
    # The Euler part is v_Euler = -(q/(4π)) t_hat * (F(u_a) - F(u_b))
    # which equals (q/(2π)) oint (-log(r)) (dx',dy')
    # So the correction adds: (q/(2π)) oint [K₀(r/Ld) + log(r)] (dx',dy')
    # = (q/(2π)) * ds * corr_integral
    # But we multiply by pv in velocity!, so segment_velocity should return per-unit-pv.
    v_corr = inv2pi * ds * corr_integral

    return v_euler + v_corr
end

"""
    velocity!(vel, prob::MultiLayerContourProblem)

Compute velocity at all nodes across all layers using modal decomposition.
"""
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N}) where {N, T}
    kernel = prob.kernel
    domain = prob.domain

    for i in 1:N
        fill!(vel[i], zero(SVector{2,T}))
    end

    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    for mode in 1:N
        lam = evals[mode]

        if abs(lam) < eps(T) * 100
            mode_kernel = EulerKernel()
        else
            Ld_mode = one(T) / sqrt(abs(lam))
            mode_kernel = QGKernel(Ld_mode)
        end

        for target_layer in 1:N
            target_contours = prob.layers[target_layer]
            projection_weight = P[target_layer, mode]
            abs(projection_weight) < eps(T) && continue

            target_idx = 0
            for tc in target_contours
                for ti in 1:nnodes(tc)
                    target_idx += 1
                    x = tc.nodes[ti]
                    v_mode = zero(SVector{2,T})

                    for source_layer in 1:N
                        source_weight = P_inv[mode, source_layer]
                        abs(source_weight) < eps(T) && continue

                        for sc in prob.layers[source_layer]
                            nsc = nnodes(sc)
                            nsc < 2 && continue
                            for sj in 1:nsc
                                a = sc.nodes[sj]
                                b = sc.nodes[mod1(sj + 1, nsc)]
                                v_mode = v_mode + source_weight * sc.pv *
                                    segment_velocity(mode_kernel, domain, x, a, b)
                            end
                        end
                    end

                    vel[target_layer][target_idx] = vel[target_layer][target_idx] +
                        projection_weight * v_mode
                end
            end
        end
    end

    return vel
end
