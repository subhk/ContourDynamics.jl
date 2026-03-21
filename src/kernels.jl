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

# 5-point Gauss-Legendre nodes and weights on [-1,1], computed once per type.
# Using @generated to const-fold the sqrt/division at compile time.
@generated function _gl5_nodes_weights(::Type{T}) where {T<:AbstractFloat}
    n2 = sqrt(3/7 - 2/7 * sqrt(6/5))
    n3 = sqrt(3/7 + 2/7 * sqrt(6/5))
    w1 = 128/225
    w2 = (322 + 13*sqrt(70)) / 900
    w3 = (322 - 13*sqrt(70)) / 900
    nodes_tuple = (-n3, -n2, 0.0, n2, n3)
    weights_tuple = (w3, w2, w1, w2, w3)
    return quote
        (SVector{5,$T}($nodes_tuple...), SVector{5,$T}($weights_tuple...))
    end
end

"""
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from node `a`
to node `b` with unit PV jump, using the 2D Euler Green's function in an
unbounded domain.

Computes the contour dynamics integral analytically:
  v_seg = -(1/(4π)) * (bx-ax, by-ay) * ∫₀¹ log|x - a - t(b-a)|² dt

The velocity direction is along `ds = b - a`, not rotated.
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
    N = total_nodes(prob)
    @assert length(vel) == N "vel length ($(length(vel))) must equal total nodes ($N)"

    # Build cumulative offset table for O(log C) node lookup via binary search.
    n_contours = length(contours)
    cum_offsets = Vector{Int}(undef, n_contours + 1)
    cum_offsets[1] = 0
    for ci in 1:n_contours
        cum_offsets[ci + 1] = cum_offsets[ci] + nnodes(contours[ci])
    end

    # Thread over target nodes — each node accumulates its velocity independently
    @inbounds Threads.@threads for i in 1:N
        # Binary search: find contour ci such that cum_offsets[ci] < i <= cum_offsets[ci+1]
        ci = searchsortedlast(cum_offsets, i - 1, 1, n_contours, Base.Order.Forward)
        local_i = i - cum_offsets[ci]
        xi = contours[ci].nodes[local_i]

        v = zero(SVector{2,T})
        for c in contours
            local nc = nnodes(c)
            nc < 2 && continue
            pv = c.pv
            for j in 1:nc
                a = c.nodes[j]
                b = next_node(c, j)
                v = v + pv * segment_velocity(kernel, domain, xi, a, b)
            end
        end
        vel[i] = v
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
            b = next_node(c, j)
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
(matching the Euler kernel), and the smooth remainder [K₀(r/Ld) + log(r)]
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
    # Correction integrand: K₀(r/Ld) + log(r) which is smooth at r=0
    # (since K₀(r/Ld) ~ -log(r) + log(2Ld) - γ as r→0)
    # The full QG integral is:
    #   v_seg = (1/(2π)) ds ∫₀¹ K₀(r/Ld) dt
    #         = (1/(2π)) ds ∫₀¹ [-log(r) + (K₀(r/Ld) + log(r))] dt
    # The -log(r) part gives the Euler contribution (with appropriate factors).
    # The correction is: (1/(2π)) ds ∫₀¹ [K₀(r/Ld) + log(r)] dt
    # This uses log(r) (not log(r/Ld)) to match the Euler singularity exactly.
    # Using log(r/Ld) would introduce a per-segment error of -log(Ld)*ds that
    # cancels for closed contours (Σ ds = 0) but not for spanning contours.

    g_nodes, g_weights = _gl5_nodes_weights(T)

    mid = (a + b) / 2
    half_ds = ds / 2

    corr_integral = zero(T)
    inv2pi = one(T) / (2 * T(π))

    for q in 1:5
        s = mid + g_nodes[q] * half_ds
        r_vec = s - x
        r2 = r_vec[1]^2 + r_vec[2]^2

        if r2 < eps(T)^2
            # K₀(r/Ld) + log(r) → log(2Ld) - γ as r→0, finite
            corr_integral += g_weights[q] * (log(2 * Ld) - T(Base.MathConstants.eulergamma))
            continue
        end

        r = sqrt(r2)
        rr = r / Ld
        # K₀(r/Ld) + log(r) is smooth and bounded near r=0
        corr_integral += g_weights[q] * (besselk(0, rr) + log(r))
    end

    # Scale: the Gauss quadrature approximates ∫₋₁¹ f(t) dt, and our
    # parameterization maps [-1,1] to [0,1] via t → (1+t)/2, giving factor 1/2.
    # But the segment parameterization is already handled by the mid/half_ds.
    # So: ∫₀¹ correction dt ≈ (1/2) * sum(w_i * f(t_i))
    corr_integral *= T(0.5)  # [-1,1] to [0,1] Jacobian

    # v_seg_QG = v_Euler + v_corr where v_Euler already has the correct sign
    # The Euler part handles the -log(r) singularity analytically.
    # The correction adds: (1/(2π)) * ds * ∫₀¹ [K₀(r/Ld) + log(r)] dt
    # (per unit PV; pv is multiplied in velocity!).
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

    # Pre-allocate scratch buffers sized to the largest layer
    max_nodes = maximum(sum(nnodes(c) for c in prob.layers[i]; init=0) for i in 1:N)
    target_nodes = Vector{SVector{2,T}}(undef, max_nodes)
    mode_vel = Vector{SVector{2,T}}(undef, max_nodes)

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

            # Flatten target nodes into pre-allocated buffer
            n_target = sum(nnodes(tc) for tc in target_contours; init=0)
            n_target == 0 && continue

            idx = 0
            for tc in target_contours
                for ti in 1:nnodes(tc)
                    idx += 1
                    target_nodes[idx] = tc.nodes[ti]
                end
            end

            @inbounds Threads.@threads for ti in 1:n_target
                x = target_nodes[ti]
                v_mode = zero(SVector{2,T})
                for source_layer in 1:N
                    source_weight = P_inv[mode, source_layer]
                    abs(source_weight) < eps(T) && continue
                    for sc in prob.layers[source_layer]
                        nsc = nnodes(sc)
                        nsc < 2 && continue
                        for sj in 1:nsc
                            a = sc.nodes[sj]
                            b = next_node(sc, sj)
                            v_mode = v_mode + source_weight * sc.pv *
                                segment_velocity(mode_kernel, domain, x, a, b)
                        end
                    end
                end
                mode_vel[ti] = v_mode
            end

            for ti in 1:n_target
                vel[target_layer][ti] = vel[target_layer][ti] + projection_weight * mode_vel[ti]
            end
        end
    end

    return vel
end
