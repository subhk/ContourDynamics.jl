# Domain-specific single-layer velocity kernels for `UnboundedDomain`.

"""
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from node `a`
to node `b` with unit PV jump, using the 2D Euler Green's function in an
unbounded domain.

Computes the contour dynamics integral analytically:
  v_seg = -(1/(4π)) * (bx-ax, by-ay) * ∫₀¹ log|x - a - t(b-a)|² dt

The velocity direction is along `ds = b - a`, not rotated.
"""
# Antiderivative for the Euler segment velocity integral.
# F(u; h, h_sq) = u*log(u² + h²) - 2u + 2h*arctan(u/h)
@inline function _euler_antideriv(u::T, h::T, h_sq::T) where {T}
    r2 = u * u + h_sq
    # Threshold: eps(T)^2 ≈ 5e-32 for Float64.  Only catches points at
    # essentially zero distance; self-segment contributions with small but
    # nonzero r remain correctly evaluated (their log-integral is meaningful).
    if r2 < eps(T)^2
        return zero(T)
    end
    val = u * log(r2) - 2 * u
    # Guard against h/h division: when h² ≤ eps(T)², the atan term
    # 2h·atan(u/h) → ±π·h is at most π·eps(T), negligible at Float64 scale.
    # Use h_sq threshold consistent with the r2 guard above.
    if h_sq > eps(T)^2
        val += 2 * h * atan(u / h)
    end
    return val
end

# Evaluate the definite logarithmic integral over a segment in local
# coordinates.  Subtracting the two antiderivatives directly loses digits when
# the target is many segment lengths away.  The centered far-field expression
# rewrites the log ratio with log1p and the angle difference as one atan.
@inline function _euler_segment_log_integral(u_a::T, h::T, ds_len::T) where {T}
    half = ds_len / T(2)
    u = u_a - half
    h_abs = abs(h)

    if abs(u) > T(16) * max(half, h_abs)
        up = u + half
        um = u - half
        h_sq = h * h
        rp2 = up * up + h_sq
        rm2 = um * um + h_sq
        log_ratio = log1p(T(4) * u * half / rm2)
        angle = h_abs > eps(T) ?
            T(2) * h_abs * atan(T(2) * h_abs * half /
                                (h_sq + u * u - half * half)) : zero(T)
        return u * log_ratio + half * (log(rp2) + log(rm2)) -
               T(4) * half + angle
    end

    h_sq = h * h
    return _euler_antideriv(u_a, h, h_sq) -
           _euler_antideriv(u_a - ds_len, h, h_sq)
end

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
    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * t_hat * _euler_segment_log_integral(u_a, h, ds_len)
end

function curved_segment_velocity(::EulerKernel, ::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)

    # Integrate only the smooth difference from the chord.  The chord's
    # logarithmic singularity is handled by the exact straight-panel formula,
    # restoring high-order convergence when x is a panel endpoint.
    straight = segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)
    g_nodes, g_weights = _gl5_nodes_weights(T)
    correction = zero(SVector{2,T})
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        s = _cubic_segment_point(a, b, κa, κb, p)
        tangent = _cubic_segment_tangent(a, b, κa, κb, p)
        s_line = a + p * ds
        r_curve = x - s
        r_line = x - s_line
        log_curve = log(max(r_curve[1]^2 + r_curve[2]^2, eps(T)^2))
        log_line = log(max(r_line[1]^2 + r_line[2]^2, eps(T)^2))
        correction += (g_weights[q] / T(2)) *
                      (log_curve * tangent - log_line * ds)
    end

    inv4pi = one(T) / (4 * T(π))
    return straight - inv4pi * correction
end

function curved_segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return segment_velocity(kernel, domain, x, a, b)

    Ld = kernel.Ld
    g_nodes, g_weights = _gl5_nodes_weights(T)
    corr_integral = zero(SVector{2,T})
    direct_integral = zero(SVector{2,T})
    min_r = T(Inf)

    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        s = _cubic_segment_point(a, b, κa, κb, p)
        tangent = _cubic_segment_tangent(a, b, κa, κb, p)
        r_vec = s - x
        r2 = r_vec[1]^2 + r_vec[2]^2
        r = sqrt(r2)
        min_r = min(min_r, r)
        if r2 > eps(T)^2
            direct_integral += (g_weights[q] / T(2)) *
                               _besselk0_approx_scalar(r / Ld) * tangent
        end
        val = r2 < eps(T)^2 ?
            log(T(2) * Ld) - T(Base.MathConstants.eulergamma) :
            _qg_smooth_correction_scalar(r / Ld, r, Ld)
        corr_integral += (g_weights[q] / T(2)) * val * tangent
    end

    inv2pi = one(T) / (2 * T(π))
    min_r > T(4) * max(Ld, ds_len) && return inv2pi * direct_integral
    return curved_segment_velocity(EulerKernel(), domain, x, a, b, κa, κb) +
           inv2pi * corr_integral
end

function curved_segment_velocity(kernel::SQGKernel{T}, domain::UnboundedDomain,
                                  x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                  κa::T, κb::T) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return segment_velocity(kernel, domain, x, a, b)

    g_nodes, g_weights = _gl5_nodes_weights(T)
    integral = zero(SVector{2,T})
    δ2 = kernel.δ^2

    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        s = _cubic_segment_point(a, b, κa, κb, p)
        tangent = _cubic_segment_tangent(a, b, κa, κb, p)
        r = x - s
        rreg = sqrt(r[1]^2 + r[2]^2 + δ2)
        integral += (g_weights[q] / T(2)) * tangent / rreg
    end

    inv2pi = one(T) / (2 * T(π))
    return inv2pi * integral
end

@inline curved_segment_velocity(k::AbstractKernel, d::AbstractDomain,
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                ::T, ::T) where {T} =
    segment_velocity(k, d, x, a, b)

@inline curved_segment_velocity(k::AbstractKernel, d::AbstractDomain,
                                x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                                κa::T, κb::T, ewald) where {T} =
    curved_segment_velocity(k, d, x, a, b, κa, κb)

# Unbounded domains don't use Ewald caches; ignore the argument.
@inline segment_velocity(k::AbstractKernel, d::UnboundedDomain,
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} =
    segment_velocity(k, d, x, a, b)

# Compute K₀(z) + log(z/2) + γ without catastrophic cancellation for small z.
# Uses the identity: K₀(z) = -(log(z/2) + γ)I₀(z) + Σ_{k=1}^∞ H_k (z²/4)^k/(k!)²
# so K₀(z) + log(z/2) + γ = -(log(z/2) + γ)(I₀(z) - 1) + Σ_{k=1}^∞ H_k (z²/4)^k/(k!)²
# Both terms are O(z²), avoiding the subtraction of two O(log(1/z)) quantities.
@inline function _besselk0_correction(z::T) where {T}
    z2_4 = (z / 2)^2
    I0_minus_1 = zero(T)
    S = zero(T)
    term = one(T)
    Hk = zero(T)
    for k in 1:25
        term *= z2_4 / T(k)^2
        Hk += one(T) / T(k)
        I0_minus_1 += term
        S += term * Hk
        abs(term * Hk) < eps(T) && break
    end
    log_z2_γ = log(z / 2) + T(Base.MathConstants.eulergamma)
    return -log_z2_γ * I0_minus_1 + S
end

@inline function _i0_approx_scalar(x::T) where {T}
    # Numerical Recipes I0 approximation paired with _besselk0_approx_scalar.
    ax = abs(x)
    if ax < T(3.75)
        y = (ax / T(3.75))^2
        return one(T) +
            y * (T(3.5156229) +
            y * (T(3.0899424) +
            y * (T(1.2067492) +
            y * (T(0.2659732) +
            y * (T(0.0360768) +
                 y * T(0.0045813))))))
    end

    y = T(3.75) / ax
    poly = T(0.39894228) +
           y * (T(0.01328592) +
           y * (T(0.00225319) +
           y * (-T(0.00157565) +
           y * (T(0.00916281) +
           y * (-T(0.02057706) +
           y * (T(0.02635537) +
           y * (-T(0.01647633) +
                y * T(0.00392377))))))))
    return exp(ax) / sqrt(ax) * poly
end

@inline function _besselk0_approx_scalar(x::T) where {T}
    # Allocation-free K0 approximation used in hot CPU loops and device kernels.
    ax = abs(x)
    ax < eps(T) && return T(Inf)

    if ax <= T(2)
        y = (ax * ax) / T(4)
        return -log(ax / T(2)) * _i0_approx_scalar(ax) +
            (-T(0.57721566) +
             y * (T(0.42278420) +
             y * (T(0.23069756) +
             y * (T(0.03488590) +
             y * (T(0.00262698) +
             y * (T(0.00010750) +
                  y * T(0.00000740)))))))
    end

    y = T(2) / ax
    poly = T(1.25331414) +
           y * (-T(0.07832358) +
           y * (T(0.02189568) +
           y * (-T(0.01062446) +
           y * (T(0.00587872) +
           y * (-T(0.00251540) +
                y * T(0.00053208))))))
    return exp(-ax) / sqrt(ax) * poly
end

"""
    segment_velocity(::QGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from `a` to `b`
using the QG Green's function G(r) = K₀(r/Ld) / (2π).

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

    # Far from a short panel, evaluate K₀ directly. The singular-subtraction
    # form is ideal near the contour but otherwise subtracts two logarithmic
    # O(1) terms to recover an exponentially small velocity.
    r0 = x - a
    p_near = clamp((r0[1] * ds[1] + r0[2] * ds[2]) / (ds_len * ds_len),
                   zero(T), one(T))
    r_near = x - (a + p_near * ds)
    min_r = sqrt(r_near[1]^2 + r_near[2]^2)
    if min_r > T(4) * max(Ld, ds_len)
        g_nodes, g_weights = _gl5_nodes_weights(T)
        mid = (a + b) / T(2)
        half_ds = ds / T(2)
        direct = zero(T)
        @inbounds for q in 1:5
            s = mid + g_nodes[q] * half_ds
            r = x - s
            direct += g_weights[q] *
                      _besselk0_approx_scalar(sqrt(r[1]^2 + r[2]^2) / Ld)
        end
        return (one(T) / (T(2) * T(π))) * half_ds * direct
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
        # K₀(r/Ld) + log(r) = [K₀(z) + log(z/2) + γ] + log(2Ld) - γ; the shared
        # scalar helper picks the series branch for small z = r/Ld to avoid
        # catastrophic cancellation between besselk(0, z) ≈ -log(z/2) - γ and
        # log(r).
        corr_integral += g_weights[q] * _qg_smooth_correction_scalar(r / Ld, r, Ld)
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
    segment_velocity(::SQGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a surface buoyancy patch contour segment from
node `a` to node `b` with unit buoyancy jump, using the regularized SQG
Green's function `G(r) = 1/(2π√(r²+δ²))`.

The contour integral is:
  v_seg = (1/(2π)) t̂ [F(u_a) - F(u_b)]

where `F(u) = log(u + √(u² + h_eff²))` and `h_eff² = h² + δ²`.

The `1/(2π)` prefactor reflects the different Green's function
normalisation: SQG uses `1/(2πr)` while Euler uses `log(r²)/(4π)`.
"""
function segment_velocity(kernel::SQGKernel{T}, ::UnboundedDomain,
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

    h_eff_sq = h * h + kernel.δ^2
    h_eff = sqrt(h_eff_sq)

    # Antiderivative F(u) = arcsinh(u / h_eff)
    # Numerically stable form of log(u + √(u² + h_eff²)) — avoids
    # catastrophic cancellation when u is large negative.
    F_a = asinh(u_a / h_eff)
    F_b = asinh(u_b / h_eff)

    inv2pi = one(T) / (2 * T(π))
    return inv2pi * t_hat * (F_a - F_b)
end
