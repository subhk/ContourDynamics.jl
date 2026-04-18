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

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * t_hat * (_euler_antideriv(u_a, h, h_sq) - _euler_antideriv(u_b, h, h_sq))
end

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
    log_z2_gamma = log(z / 2) + T(Base.MathConstants.eulergamma)
    return -log_z2_gamma * I0_minus_1 + S
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
        # K₀(r/Ld) + log(r) = [K₀(z) + log(z/2) + γ] + log(2Ld) - γ
        # For small z = r/Ld, use series to avoid catastrophic cancellation
        # between besselk(0, z) ≈ -log(z/2) - γ and log(r).
        if rr < T(0.5)
            val = _besselk0_correction(rr) + log(2 * Ld) - T(Base.MathConstants.eulergamma)
        else
            val = besselk(0, rr) + log(r)
        end
        corr_integral += g_weights[q] * val
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
Green's function `G(r) = -1/(2π√(r²+δ²))`.

The contour integral is:
  v_seg = -(1/(2π)) t̂ [F(u_a) - F(u_b)]

where `F(u) = log(u + √(u² + h_eff²))` and `h_eff² = h² + δ²`.

The `-(1/(2π))` prefactor (vs `-(1/(4π))` for Euler) reflects the different
Green's function normalisation: SQG uses `1/(2πr)` while Euler uses
`log(r²)/(4π)`.
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

    h_eff_sq = h * h + kernel.delta^2
    h_eff = sqrt(h_eff_sq)

    # Antiderivative F(u) = arcsinh(u / h_eff)
    # Numerically stable form of log(u + √(u² + h_eff²)) — avoids
    # catastrophic cancellation when u is large negative.
    F_a = asinh(u_a / h_eff)
    F_b = asinh(u_b / h_eff)

    inv2pi = one(T) / (2 * T(π))
    return -inv2pi * t_hat * (F_a - F_b)
end
