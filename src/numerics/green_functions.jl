# Green-function approximations and stable panel integrals.

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
    return _euler_antideriv(u_a, h, h_sq) - _euler_antideriv(u_a - ds_len, h, h_sq)
end

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

# Evaluate asinh(u_a/h) - asinh(u_b/h) without subtracting nearly equal
# antiderivatives when the whole panel is far to one side of the target.  The
# half-difference identity
#
#   tanh((asinh(x) - asinh(y))/2) = (x-y)/(√(1+x²) + √(1+y²))
#
# reduces that case to a small, well-resolved atanh argument.  When the panel
# straddles the target, the direct subtraction is already well-conditioned and
# avoids rounding the atanh argument to one for extremely small regularization.
@inline function _sqg_asinh_difference(u_a::T, u_b::T, h_eff::T,
                                       ds_len::T) where {T}
    signbit(u_a) != signbit(u_b) &&
        return asinh(u_a / h_eff) - asinh(u_b / h_eff)

    radius_a = hypot(u_a, h_eff)
    radius_b = hypot(u_b, h_eff)
    scale = max(radius_a, radius_b)
    ratio = (ds_len / scale) / (radius_a / scale + radius_b / scale)
    # Near an endpoint the ratio approaches one, where rounding can make
    # `atanh(ratio)` inaccurate or infinite.  In that regime the two asinh
    # values are well separated, so direct subtraction is the stable form.
    ratio < T(0.5) && return T(2) * atanh(ratio)
    return asinh(u_a / h_eff) - asinh(u_b / h_eff)
end

@inline function _sqg_erf_over_r(α::T, r2::T) where {T}
    if r2 > eps(T)^2
        r = sqrt(r2)
        return erf(α * r) / r
    end
    return T(2) * α / sqrt(T(π))
end
