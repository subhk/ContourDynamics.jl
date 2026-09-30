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

# ── Modified Bessel function K₀ ───────────────────────────────────────────
#
# Float64 and Float32 evaluate precomputed tables, fitted in 512-bit arithmetic
# against K₀(x) = ∫₀^∞ exp(-x cosh t) dt (trapezoid rule):
#   x ≤ 2: power series in y = x²/4 (13 terms);
#   x > 2: e^x √x K₀(x) as a 26-term Chebyshev series in t = 4/x - 1.
# Both are accurate to a few ulps. Other float types sum the series and a
# continued fraction to their own precision instead.

# a_k = 1/(k!)² and b_k = (H_k - γ)/(k!)², k = 1:13, with H_k the harmonic
# numbers, so that I₀(x) - 1 = Σ a_k y^k and K₀(x) + γ + log(x/2) I₀(x) = Σ b_k y^k.
const _K0_SERIES_I0 = (
    1.0, 0.25, 0.027777777777777776, 0.001736111111111111, 6.944444444444444e-5,
    1.9290123456790124e-6, 3.936759889140842e-8, 6.151187326782565e-10,
    7.594058428126624e-12, 7.594058428126623e-14, 6.276081345559193e-16,
    4.358389823304995e-18, 2.5789288895295828e-20)
const _K0_SERIES_K0 = (
    0.42278433509846713, 0.23069608377461678, 0.0348921574564389,
    0.0026147876188052093, 0.00011848039364109726, 3.6126241031992037e-6,
    7.935096521304209e-8, 1.3167486730385647e-9, 1.709994072705808e-11,
    1.785934656987074e-13, 1.5330343403208473e-15, 1.1009270959725744e-17,
    6.712740659979047e-20)
# Chebyshev coefficients of e^x √x K₀(x), t = 4/x - 1 (first term halved).
const _K0_CHEBYSHEV = (
    1.2201515410329777, -0.0314481013119645, 0.0015698838857300533,
    -0.00012849549581627802, 1.3949813718876499e-5, -1.8317555227191193e-6,
    2.766813639445015e-7, -4.660489897687947e-8, 8.574034017414225e-9,
    -1.6975345093890614e-9, 3.5773972814003283e-10, -7.957489244477396e-11,
    1.8559491149549264e-11, -4.514597883374519e-12, 1.1403405882073441e-12,
    -2.9800969231481784e-13, 8.032890775068373e-14, -2.227513326746296e-14,
    6.340076476276645e-15, -1.848593377920907e-15, 5.5120559994043335e-16,
    -1.6782311257549006e-16, 5.210391777643554e-17, -1.6475805939842632e-17,
    5.3004337711773354e-18, -1.7331712005821e-18)

# K₀(z) + log(z/2) + γ = Σ b_k y^k - log(z/2)(I₀(z) - 1), y = z²/4, for z ≤ 2.
# Both sums are O(z²): no subtraction of two O(log(1/z)) quantities.
@inline function _besselk0_correction(z::T) where {T<:Union{Float32, Float64}}
    iszero(z) && return zero(T)
    y = z * z / 4
    i0_minus_1 = y * evalpoly(y, map(T, _K0_SERIES_I0))
    return y * evalpoly(y, map(T, _K0_SERIES_K0)) - log(z / 2) * i0_minus_1
end

@inline function _besselk0_correction(z::T) where {T}
    iszero(z) && return zero(T)
    y = z * z / 4
    γ = T(Base.MathConstants.eulergamma)
    i0_minus_1 = zero(T)
    s = zero(T)
    term = one(T)
    harmonic = zero(T)
    for k in 1:1000
        term *= y / T(k)^2
        harmonic += one(T) / T(k)
        i0_minus_1 += term
        s += term * (harmonic - γ)
        term * harmonic < eps(T) * abs(s) && break
    end
    return s - log(z / 2) * i0_minus_1
end

# Chebyshev series Σ c_k T_{k-1}(t) (c₁ already halved), by Clenshaw's recurrence.
@inline function _chebyshev_series(c::NTuple{N,T}, t::T) where {N, T}
    b1 = zero(T)
    b2 = zero(T)
    for k in N:-1:2
        b1, b2 = c[k] + 2 * t * b1 - b2, b1
    end
    return c[1] + t * b1 - b2
end

# K₀(x) for x ≥ 0 (K₀(0) = ∞).
@inline function _besselk0_scalar(x::T) where {T<:Union{Float32, Float64}}
    iszero(x) && return T(Inf)
    x <= 2 && return _besselk0_correction(x) - log(x / 2) - T(Base.MathConstants.eulergamma)
    return exp(-x) / sqrt(x) * _chebyshev_series(map(T, _K0_CHEBYSHEV), 4 / x - 1)
end

function _besselk0_scalar(x::T) where {T}
    iszero(x) && return T(Inf)
    x <= 2 && return _besselk0_correction(x) - log(x / 2) - T(Base.MathConstants.eulergamma)
    # Steed's evaluation of Temme's continued fraction CF2 (Numerical Recipes
    # §6.7, order 0); it converges for x ≥ 2 at any working precision.
    b = 2 * (1 + x)
    d = 1 / b
    delh = d
    q1 = zero(T)
    q2 = one(T)
    a1 = one(T) / 4
    q = a1
    c = a1
    a = -a1
    s = 1 + q * delh
    for i in 2:1_000_000
        a -= 2 * (i - 1)
        c = -a * c / i
        q1, q2 = q2, (q1 - b * q2) / a
        q += c * q2
        b += 2
        d = 1 / (b + a * d)
        delh = (b * d - 1) * delh
        dels = q * delh
        s += dels
        abs(dels) < eps(T) * abs(s) && break
    end
    return sqrt(T(π) / (2 * x)) * exp(-x) / s
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
