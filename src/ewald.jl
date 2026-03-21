"""
    EwaldCache{T}

Precomputed data for Ewald summation in periodic domains.
"""
struct EwaldCache{T<:AbstractFloat}
    alpha::T
    kx::Vector{T}
    ky::Vector{T}
    fourier_coeffs::Matrix{T}
    n_images::Int
end

"""
    build_ewald_cache(domain::PeriodicDomain, kernel::EulerKernel; n_fourier=8, n_images=2)

Precompute Fourier-space coefficients for Ewald summation.
"""
function build_ewald_cache(domain::PeriodicDomain{T}, ::EulerKernel;
                           n_fourier::Int=8, n_images::Int=2) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    alpha = sqrt(T(π)) / min(Lx, Ly)

    kx = [T(2π * m) / (2 * Lx) for m in -n_fourier:n_fourier]
    ky = [T(2π * n) / (2 * Ly) for n in -n_fourier:n_fourier]

    nk = length(kx)
    fourier_coeffs = zeros(T, nk, nk)
    area = 4 * Lx * Ly

    for (mi, kxi) in enumerate(kx)
        for (ni, kyi) in enumerate(ky)
            k2 = kxi^2 + kyi^2
            if k2 > eps(T)
                fourier_coeffs[mi, ni] = exp(-k2 / (4 * alpha^2)) / (k2 * area)
            end
        end
    end

    return EwaldCache(alpha, kx, ky, fourier_coeffs, n_images)
end

"""
    build_ewald_cache(domain::PeriodicDomain, kernel::QGKernel; n_fourier=8, n_images=2)

Ewald cache for QG kernel in periodic domain.
"""
function build_ewald_cache(domain::PeriodicDomain{T}, kernel::QGKernel{T};
                           n_fourier::Int=8, n_images::Int=2) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    Ld = kernel.Ld
    alpha = sqrt(T(π)) / min(Lx, Ly)

    kx = [T(2π * m) / (2 * Lx) for m in -n_fourier:n_fourier]
    ky = [T(2π * n) / (2 * Ly) for n in -n_fourier:n_fourier]

    nk = length(kx)
    fourier_coeffs = zeros(T, nk, nk)
    area = 4 * Lx * Ly

    for (mi, kxi) in enumerate(kx)
        for (ni, kyi) in enumerate(ky)
            k2 = kxi^2 + kyi^2
            if k2 > eps(T)
                fourier_coeffs[mi, ni] = exp(-k2 / (4 * alpha^2)) /
                    ((k2 + one(T) / Ld^2) * area)
            end
        end
    end

    return EwaldCache(alpha, kx, ky, fourier_coeffs, n_images)
end

# Cache storage — keyed by (Lx, Ly, kernel_type, Ld) hash.
# Uses UInt64 key and typed value Dict for type stability.
const _ewald_caches_f64 = Dict{UInt64, EwaldCache{Float64}}()
const _ewald_caches_f32 = Dict{UInt64, EwaldCache{Float32}}()
const _ewald_cache_lock = ReentrantLock()
const _EWALD_CACHE_MAX = 64  # prevent unbounded growth

function _cache_key_hash(domain::PeriodicDomain, ::EulerKernel)
    hash((domain.Lx, domain.Ly, EulerKernel, 0))
end
function _cache_key_hash(domain::PeriodicDomain, k::QGKernel)
    hash((domain.Lx, domain.Ly, QGKernel, k.Ld))
end

_ewald_cache_dict(::Type{Float64}) = _ewald_caches_f64
_ewald_cache_dict(::Type{Float32}) = _ewald_caches_f32

function _get_ewald_cache(domain::PeriodicDomain{T}, kernel::AbstractKernel) where {T}
    key = _cache_key_hash(domain, kernel)
    caches = _ewald_cache_dict(T)
    cache = lock(_ewald_cache_lock) do
        if !haskey(caches, key)
            if length(caches) >= _EWALD_CACHE_MAX
                empty!(caches)
            end
            caches[key] = build_ewald_cache(domain, kernel)
        end
        caches[key]
    end
    return cache
end

"""Clear all cached Ewald data."""
function clear_ewald_cache!()
    lock(_ewald_cache_lock) do
        empty!(_ewald_caches_f64)
        empty!(_ewald_caches_f32)
    end
end

"""
    _expint_e1(x)

Compute the exponential integral E₁(x) = ∫_x^∞ e^{-t}/t dt for x > 0.
"""
function _expint_e1(x::T) where {T<:AbstractFloat}
    if x <= zero(T)
        return T(Inf)
    end
    if x < one(T)
        # Series: E₁(x) = -γ - ln(x) + Σ_{n=1}^∞ (-1)^{n+1} x^n / (n * n!)
        γ = T(Base.MathConstants.eulergamma)
        s = -γ - log(x)
        term = one(T)
        for n in 1:50
            term *= -x / T(n)
            s += term / T(n)
            abs(term / n) < eps(T) * abs(s) && break
        end
        return s
    else
        # Continued fraction for large x
        ex = exp(-x)
        cf = zero(T)
        for k in 30:-1:1
            cf = T(k) / (one(T) + T(k) / (x + cf))
        end
        return ex / (x + cf)
    end
end

"""
    segment_velocity(kernel::EulerKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using Ewald summation.

Uses singular subtraction: the log singularity is handled analytically by the
unbounded Euler segment velocity, and only the smooth periodic correction
`G_per - G_∞` is integrated with 3-point Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: (1/(4π))[E₁(α²r²) + log(r²)] → (1/(4π))(-γ - 2ln α) as r→0
- Non-central real-space: (1/(4π)) Σ_{images≠0} E₁(α²|r+shift|²)  (smooth)
- Fourier space: Σ_{k≠0} coeff * cos(k·r)  (smooth)
"""
function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Analytic unbounded Euler contribution (handles the log singularity exactly)
    v_unbounded = segment_velocity(EulerKernel(), UnboundedDomain(), x, a, b)

    # Smooth periodic correction: G_per(r) - G_∞(r)
    # G_∞(r) = -(1/(4π)) log(r²), so we subtract it from the central Ewald image.
    # All terms in the correction are bounded at r=0 → GL quadrature is accurate.
    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    inv4pi = one(T) / (4 * T(π))
    gamma_euler = T(Base.MathConstants.eulergamma)

    corr_integral = zero(T)

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        r_vec0 = x - s_pt
        G_corr = zero(T)

        # Real-space Ewald sum with central-image singularity subtracted
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = r_vec0 - shift
                r2 = r_vec[1]^2 + r_vec[2]^2
                if px == 0 && py == 0
                    # Central image: compute E₁(α²r²) + log(r²) which is smooth at r=0.
                    # Limit as r→0: E₁(x) + log(x) → -γ, and log(r²) = log(x/α²),
                    # so E₁(α²r²) + log(r²) → -γ - 2*log(α).
                    if r2 > eps(T)
                        G_corr += inv4pi * (_expint_e1(alpha^2 * r2) + log(r2))
                    else
                        G_corr += inv4pi * (-gamma_euler - 2 * log(alpha))
                    end
                else
                    # Non-central images: already smooth
                    if r2 > eps(T)
                        G_corr += inv4pi * _expint_e1(alpha^2 * r2)
                    end
                end
            end
        end

        # Fourier-space sum (smooth)
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                phase = kxi * r_vec0[1] + kyi * r_vec0[2]
                G_corr += coeff * cos(phase)
            end
        end

        corr_integral += g_weights[q] * G_corr
    end

    # v_periodic = v_unbounded + ds * ∫₀¹ [G_per - G_∞] dt
    # The ∫₀¹ dt = (1/2) ∫_{-1}^{1} dt' gives the half_ds factor.
    return v_unbounded + half_ds * corr_integral
end

"""
    segment_velocity(kernel::QGKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using the QG kernel.

Decomposes the periodic QG Green's function as:
  G_QG_per = G_Euler_per + G_correction
where the Euler part is handled by the validated Ewald summation, and the
correction is a smooth, rapidly convergent Fourier series:
  G_corr(r) = (1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²))
with κ = 1/Ld.  Coefficients decay as 1/k⁴, so the truncated sum converges
without Gaussian damping.
"""
function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Euler periodic part (handles the log singularity via Ewald)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b)

    # Smooth QG–Euler correction via Fourier sum.
    # G_QG_per - G_Euler_per = (1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²))
    euler_cache = _get_ewald_cache(domain, EulerKernel())
    kappa2 = one(T) / kernel.Ld^2
    area = 4 * domain.Lx * domain.Ly

    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    corr_integral = zero(T)
    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        r_vec = x - s_pt
        G_corr = zero(T)
        for kxi in euler_cache.kx
            for kyi in euler_cache.ky
                k2 = kxi^2 + kyi^2
                k2 < eps(T) && continue
                coeff = kappa2 / (k2 * (k2 + kappa2) * area)
                phase = kxi * r_vec[1] + kyi * r_vec[2]
                G_corr += coeff * cos(phase)
            end
        end
        corr_integral += g_weights[q] * G_corr
    end

    return v_euler + half_ds * corr_integral
end
