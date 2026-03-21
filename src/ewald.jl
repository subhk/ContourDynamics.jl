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

# Cache storage — keyed by concrete (Lx, Ly, kernel_type, Ld) tuple to
# avoid hash collisions.  Ld = 0 for EulerKernel.
const _EwaldCacheKey = Tuple{Any, Any, Any, Any}   # (Lx, Ly, kernel type, Ld)
const _ewald_caches = Dict{_EwaldCacheKey, Any}()
const _ewald_cache_lock = ReentrantLock()
const _EWALD_CACHE_MAX = 64  # prevent unbounded growth

_cache_key(domain::PeriodicDomain, ::EulerKernel) = (domain.Lx, domain.Ly, EulerKernel, 0)
_cache_key(domain::PeriodicDomain, k::QGKernel) = (domain.Lx, domain.Ly, QGKernel, k.Ld)

function _get_ewald_cache(domain::PeriodicDomain{T}, kernel::AbstractKernel) where {T}
    key = _cache_key(domain, kernel)
    cache = lock(_ewald_cache_lock) do
        if !haskey(_ewald_caches, key)
            if length(_ewald_caches) >= _EWALD_CACHE_MAX
                empty!(_ewald_caches)
            end
            _ewald_caches[key] = build_ewald_cache(domain, kernel)
        end
        _ewald_caches[key]
    end
    return cache::EwaldCache{T}
end

"""Clear all cached Ewald data."""
function clear_ewald_cache!()
    lock(_ewald_cache_lock) do
        empty!(_ewald_caches)
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

The contour dynamics segment velocity for the Euler kernel is:
  v_seg = ds * ∫₀¹ G_per(x - s(t)) dt

where G_per is the periodic Green's function and ds = b - a.

The Ewald decomposition splits G_per = G_real + G_fourier:
- Real space: G_real(r) = +(1/(4π)) Σ_images E₁(α²|r+shift|²)
- Fourier space: G_fourier(r) = (1/A) Σ_{k≠0} exp(-k²/(4α²)) cos(k·r) / k²
"""
function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # 3-point Gauss-Legendre quadrature on [-1,1]
    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    inv4pi = one(T) / (4 * T(π))

    G_integral = zero(T)

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        G_val = zero(T)

        # Real-space Ewald sum for G(r) = -(1/(2π)) log(r) = -(1/(4π)) log(r²).
        # The Ewald real-space part is: +(1/(4π)) Σ_images E₁(α²|r+shift|²).
        # For small α²r²: E₁(x) ≈ -ln(x) - γ, so
        #   (1/(4π)) E₁(α²r²) ≈ -(1/(4π)) ln(r²) + const = G(r) + const.
        # Thus the real-space sum recovers G for the central image, with
        # exponentially decaying contributions from periodic images.
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = x - s_pt - shift
                r2 = r_vec[1]^2 + r_vec[2]^2
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
                r_vec = x - s_pt
                phase = kxi * r_vec[1] + kyi * r_vec[2]
                G_val += coeff * cos(phase)
            end
        end

        G_integral += g_weights[q] * G_val
    end

    # v_seg = ds * (1/2) * G_integral
    # The 1/2 comes from the change of variables: ∫₀¹ dt = (1/2) ∫_{-1}^{1} dt'
    return half_ds * G_integral
end

function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    Ld = kernel.Ld

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    inv2pi = one(T) / (2 * T(π))

    G_integral = zero(T)

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        G_val = zero(T)

        # Real-space: QG Green's function G(r) = -(1/2π) K₀(r/Ld), screened
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = x - s_pt - shift
                r2 = r_vec[1]^2 + r_vec[2]^2
                if r2 > eps(T)
                    r = sqrt(r2)
                    rr = r / Ld
                    K0_val = besselk(0, rr)
                    screening = erfc(alpha * r)
                    G_val -= inv2pi * K0_val * screening
                end
            end
        end

        # Fourier-space sum
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                r_vec = x - s_pt
                phase = kxi * r_vec[1] + kyi * r_vec[2]
                G_val += coeff * cos(phase)
            end
        end

        G_integral += g_weights[q] * G_val
    end

    return half_ds * G_integral
end
