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
    build_ewald_cache(domain::PeriodicDomain, kernel::EulerKernel; n_fourier=16, n_images=4)

Precompute Fourier-space coefficients for Ewald summation.
"""
function build_ewald_cache(domain::PeriodicDomain{T}, ::EulerKernel;
                           n_fourier::Int=16, n_images::Int=4) where {T}
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
    build_ewald_cache(domain::PeriodicDomain, kernel::QGKernel; n_fourier=16, n_images=4)

Ewald cache for QG kernel in periodic domain.
"""
function build_ewald_cache(domain::PeriodicDomain{T}, kernel::QGKernel{T};
                           n_fourier::Int=16, n_images::Int=4) where {T}
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

# Cache storage
const _ewald_caches = Dict{UInt64, EwaldCache}()
const _ewald_cache_lock = ReentrantLock()

function _get_ewald_cache(domain::PeriodicDomain, kernel::AbstractKernel)
    key = hash((domain.Lx, domain.Ly, kernel))
    lock(_ewald_cache_lock) do
        if !haskey(_ewald_caches, key)
            _ewald_caches[key] = build_ewald_cache(domain, kernel)
        end
    end
    return _ewald_caches[key]
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
        # Series: E₁(x) = -γ - ln(x) - Σ_{n=1}^∞ (-x)^n / (n * n!)
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

Uses the Ewald decomposition of the periodic Green's function G_per = G_real + G_fourier:
- Real space: G_real(r) = (1/4π) Σ_images E₁(α²|r+shift|²)
- Fourier space: G_fourier(r) = (1/A) Σ_{k≠0} exp(-k²/(4α²)) cos(k·r) / k²

The contour dynamics velocity integrates G along the segment:
  v_seg = ∫_a^b G_per(x - s) (-ds_y, ds_x)
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
    # Parameterize: s(t) = mid + t*half_ds, ds' = half_ds dt
    # (-ds'_y, ds'_x) = (-half_ds_y, half_ds_x) dt
    half_ds_perp = SVector{2,T}(-half_ds[2], half_ds[1])

    inv4pi = one(T) / (4 * T(π))

    v = zero(SVector{2,T})

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        G_val = zero(T)

        # Real-space sum: G_real = (1/4π) Σ_images E₁(α²r²)
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

        # Fourier-space sum: G_fourier = Σ_{k≠0} coeff_k * cos(k·r)
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                r_vec = x - s_pt
                phase = kxi * r_vec[1] + kyi * r_vec[2]
                G_val += coeff * cos(phase)
            end
        end

        v = v + g_weights[q] * G_val * half_ds_perp
    end

    # Sign: the contour dynamics velocity is v_seg = -∫ G (-dy', dx'),
    # and we computed ∫ G (-dy', dx'), so negate.
    return -v
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
    half_ds_perp = SVector{2,T}(-half_ds[2], half_ds[1])

    inv2pi = one(T) / (2 * T(π))

    v = zero(SVector{2,T})

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds
        G_val = zero(T)

        # Real-space: QG Green's function G(r) = -(1/2π) K₀(r/Ld), screened by erfc
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
                    G_val += -inv2pi * K0_val * screening
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

        v = v + g_weights[q] * G_val * half_ds_perp
    end

    return -v
end
