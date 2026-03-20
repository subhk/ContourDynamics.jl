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
    segment_velocity(kernel::EulerKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using Ewald summation.
"""
function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    v = zero(SVector{2,T})
    inv2pi = one(T) / (2 * T(π))

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds

        # Real-space sum over periodic images
        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = s_pt + shift - x
                r2 = r_vec[1]^2 + r_vec[2]^2
                r2 < eps(T) && continue
                r = sqrt(r2)

                screening = erfc(alpha * r) / r2
                perp = SVector{2,T}(-r_vec[2], r_vec[1])
                v = v + g_weights[q] * inv2pi * screening * perp
            end
        end

        # Fourier-space sum
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue

                phase_diff = (kxi * s_pt[1] + kyi * s_pt[2]) - (kxi * x[1] + kyi * x[2])
                s_phase = sin(phase_diff)
                v = v + g_weights[q] * coeff * SVector{2,T}(-kyi, kxi) * s_phase
            end
        end
    end

    return v * (ds_len / 2)
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

    v = zero(SVector{2,T})
    inv2pi = one(T) / (2 * T(π))

    for q in 1:3
        s_pt = mid + g_nodes[q] * half_ds

        for px in -cache.n_images:cache.n_images
            for py in -cache.n_images:cache.n_images
                shift = SVector{2,T}(2 * Lx * px, 2 * Ly * py)
                r_vec = s_pt + shift - x
                r2 = r_vec[1]^2 + r_vec[2]^2
                r2 < eps(T) && continue
                r = sqrt(r2)

                rr = r / Ld
                K1_val = besselk(1, rr)
                screening = erfc(alpha * r)
                perp = SVector{2,T}(-r_vec[2], r_vec[1]) / r
                v = v + g_weights[q] * inv2pi / Ld * K1_val * screening * perp
            end
        end

        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue

                phase_diff = (kxi * s_pt[1] + kyi * s_pt[2]) - (kxi * x[1] + kyi * x[2])
                s_phase = sin(phase_diff)
                v = v + g_weights[q] * coeff * SVector{2,T}(-kyi, kxi) * s_phase
            end
        end
    end

    return v * (ds_len / 2)
end
