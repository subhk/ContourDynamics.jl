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
    alpha = sqrt(T(π)) / sqrt(Lx * Ly)

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
    build_ewald_cache(domain::PeriodicDomain, kernel::SQGKernel; n_fourier=8, n_images=2)

Ewald cache for SQG kernel in periodic domain.

The SQG Green's function `G(r) = -1/(2πr)` is split via Ewald summation.
Fourier coefficients are `(2π/|k|) exp(-k²/(4α²)) / A`, reflecting the
fractional Laplacian's half-order (`1/|k|` vs Euler's `1/k²`).
"""
function build_ewald_cache(domain::PeriodicDomain{T}, kernel::SQGKernel{T};
                           n_fourier::Int=8, n_images::Int=2) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    alpha = sqrt(T(π)) / sqrt(Lx * Ly)

    kx = [T(2π * m) / (2 * Lx) for m in -n_fourier:n_fourier]
    ky = [T(2π * n) / (2 * Ly) for n in -n_fourier:n_fourier]

    nk = length(kx)
    fourier_coeffs = zeros(T, nk, nk)
    area = 4 * Lx * Ly

    for (mi, kxi) in enumerate(kx)
        for (ni, kyi) in enumerate(ky)
            k2 = kxi^2 + kyi^2
            if k2 > eps(T)
                k_mag = sqrt(k2)
                fourier_coeffs[mi, ni] = 2 * T(π) * exp(-k2 / (4 * alpha^2)) / (k_mag * area)
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
    alpha = sqrt(T(π)) / sqrt(Lx * Ly)

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

# Cache storage — keyed by (Lx, Ly, kernel_type, Ld) tuples with snapped values.
# Values are snapped to a canonical grid (1024 ULPs) so that near-identical
# domain parameters from different arithmetic paths share the same cache entry.
# FIFO eviction via _ewald_key_order vectors: oldest entries evicted first.
#
# NOTE: These are module-level globals shared across all problems and threads.
# clear_ewald_cache!() affects ALL concurrent simulations.  Tests should call
# it in a setup block to avoid cache pollution between test cases.
@inline function _snap(x::T) where {T<:AbstractFloat}
    e = T(1024) * eps(x)
    return round(x / e) * e
end
const _EwaldCacheKey{T} = Tuple{T, T, DataType, T}  # (Lx, Ly, kernel_type, Ld)
const _ewald_caches_f64 = Dict{_EwaldCacheKey{Float64}, EwaldCache{Float64}}()
const _ewald_caches_f32 = Dict{_EwaldCacheKey{Float32}, EwaldCache{Float32}}()
const _ewald_key_order_f64 = _EwaldCacheKey{Float64}[]
const _ewald_key_order_f32 = _EwaldCacheKey{Float32}[]
const _ewald_cache_lock = ReentrantLock()
const _EWALD_CACHE_MAX = 64  # prevent unbounded growth

function _cache_key(domain::PeriodicDomain{T}, ::EulerKernel) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), EulerKernel, zero(T))::_EwaldCacheKey{T}
end
function _cache_key(domain::PeriodicDomain{T}, k::QGKernel{T}) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), QGKernel{T}, _snap(k.Ld))::_EwaldCacheKey{T}
end
function _cache_key(domain::PeriodicDomain{T}, k::SQGKernel{T}) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), SQGKernel{T}, _snap(k.delta))::_EwaldCacheKey{T}
end

_ewald_cache_dict(::Type{Float64}) = _ewald_caches_f64
_ewald_cache_dict(::Type{Float32}) = _ewald_caches_f32
_ewald_key_order(::Type{Float64}) = _ewald_key_order_f64
_ewald_key_order(::Type{Float32}) = _ewald_key_order_f32

# Generic fallback for other float types (BigFloat, Float16, etc.)
const _ewald_caches_generic = Dict{DataType, Any}()
const _ewald_key_order_generic = Dict{DataType, Any}()

function _ewald_cache_dict(::Type{T}) where {T<:AbstractFloat}
    caches = get!(_ewald_caches_generic, T) do
        Dict{_EwaldCacheKey{T}, EwaldCache{T}}()
    end
    return caches::Dict{_EwaldCacheKey{T}, EwaldCache{T}}
end

function _ewald_key_order(::Type{T}) where {T<:AbstractFloat}
    order = get!(_ewald_key_order_generic, T) do
        _EwaldCacheKey{T}[]
    end
    return order::_EwaldCacheKey{T}[]
end

function _get_ewald_cache(domain::PeriodicDomain{T}, kernel::AbstractKernel) where {T}
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    # Fast path: quick read under lock to check if cache already exists.
    # After warm-up, this is the only lock acquisition needed per call.
    cached = lock(_ewald_cache_lock) do
        get(caches, key, nothing)
    end
    cached !== nothing && return cached

    # Slow path: build outside the lock, then store under the lock.
    new_cache = build_ewald_cache(domain, kernel)

    lock(_ewald_cache_lock) do
        # Double-check: another thread may have built it while we were computing.
        existing = get(caches, key, nothing)
        if existing !== nothing
            return existing
        end
        while length(caches) >= _EWALD_CACHE_MAX && !isempty(order)
            old_key = popfirst!(order)
            delete!(caches, old_key)
        end
        caches[key] = new_cache
        push!(order, key)
        return new_cache
    end
end

# Pre-fetch Ewald cache for use in threaded velocity computation.
# Returns `nothing` for unbounded domains (no cache needed).
_prefetch_ewald(::UnboundedDomain, ::AbstractKernel) = nothing
_prefetch_ewald(domain::PeriodicDomain, ::EulerKernel) = _get_ewald_cache(domain, EulerKernel())
_prefetch_ewald(domain::PeriodicDomain, kernel::QGKernel) = _get_ewald_cache(domain, EulerKernel())
_prefetch_ewald(domain::PeriodicDomain, kernel::SQGKernel) = _get_ewald_cache(domain, kernel)

"""
    setup_ewald_cache!(domain, kernel; n_fourier=8, n_images=2)

Pre-build and store an Ewald cache with custom parameters.  Call this before
`evolve!` to override the default `n_fourier=8`, `n_images=2`.  The cached
result is used automatically by all subsequent velocity computations on
the same domain/kernel combination.
"""
function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::AbstractKernel;
                            n_fourier::Int=8, n_images::Int=2) where {T}
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    lock(_ewald_cache_lock) do
        if !haskey(caches, key)
            while length(caches) >= _EWALD_CACHE_MAX && !isempty(order)
                old_key = popfirst!(order)
                delete!(caches, old_key)
            end
            push!(order, key)
        end
        caches[key] = build_ewald_cache(domain, kernel; n_fourier=n_fourier, n_images=n_images)
    end
    return nothing
end

"""
    setup_ewald_cache!(domain, kernel::QGKernel; n_fourier=8, n_images=2)

Pre-build Ewald caches for QG velocity computation.  The periodic QG velocity
decomposes as `G_QG_per = G_Euler_per - G_correction`, so the velocity path
uses an Euler Ewald cache internally.  This method builds both the QG-specific
cache and the Euler cache that the velocity path actually reads, ensuring that
custom `n_fourier`/`n_images` parameters take effect.
"""
function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::QGKernel{T};
                            n_fourier::Int=8, n_images::Int=2) where {T}
    # Store QG-specific cache (for direct queries / introspection)
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    lock(_ewald_cache_lock) do
        if !haskey(caches, key)
            while length(caches) >= _EWALD_CACHE_MAX && !isempty(order)
                old_key = popfirst!(order)
                delete!(caches, old_key)
            end
            push!(order, key)
        end
        caches[key] = build_ewald_cache(domain, kernel; n_fourier=n_fourier, n_images=n_images)
    end
    # Also build the Euler cache that the QG velocity path actually uses
    setup_ewald_cache!(domain, EulerKernel(); n_fourier=n_fourier, n_images=n_images)
    return nothing
end

"""Clear all cached Ewald data."""
function clear_ewald_cache!()
    lock(_ewald_cache_lock) do
        empty!(_ewald_caches_f64)
        empty!(_ewald_caches_f32)
        empty!(_ewald_caches_generic)
        empty!(_ewald_key_order_f64)
        empty!(_ewald_key_order_f32)
        empty!(_ewald_key_order_generic)
    end
end

"""
    _expint_e1(x)

Compute the exponential integral E₁(x) = ∫_x^∞ e^{-t}/t dt for x > 0.
"""
function _expint_e1(x::T) where {T<:AbstractFloat}
    x < zero(T) && throw(DomainError(x, "E₁(x) is not real-valued for x < 0"))
    if x == zero(T)
        return T(Inf)
    end
    if x < T(2)
        # Series: E₁(x) = -γ - ln(x) + Σ_{n=1}^∞ (-1)^{n+1} x^n / (n * n!)
        # Converges well for x < 2: terms decay as x^n/(n*n!).
        γ = T(Base.MathConstants.eulergamma)
        s = -γ - log(x)
        term = one(T)
        max_terms = max(60, ceil(Int, -2 * log(eps(T))))  # scale with precision
        for n in 1:max_terms
            term *= -x / T(n)
            s -= term / T(n)
            abs(term / T(n)) < eps(T) * abs(s) && break
        end
        return s
    else
        # Continued fraction for x ≥ 2.
        # Convergence ratio ≈ 1/(x+1); at x=2 need ~54 terms for Float64.
        # Cap at 300 terms: for x ≥ 2 the CF converges well within this.
        # For very large x, E₁(x) ≈ exp(-x)/x underflows to zero; short-circuit
        # to avoid InexactError from ceil(Int, Inf) when log(x/(x+1)) rounds to 0.
        ex = exp(-x)
        ex == zero(T) && return zero(T)
        cf = zero(T)
        log_ratio = log(T(x) / (T(x) + one(T)))
        n_cf = if log_ratio < -eps(T)
            min(300, max(60, ceil(Int, -log(eps(T)) / (-log_ratio))))
        else
            60  # x so large the CF converges trivially
        end
        for k in n_cf:-1:1
            cf = T(k) / (one(T) + T(k) / (x + cf))
        end
        return ex / (x + cf)
    end
end
