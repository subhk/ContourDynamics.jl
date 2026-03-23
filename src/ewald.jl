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
    build_ewald_cache(domain::PeriodicDomain, kernel::SQGKernel; n_fourier=8, n_images=2)

Ewald cache for SQG kernel in periodic domain.

The SQG Green's function `G(r) = -1/(2πr)` is split via Ewald summation.
Fourier coefficients are `(2π/|k|) exp(-k²/(4α²)) / A`, reflecting the
fractional Laplacian's half-order (`1/|k|` vs Euler's `1/k²`).
"""
function build_ewald_cache(domain::PeriodicDomain{T}, kernel::SQGKernel{T};
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
const _ewald_caches_generic = Dict{Any, Any}()
const _ewald_key_order_generic = Any[]
_ewald_cache_dict(::Type{T}) where {T<:AbstractFloat} = _ewald_caches_generic
_ewald_key_order(::Type{T}) where {T<:AbstractFloat} = _ewald_key_order_generic

function _get_ewald_cache(domain::PeriodicDomain{T}, kernel::AbstractKernel) where {T}
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    # Fast path: check under lock, return immediately if cached
    existing = lock(_ewald_cache_lock) do
        get(caches, key, nothing)
    end
    existing !== nothing && return existing
    # Slow path: build outside lock to avoid blocking other threads
    new_cache = build_ewald_cache(domain, kernel)
    cache = lock(_ewald_cache_lock) do
        # Re-check: another thread may have inserted while we were building
        if haskey(caches, key)
            caches[key]
        else
            while length(caches) >= _EWALD_CACHE_MAX && !isempty(order)
                old_key = popfirst!(order)
                delete!(caches, old_key)
            end
            caches[key] = new_cache
            push!(order, key)
            new_cache
        end
    end
    return cache
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
decomposes as `G_QG_per = G_Euler_per + G_correction`, so the velocity path
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
    if x <= zero(T)
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
            s += term / T(n)
            abs(term / T(n)) < eps(T) * abs(s) && break
        end
        return s
    else
        # Continued fraction for x ≥ 2.
        # Convergence ratio ≈ 1/(x+1); at x=2 need ~54 terms for Float64.
        # Cap at 300 terms: for x ≥ 2 the CF converges well within this.
        ex = exp(-x)
        cf = zero(T)
        n_cf = min(300, max(60, ceil(Int, -log(eps(T)) / log(T(x) / (T(x) + one(T))))))
        for k in n_cf:-1:1
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
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

function segment_velocity(kernel::EulerKernel, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
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

        # Fourier-space sum (smooth).
        # r_vec0 is intentionally NOT minimum-image wrapped: cos(k·r) is
        # periodic with the same periods as the domain, so wrapping has no
        # effect on the phase.
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
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, EulerKernel()))
end

function segment_velocity(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           euler_cache::EwaldCache{T}) where {T}
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Euler periodic part (handles the log singularity via Ewald)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b, euler_cache)

    # Smooth QG–Euler correction via Fourier sum.
    # G_QG_per - G_Euler_per = (1/A) Σ_{k≠0} cos(k·r) κ²/(k²(k²+κ²))
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

"""
    segment_velocity(kernel::SQGKernel, domain::PeriodicDomain, x, a, b)

Velocity at point `x` from segment `a→b` in a periodic domain using the SQG kernel.

Uses singular subtraction: the regularized unbounded SQG velocity (exact arcsinh
antiderivative with ``\\delta > 0``) handles the ``1/r`` singularity analytically, and the
smooth periodic correction ``G_{\\text{per}} - G_\\infty`` is integrated with 3-point
Gauss-Legendre quadrature.

The periodic correction decomposes as:
- Central-image real-space: ``-(1/(2\\pi))[\\operatorname{erfc}(\\alpha r)/r - 1/\\sqrt{r^2+\\delta^2}]``
  (finite at GL quadrature points since ``r > 0``)
- Non-central real-space: ``-(1/(2\\pi)) \\operatorname{erfc}(\\alpha|\\mathbf{r}-\\mathbf{R}_n|)/|\\mathbf{r}-\\mathbf{R}_n|``
- Fourier space: ``-(1/(2\\pi)) \\sum c_k \\cos(\\mathbf{k}\\cdot\\mathbf{r})``
  with ``c_k = (2\\pi/|k|) e^{-k^2/(4\\alpha^2)}/A``
"""
function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    return segment_velocity(kernel, domain, x, a, b, _get_ewald_cache(domain, kernel))
end

function segment_velocity(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                           cache::EwaldCache{T}) where {T}
    alpha = cache.alpha
    delta = kernel.delta
    delta_sq = delta * delta
    Lx, Ly = domain.Lx, domain.Ly

    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})

    # Exact unbounded regularized SQG velocity (handles the 1/r singularity via δ)
    v_unbounded = segment_velocity(kernel, UnboundedDomain(), x, a, b)

    # Smooth periodic correction: G_per_SQG(r) - G_∞_SQG_reg(r)
    # G_∞_SQG_reg(r) = -(1/(2π))/√(r²+δ²) is the regularized unbounded Green's function.
    # All correction terms are finite at GL quadrature points (r > 0).
    g_nodes = SVector{3,T}(-sqrt(T(3)/T(5)), zero(T), sqrt(T(3)/T(5)))
    g_weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    mid = (a + b) / 2
    half_ds = ds / 2

    inv2pi = one(T) / (2 * T(π))

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
                    # Central image: evaluate Ewald real-space at √(r²+δ²) instead
                    # of r so the subtraction erfc(α·r_reg)/r_reg - 1/r_reg
                    # = -erf(α·r_reg)/r_reg is bounded by 1/δ for all r ≥ 0.
                    # This is an O(δ²/r) perturbation, within the existing
                    # regularization error. Avoids 1/r divergence when contour
                    # nodes from different PV levels pass very close together.
                    r_reg = sqrt(r2 + delta_sq)
                    G_corr += inv2pi * erf(alpha * r_reg) / r_reg
                else
                    # Non-central images: -(1/(2π)) erfc(α|r|)/|r|  (smooth)
                    if r2 > eps(T)
                        r = sqrt(r2)
                        G_corr -= inv2pi * erfc(alpha * r) / r
                    end
                end
            end
        end

        # Fourier-space sum: -(1/(2π)) Σ fourier_coeffs cos(k·r)
        for (mi, kxi) in enumerate(cache.kx)
            for (ni, kyi) in enumerate(cache.ky)
                coeff = cache.fourier_coeffs[mi, ni]
                abs(coeff) < eps(T) && continue
                phase = kxi * r_vec0[1] + kyi * r_vec0[2]
                G_corr -= inv2pi * coeff * cos(phase)
            end
        end

        corr_integral += g_weights[q] * G_corr
    end

    return v_unbounded + half_ds * corr_integral
end
