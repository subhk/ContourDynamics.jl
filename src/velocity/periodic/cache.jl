"""
    EwaldCache{T}

Precomputed data for Ewald summation in periodic domains. The cache stores the
splitting parameter, Fourier wavenumbers and coefficients, and the real-space
image radius used by periodic velocity and energy evaluations.

`fourier_coeffs` are the velocity Green's-function coefficients (periodic Euler
coefficients for the QG cache), `corr_coeffs` the Gaussian-damped QG-minus-Euler
velocity correction (QG only), and `energy_coeffs` the Fourier coefficients of
the periodic contour-energy potential. All tables are aligned to `kx` × `ky`,
which must be uniform grids symmetric about zero, and must be even in `kx` and
in `ky`, as every coefficient that depends on `|k|` only is.
"""
mutable struct EwaldCache{T<:AbstractFloat}
    # All fields are constant: a cache is never modified, but as a mutable
    # struct it is shared by reference, so reading it from the cache registry
    # does not copy its twelve fields into a fresh box (Julia 1.10 and 1.11
    # do on every lookup).
    const α::T
    const kx::Vector{T}
    const ky::Vector{T}
    const fourier_coeffs::Matrix{T}
    const n_images::Int
    # QG only: Fourier coefficients κ²ĉ_k of the Ewald-split QG-minus-Euler
    # velocity correction (see numerics/periodic_ewald.jl). Empty (0×0) for the
    # Euler and SQG caches.
    const corr_coeffs::Matrix{T}
    const energy_coeffs::Matrix{T}
    # Derived by the constructor for `_ewald_cosine_sum`: the grid spacings
    # and each table folded onto the modes m, n ≥ 0 (0×0 when absent).
    const dkx::T
    const dky::T
    const fourier_cos::Matrix{T}
    const corr_cos::Matrix{T}
    const energy_cos::Matrix{T}

    function EwaldCache(α::T, kx::Vector{T}, ky::Vector{T}, fourier_coeffs::Matrix{T},
                        n_images::Integer, corr_coeffs::Matrix{T},
                        energy_coeffs::Matrix{T}) where {T<:AbstractFloat}
        fold(table, name) = _fold_ewald_table(table, length(kx), length(ky), name)
        return new{T}(α, kx, ky, fourier_coeffs, Int(n_images), corr_coeffs,
                      energy_coeffs,
                      _ewald_grid_spacing(kx, :kx), _ewald_grid_spacing(ky, :ky),
                      fold(fourier_coeffs, :fourier_coeffs),
                      fold(corr_coeffs, :corr_coeffs),
                      fold(energy_coeffs, :energy_coeffs))
    end
end

# Six-field form kept for callers that build a cache by hand; it carries no
# energy table.
EwaldCache(α::T, kx::Vector{T}, ky::Vector{T}, fourier_coeffs::Matrix{T},
           n_images::Integer, corr_coeffs::Matrix{T}) where {T<:AbstractFloat} =
    EwaldCache(α, kx, ky, fourier_coeffs, Int(n_images), corr_coeffs, zeros(T, 0, 0))

# ASCII spelling retained for backwards compatibility.
function Base.getproperty(cache::EwaldCache, name::Symbol)
    name === :alpha && return getfield(cache, :α)
    return getfield(cache, name)
end

Base.propertynames(::EwaldCache, private::Bool=false) =
    (:α, :alpha, :kx, :ky, :fourier_coeffs, :n_images, :corr_coeffs, :energy_coeffs,
     (private ? (:dkx, :dky, :fourier_cos, :corr_cos, :energy_cos) : ())...)

# Spacing dk of a wavenumber grid dk·(-K:K); zero when it has no nonzero mode.
function _ewald_grid_spacing(k::Vector{T}, name::Symbol) where {T}
    isempty(k) && return zero(T)
    isodd(length(k)) || throw(ArgumentError(
        "EwaldCache `$name` must be symmetric about zero; got $(length(k)) wavenumbers"))
    K = length(k) ÷ 2
    dk = K == 0 ? zero(T) : k[K + 2]
    for m in -K:K
        abs(k[K + 1 + m] - m * dk) <= 16 * eps(T) * abs(m * dk) || throw(ArgumentError(
            "EwaldCache `$name` must be a uniform grid dk·(-K:K)"))
    end
    return dk
end

# Fold a table aligned to the kx × ky grid onto the modes m, n ≥ 0 for
# `_ewald_cosine_sum`. Summing the distinct sign variants (±m, ±n) gives the
# exact cos·cos coefficient; the sin·sin remainder must vanish.
function _fold_ewald_table(c::Matrix{T}, nkx::Int, nky::Int, name::Symbol) where {T}
    isempty(c) && return zeros(T, 0, 0)
    size(c) == (nkx, nky) || throw(DimensionMismatch(
        "EwaldCache `$name` is $(size(c, 1))×$(size(c, 2)) but the wavenumber grid is $nkx×$nky"))
    Kx, Ky = nkx ÷ 2, nky ÷ 2
    tol = 1024 * eps(T) * maximum(abs, c)
    w = zeros(T, Kx + 1, Ky + 1)
    for n in 0:Ky, m in 0:Kx
        pp = c[Kx + 1 + m, Ky + 1 + n]
        mp = c[Kx + 1 - m, Ky + 1 + n]
        pm = c[Kx + 1 + m, Ky + 1 - n]
        mm = c[Kx + 1 - m, Ky + 1 - n]
        w[m + 1, n + 1] = m == 0 ? (n == 0 ? pp : pp + pm) :
                          n == 0 ? pp + mp : pp + mp + pm + mm
        m > 0 && n > 0 && abs(pp - mp - pm + mm) > tol && throw(ArgumentError(
            "EwaldCache `$name` must be even in kx and in ky"))
    end
    return w
end

# Cosine table pulled out of `cache` for an evaluation that needs it. A cache
# built by hand may lack the QG correction or energy table, and summing with
# its empty stand-in would silently drop that Fourier part.
@inline function _required_ewald_table(cache::EwaldCache, table::Matrix, name::Symbol)
    isempty(table) && !isempty(cache.kx) && throw(ArgumentError(
        "EwaldCache has no `$name` table; build the cache with build_ewald_cache"))
    return table
end

@inline function _validate_ewald_truncation(n_fourier::Int, n_images::Int)
    n_fourier >= 0 || throw(ArgumentError(
        "n_fourier must be non-negative; got $n_fourier"))
    n_images >= 0 || throw(ArgumentError(
        "n_images must be non-negative; got $n_images"))
    return nothing
end

# Splitting parameter for `n_fourier` modes and `n_images` image rings. Both
# sums stop where their Gaussian factor falls below e^{-cutoff}: the largest α
# whose first omitted Fourier mode k does keeps the real-space sums as short
# as the Fourier table allows (their reach is √cutoff/α). When the image block
# cannot reach that far (few images, or high precision), α instead balances
# the two truncation errors, e^{-α²R²} = e^{-k²/4α²} at the block edge R.
function _ewald_alpha(Lx::T, Ly::T, n_fourier::Int, n_images::Int) where {T}
    k_omitted = T(π) * (n_fourier + 1) / max(Lx, Ly)
    block = (2 * n_images + 1) * min(Lx, Ly)
    return max(k_omitted / (2 * sqrt(_ewald_real_cutoff(T))),
               sqrt(k_omitted / (2 * block)))
end

# Shared Ewald setup: splitting parameter, Fourier wavenumbers, and domain area.
function _ewald_wavenumbers(domain::PeriodicDomain{T}, n_fourier::Int,
                            n_images::Int) where {T}
    Lx, Ly = domain.Lx, domain.Ly
    α = _ewald_alpha(Lx, Ly, n_fourier, n_images)
    # `2π * m` would contaminate extended-precision types with a Float64
    # product, so convert π to T before touching the integer index.
    kx = [T(π) * m / Lx for m in -n_fourier:n_fourier]
    ky = [T(π) * n / Ly for n in -n_fourier:n_fourier]
    area = 4 * Lx * Ly
    return α, kx, ky, area
end

# Per-kernel Ewald Fourier coefficient at squared wavenumber `k2`:
# - Euler: exp(-k²/4α²)/(k²A), the standard Ewald split of the 2-D log kernel.
# - SQG:   the 2-D transform of erf(α r_δ)/r_δ over A, the quasi-2-D Ewald
#          split of the regularized kernel 1/r_δ; at δ = 0 it reduces to
#          (2π/|k|) erfc(|k|/2α)/A, the fractional Laplacian's half-order
#          (1/|k| vs Euler's 1/k²). The erfcx form keeps e^{|k|δ} finite.
@inline _ewald_fourier_coefficient(::EulerKernel, k2::T, α::T, area::T) where {T} =
    exp(-k2 / (4 * α^2)) / (k2 * area)
@inline function _ewald_fourier_coefficient(kernel::SQGKernel{T}, k2::T, α::T,
                                            area::T) where {T}
    k = sqrt(k2)
    δ = kernel.δ
    y = k / (2 * α)
    return T(π) / (k * area) *
           (exp(-y * y - α * α * δ * δ) * erfcx(y + α * δ) + exp(-k * δ) * erfc(y - α * δ))
end

# Fourier coefficient of the periodic contour-energy potential at k ≠ 0 (see
# numerics/periodic_ewald.jl): 4π times the scaled QG correction, whose κ = 0
# limit is the Euler potential, and the neutralized SQG potential.
@inline _ewald_energy_coefficient(::EulerKernel, k2::T, α::T, area::T) where {T} =
    _scaled_qg_correction_coefficient(k2, zero(T), α, area, 4 * T(π))
@inline _ewald_energy_coefficient(kernel::QGKernel{T}, k2::T, α::T, area::T) where {T} =
    _scaled_qg_correction_coefficient(k2, inv(kernel.Ld^2), α, area, 4 * T(π))
@inline _ewald_energy_coefficient(kernel::SQGKernel{T}, k2::T, α::T, area::T) where {T} =
    _sqg_energy_fourier_coefficient(k2, α, kernel.δ, area)

"""
    build_ewald_cache(domain::PeriodicDomain, kernel; n_fourier=16, n_images=2)

Precompute Fourier-space coefficients for Ewald summation. The Euler and SQG
caches differ only in the coefficient formulas; the QG cache additionally
carries the QG correction table (see its method).

The splitting parameter `α` is chosen from the truncation: the largest value
whose first omitted Fourier mode is negligible at the working precision, so
more Fourier modes make the real-space image sums shorter-ranged. If the
`n_images` block cannot then keep the real-space truncation equally small, `α`
balances the two truncation errors instead.
"""
function build_ewald_cache(domain::PeriodicDomain{T},
                           kernel::Union{EulerKernel, SQGKernel{T}};
                           n_fourier::Int=16, n_images::Int=2) where {T}
    _validate_ewald_truncation(n_fourier, n_images)
    α, kx, ky, area = _ewald_wavenumbers(domain, n_fourier, n_images)
    nk = length(kx)
    fourier_coeffs = zeros(T, nk, nk)
    energy_coeffs = zeros(T, nk, nk)
    for (mi, kxi) in enumerate(kx)
        for (ni, kyi) in enumerate(ky)
            k2 = kxi^2 + kyi^2
            # Only the zero mode is absent; nonzero wavenumbers become small
            # as the numeric domain lengths grow.
            if !iszero(k2)
                fourier_coeffs[mi, ni] = _ewald_fourier_coefficient(kernel, k2, α, area)
                energy_coeffs[mi, ni] = _ewald_energy_coefficient(kernel, k2, α, area)
            end
        end
    end
    return EwaldCache(α, kx, ky, fourier_coeffs, n_images, zeros(T, 0, 0), energy_coeffs)
end

"""
    build_ewald_cache(domain::PeriodicDomain, kernel::QGKernel; n_fourier=16, n_images=2)

Ewald cache for the QG kernel in a periodic domain.

The periodic QG velocity is the periodic Euler velocity plus a smooth
QG-minus-Euler correction, so `fourier_coeffs` are the periodic Euler
coefficients and `corr_coeffs` the Fourier part of the Ewald-split correction
(see numerics/periodic_ewald.jl); `energy_coeffs` hold the Fourier part of the
periodic QG contour-energy potential. When the deformation radius is short
compared with the domain (`κ² > 16α²`, κ = 1/Ld), the velocity and energy sum
the QG kernel directly over periodic images instead and use only the Euler
tables.
"""
function build_ewald_cache(domain::PeriodicDomain{T}, kernel::QGKernel{T};
                           n_fourier::Int=16, n_images::Int=2) where {T}
    _validate_ewald_truncation(n_fourier, n_images)
    kappa2 = one(T) / kernel.Ld^2
    α, kx, ky, area = _ewald_wavenumbers(domain, n_fourier, n_images)
    nk = length(kx)
    fourier_coeffs = zeros(T, nk, nk)
    corr_coeffs = zeros(T, nk, nk)
    energy_coeffs = zeros(T, nk, nk)
    for (mi, kxi) in enumerate(kx)
        for (ni, kyi) in enumerate(ky)
            k2 = kxi^2 + kyi^2
            if !iszero(k2)
                fourier_coeffs[mi, ni] = _ewald_fourier_coefficient(EulerKernel(), k2, α, area)
                corr_coeffs[mi, ni] = _scaled_qg_correction_coefficient(
                    k2, kappa2, α, area, kappa2)
                energy_coeffs[mi, ni] = _ewald_energy_coefficient(kernel, k2, α, area)
            end
        end
    end
    return EwaldCache(α, kx, ky, fourier_coeffs, n_images, corr_coeffs, energy_coeffs)
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
# Generic floating-point types use a heterogeneous registry. Include the value
# precision in the key so a BigFloat cache built at one precision is never
# reused by a higher-precision calculation with numerically equal parameters.
const _ewald_caches_generic = Dict{Any,Any}()
const _ewald_key_order_generic = Any[]
const _ewald_cache_lock = ReentrantLock()
const _EWALD_CACHE_MAX = 64  # prevent unbounded growth
# Keys configured through `setup_ewald_cache!` are never evicted, so a custom
# truncation cannot silently revert to the defaults after many other domains
# or deformation radii have been cached.
const _ewald_pinned_f64 = Set{_EwaldCacheKey{Float64}}()
const _ewald_pinned_f32 = Set{_EwaldCacheKey{Float32}}()
const _ewald_pinned_generic = Set{Any}()

# Insert `cache` under `key`, evicting the oldest unpinned FIFO entries past the
# cap. Caller holds _ewald_cache_lock. Order is only touched for new keys, so
# overwriting an existing key leaves its FIFO position unchanged.
function _store_ewald_cache!(caches::AbstractDict, order::AbstractVector, key, cache,
                             pinned::AbstractSet; pin::Bool=false)
    if !haskey(caches, key)
        while length(caches) >= _EWALD_CACHE_MAX
            victim = findfirst(k -> !(k in pinned), order)
            victim === nothing && break
            delete!(caches, order[victim])
            deleteat!(order, victim)
        end
        push!(order, key)
    end
    pin && push!(pinned, key)
    caches[key] = cache
    return cache
end

function _cache_key(domain::PeriodicDomain{T}, ::EulerKernel) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), EulerKernel, zero(T))::_EwaldCacheKey{T}
end
function _cache_key(domain::PeriodicDomain{T}, k::QGKernel{T}) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), QGKernel{T}, _snap(k.Ld))::_EwaldCacheKey{T}
end
function _cache_key(domain::PeriodicDomain{T}, k::SQGKernel{T}) where {T}
    (_snap(domain.Lx), _snap(domain.Ly), SQGKernel{T}, _snap(k.δ))::_EwaldCacheKey{T}
end

_ewald_cache_dict(::Type{Float64}) = _ewald_caches_f64
_ewald_cache_dict(::Type{Float32}) = _ewald_caches_f32
_ewald_key_order(::Type{Float64}) = _ewald_key_order_f64
_ewald_key_order(::Type{Float32}) = _ewald_key_order_f32
_ewald_pinned(::Type{Float64}) = _ewald_pinned_f64
_ewald_pinned(::Type{Float32}) = _ewald_pinned_f32

@inline _kernel_value_precision(::EulerKernel) = 0
@inline _kernel_value_precision(kernel::QGKernel) = precision(kernel.Ld)
@inline _kernel_value_precision(kernel::SQGKernel) = precision(kernel.δ)
@inline function _generic_cache_key(domain::PeriodicDomain{T}, kernel) where {T}
    return (T, precision(domain.Lx), precision(domain.Ly),
            _kernel_value_precision(kernel), _cache_key(domain, kernel))
end

function _get_ewald_cache(domain::PeriodicDomain{T}, kernel::AbstractKernel) where {T<:Union{Float64, Float32}}
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    # Fast path: quick read under lock to check if cache already exists.
    # After warm-up, this is the only lock acquisition needed per call. Use the
    # `@lock` macro (lock/try/finally, no closure) rather than `lock(f) do ... end`,
    # which would box `caches`/`key` into a closure (~48 B per velocity! call).
    cached = Base.@lock _ewald_cache_lock get(caches, key, nothing)
    cached !== nothing && return cached

    # Slow path: build outside the lock, then store under the lock.
    new_cache = build_ewald_cache(domain, kernel)

    lock(_ewald_cache_lock) do
        # Double-check: another thread may have built it while we were computing.
        existing = get(caches, key, nothing)
        existing !== nothing && return existing
        return _store_ewald_cache!(caches, order, key, new_cache, _ewald_pinned(T))
    end
end

# Generic float types use the same bounded, locked cache policy. A type assertion
# at the lookup boundary restores the concrete cache type after reading from the
# heterogeneous registry.
function _get_ewald_cache(domain::PeriodicDomain{T},
                          kernel::AbstractKernel) where {T<:AbstractFloat}
    key = _generic_cache_key(domain, kernel)
    cached = Base.@lock _ewald_cache_lock get(_ewald_caches_generic, key, nothing)
    cached !== nothing && return cached::EwaldCache{T}

    new_cache = build_ewald_cache(domain, kernel)
    lock(_ewald_cache_lock) do
        existing = get(_ewald_caches_generic, key, nothing)
        existing !== nothing && return existing::EwaldCache{T}
        return _store_ewald_cache!(_ewald_caches_generic,
                                   _ewald_key_order_generic, key, new_cache,
                                   _ewald_pinned_generic)
    end
end

# Pre-fetch Ewald cache for use in threaded velocity computation.
# Returns `nothing` for unbounded domains (no cache needed).
_prefetch_ewald(::UnboundedDomain, ::AbstractKernel) = nothing
_prefetch_ewald(domain::PeriodicDomain,
                kernel::Union{EulerKernel, QGKernel, SQGKernel}) =
    _get_ewald_cache(domain, kernel)

# Store and pin `cache` for (domain, kernel) in the precision-appropriate registry.
function _store_ewald!(domain::PeriodicDomain{T}, kernel::AbstractKernel,
                       cache) where {T<:Union{Float64, Float32}}
    key = _cache_key(domain, kernel)
    caches = _ewald_cache_dict(T)
    order = _ewald_key_order(T)
    lock(_ewald_cache_lock) do
        _store_ewald_cache!(caches, order, key, cache, _ewald_pinned(T); pin=true)
    end
    return nothing
end

function _store_ewald!(domain::PeriodicDomain{T}, kernel::AbstractKernel,
                       cache) where {T<:AbstractFloat}
    key = _generic_cache_key(domain, kernel)
    lock(_ewald_cache_lock) do
        _store_ewald_cache!(_ewald_caches_generic,
                            _ewald_key_order_generic, key, cache,
                            _ewald_pinned_generic; pin=true)
    end
    return nothing
end

"""
    setup_ewald_cache!(domain, kernel; n_fourier=16, n_images=2)

Pre-build and store an Ewald cache with custom parameters.  Call this before
`evolve!` to override the default `n_fourier=16`, `n_images=2`.  The cached
result is used automatically by all subsequent velocity computations on
the same domain/kernel combination.

For `QGKernel` the stored cache carries both the Euler periodic coefficients
and the QG correction coefficients (see [`build_ewald_cache`](@ref)), and an
Euler-keyed cache is also stored so pure-Euler evaluations on the same domain
share the warmed `n_fourier`/`n_images` setup. For `MultiLayerQGKernel` the
caches of every vertical mode are configured: the Euler cache for the
barotropic mode and a QG cache per baroclinic deformation radius.

Configured caches are kept until [`clear_ewald_cache!`](@ref); they are exempt
from the eviction that bounds the number of automatically built caches.
"""
function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::AbstractKernel;
                            n_fourier::Int=16,
                            n_images::Int=2) where {T<:AbstractFloat}
    _validate_ewald_truncation(n_fourier, n_images)
    _store_ewald!(domain, kernel,
        build_ewald_cache(domain, kernel; n_fourier=n_fourier, n_images=n_images))
    kernel isa QGKernel &&
        setup_ewald_cache!(domain, EulerKernel(); n_fourier=n_fourier, n_images=n_images)
    return nothing
end

function setup_ewald_cache!(domain::PeriodicDomain{T}, kernel::MultiLayerQGKernel;
                            n_fourier::Int=16,
                            n_images::Int=2) where {T<:AbstractFloat}
    _validate_ewald_truncation(n_fourier, n_images)
    for λ in kernel.eigenvalues
        _dispatch_qg_mode(kernel, T(λ)) do mode_kernel
            setup_ewald_cache!(domain, mode_kernel; n_fourier=n_fourier, n_images=n_images)
        end
    end
    return nothing
end

"""Clear all cached Ewald data."""
function clear_ewald_cache!()
    lock(_ewald_cache_lock) do
        empty!(_ewald_caches_f64)
        empty!(_ewald_caches_f32)
        empty!(_ewald_key_order_f64)
        empty!(_ewald_key_order_f32)
        empty!(_ewald_caches_generic)
        empty!(_ewald_key_order_generic)
        empty!(_ewald_pinned_f64)
        empty!(_ewald_pinned_f32)
        empty!(_ewald_pinned_generic)
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
        # Continued fraction e^{-x}/(x+1- 1²/(x+3- 2²/(x+5- ...))) by the
        # modified Lentz method, stopped once converged: about 50 terms at
        # x = 2 and fewer than 10 beyond x ≈ 30. For very large x, E₁(x)
        # underflows to zero.
        ex = exp(-x)
        ex == zero(T) && return zero(T)
        tiny = floatmin(T) / eps(T)
        b = x + one(T)
        c = one(T) / tiny
        d = one(T) / b
        h = d
        for i in 1:500
            a = -T(i)^2
            b += T(2)
            d = one(T) / (a * d + b)
            c = b + a / c
            del = c * d
            h *= del
            abs(del - one(T)) <= eps(T) && break
        end
        return h * ex
    end
end
