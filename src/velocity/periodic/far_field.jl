# Structure-factor far field for bulk periodic velocity evaluation.
#
# The Fourier part of every periodic Ewald kernel is linear in the sources:
#
#     Σ_j pv_j ∫_j Σ_k c_k cos(k·(x - s)) t ds = Σ_k c_k Re[e^{ik·x} S(k)],
#     S(k) = Σ_j pv_j ∫_j e^{-ik·s} t ds,
#
# so one pass over the segments builds S and each target then needs one sum
# over the modes: O(N·modes) in all, where the pairwise evaluation costs
# O(N²·modes). The pair loop keeps only the real-space part, through a cache
# with the same α and image block but no Fourier modes. S uses the pairwise
# correction's 5-point Gauss–Legendre points, and c_{-k} = c_k with
# S(-k) = conj S(k) restrict the sum to the half plane ky > 0 or ky = 0 < kx.

mutable struct _EwaldFarField{T}
    cache::EwaldCache{T}          # full cache the coefficients come from
    near::EwaldCache{T}           # same α and image block, no Fourier modes
    coeff::Matrix{T}              # 2c_k on the half plane, (2K+1) × (K+1)
    sx::Matrix{Complex{T}}        # S(k), x and y components
    sy::Matrix{Complex{T}}
    ex::Vector{Complex{T}}        # e^{-imθx}, m = -K:K, for one source point
    ey::Vector{Complex{T}}        # e^{-inθy}, n = 0:K
end

# Fourier coefficient of the bulk velocity at full-table index (i, j): the
# Euler table, plus the QG-minus-Euler correction for QG, or the SQG table
# with its 1/(2π) velocity normalization.
@inline _far_field_coefficient(::EulerKernel, cache::EwaldCache, i, j) =
    cache.fourier_coeffs[i, j]
@inline _far_field_coefficient(::QGKernel, cache::EwaldCache, i, j) =
    cache.fourier_coeffs[i, j] + cache.corr_coeffs[i, j]
@inline _far_field_coefficient(::SQGKernel{T}, cache::EwaldCache{T}, i, j) where {T} =
    cache.fourier_coeffs[i, j] / (2 * T(π))

@inline _uses_far_field(::AbstractKernel, cache::EwaldCache) = true
@inline _uses_far_field(kernel::QGKernel, cache::EwaldCache) =
    !_qg_uses_direct_images(inv(kernel.Ld^2), cache.α)

# 2c_k over the half-plane mode grid m = -K:K, n = 0:K (zero for n = 0, m ≤ 0).
function _far_field_coefficients(cache::EwaldCache{T}, kernel::AbstractKernel) where {T}
    kernel isa QGKernel && _required_ewald_table(cache, cache.corr_cos, :corr_coeffs)
    K = length(cache.kx) ÷ 2
    coeff = zeros(T, 2K + 1, K + 1)
    for n in 0:K, m in -K:K
        (n > 0 || m > 0) || continue
        coeff[K + 1 + m, n + 1] = 2 * _far_field_coefficient(kernel, cache, K + 1 + m, K + 1 + n)
    end
    return coeff
end

# Cache with the α and image block of `cache` but no Fourier modes: its pair
# evaluations are the real-space part only.
_real_space_cache(cache::EwaldCache{T}) where {T} =
    EwaldCache(cache.α, T[], T[], zeros(T, 0, 0), cache.n_images, zeros(T, 0, 0), zeros(T, 0, 0))

function _EwaldFarField(cache::EwaldCache{T}, kernel::AbstractKernel) where {T}
    K = length(cache.kx) ÷ 2
    return _EwaldFarField{T}(cache, _real_space_cache(cache),
                             _far_field_coefficients(cache, kernel),
                             zeros(Complex{T}, 2K + 1, K + 1),
                             zeros(Complex{T}, 2K + 1, K + 1),
                             Vector{Complex{T}}(undef, 2K + 1),
                             Vector{Complex{T}}(undef, K + 1))
end

# Far fields kept per scratch; one per cache in use (a multi-layer problem
# alternates between its modes' caches).
const _MAX_FAR_FIELDS = 8

# The far field for `kernel` with the prefetched `ewald` cache, its structure
# factor cleared, or `nothing` when the pair loop needs no split (unbounded
# domains, and QG summed directly over images).
_prepare_far_field!(scratch, ::AbstractKernel, ::Nothing) = nothing

function _prepare_far_field!(scratch::_VelocityScratch{T}, kernel::AbstractKernel,
                             cache::EwaldCache{T}) where {T}
    _uses_far_field(kernel, cache) || return nothing
    fields = scratch.ewald_far
    if !(fields isa Vector{_EwaldFarField{T}})
        fields = _EwaldFarField{T}[]
        scratch.ewald_far = fields
    end
    index = 0
    for (i, f) in pairs(fields)
        f.cache === cache && (index = i; break)
    end
    if index == 0
        length(fields) >= _MAX_FAR_FIELDS && popfirst!(fields)
        push!(fields, _EwaldFarField(cache, kernel))
        index = length(fields)
    end
    far = fields[index]
    fill!(far.sx, zero(Complex{T}))
    fill!(far.sy, zero(Complex{T}))
    return far
end

# Cache for the pair loop: the full cache without a far field, else its
# real-space-only twin.
@inline _near_field_ewald(ewald, ::Nothing) = ewald
@inline _near_field_ewald(ewald, far::_EwaldFarField) = far.near

# Add `weight · pv`-weighted segments of `contours` to the structure factor,
# at the Gauss–Legendre points of the straight or cubic segment velocity.
function _accumulate_structure_factor!(far::_EwaldFarField{T},
                                       contours::Vector{PVContour{T}},
                                       curvatures, weight::T) where {T}
    K = length(far.ey) - 1
    dkx, dky = far.cache.dkx, far.cache.dky
    g_nodes, g_weights = _gl5_nodes_weights(T)
    for (ci, c) in pairs(contours)
        nc = nnodes(c)
        nc < 2 && continue
        pv = weight * c.pv
        κ = curvatures[ci]
        @inbounds for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            dsx = b[1] - a[1]
            dsy = b[2] - a[2]
            ds_len = sqrt(dsx^2 + dsy^2)
            ds_len < eps(T) && continue
            κa = κ[j]
            κb = κ[mod1(j + 1, nc)]
            curved = max(abs(κa), abs(κb)) * ds_len > sqrt(eps(T))
            for q in 1:5
                p = (one(T) + g_nodes[q]) / 2
                sx, sy, tx, ty = curved ?
                    _cubic_point_tangent_scalar(a[1], a[2], b[1], b[2], κa, κb, p) :
                    (a[1] + p * dsx, a[2] + p * dsy, dsx, dsy)
                w = pv * g_weights[q] / 2
                _add_structure_factor_point!(far.sx, far.sy, far.ex, far.ey, K,
                                             dkx * sx, dky * sy, w * tx, w * ty)
            end
        end
    end
    return far
end

# S(k) += (wx, wy) e^{-i(mθx + nθy)} over the (2K+1) × (K+1) mode grid, with
# the phases of both axes built by complex rotation in the scratch `ex`, `ey`.
@inline function _add_structure_factor_point!(sx::Matrix{Complex{T}}, sy::Matrix{Complex{T}},
                                              ex::Vector{Complex{T}}, ey::Vector{Complex{T}},
                                              K::Int, θx::T, θy::T, wx::T, wy::T) where {T}
    rx = cis(-θx)
    ry = cis(-θy)
    @inbounds begin
        ex[K + 1] = one(Complex{T})
        for m in 1:K
            ex[K + 1 + m] = ex[K + m] * rx
            ex[K + 1 - m] = conj(ex[K + 1 + m])
        end
        ey[1] = one(Complex{T})
        for n in 1:K
            ey[n + 1] = ey[n] * ry
        end
        for n in 1:(K + 1)
            wxn = wx * ey[n]
            wyn = wy * ey[n]
            for m in 1:(2K + 1)
                sx[m, n] += ex[m] * wxn
                sy[m, n] += ex[m] * wyn
            end
        end
    end
    return nothing
end

# Σ_k 2c_k Re[e^{ik·x} S(k)] over the half plane: the Fourier part of the
# velocity at `x` from every accumulated source. Reads `far` only, so targets
# can be evaluated concurrently.
@inline _far_field_velocity(::Nothing, x::SVector{2,T}) where {T} = zero(SVector{2,T})

@inline function _far_field_velocity(far::_EwaldFarField{T}, x::SVector{2,T}) where {T}
    K = length(far.ey) - 1
    coeff, sx, sy = far.coeff, far.sx, far.sy
    rx = cis(far.cache.dkx * x[1])
    ry = cis(far.cache.dky * x[2])
    gx_first = cis(-K * far.cache.dkx * x[1])
    gy = one(Complex{T})
    vx = zero(T)
    vy = zero(T)
    @inbounds for n in 1:(K + 1)
        row_x = zero(Complex{T})
        row_y = zero(Complex{T})
        gx = gx_first
        for m in 1:(2K + 1)
            cg = coeff[m, n] * gx
            row_x += cg * sx[m, n]
            row_y += cg * sy[m, n]
            gx *= rx
        end
        vx += real(gy * row_x)
        vy += real(gy * row_y)
        gy *= ry
    end
    return SVector{2,T}(vx, vy)
end

# ── Energy: Parseval ─────────────────────────────────────────────────────
#
# The Fourier part of a periodic contour-energy double sum is a sum over modes,
#
#     Σ_{i,j} q_i q_j ∫∫ Σ_{k≠0} c_k cos(k·(x - x')) dx·dx' = Σ_{k≠0} c_k |T(k)|²,
#     T(k) = Σ_j q_j ∫_j e^{-ik·s} ds,
#
# with T built from the energy's straight-segment 3-point Gauss–Legendre rule,
# so it equals the pairwise sum it replaces; the pair loop keeps the real-space
# part through a cache without Fourier modes.

# Energy-potential Fourier coefficient at full-table index (i, j): the energy
# table, or, for QG summed directly over images, the periodic Euler table the
# potential subtracts, scaled by -4π/κ².
@inline _energy_far_coefficient(::Union{EulerKernel, SQGKernel}, cache::EwaldCache, i, j) =
    cache.energy_coeffs[i, j]
@inline function _energy_far_coefficient(kernel::QGKernel{T}, cache::EwaldCache{T},
                                         i, j) where {T}
    kappa2 = inv(kernel.Ld^2)
    _qg_uses_direct_images(kappa2, cache.α) || return cache.energy_coeffs[i, j]
    return -4 * T(π) / kappa2 * cache.fourier_coeffs[i, j]
end

# 2c_k of the energy potential over the half-plane mode grid.
function _energy_far_coefficients(cache::EwaldCache{T}, kernel::AbstractKernel) where {T}
    _required_ewald_table(cache, cache.energy_cos, :energy_coeffs)
    K = length(cache.kx) ÷ 2
    coeff = zeros(T, 2K + 1, K + 1)
    for n in 0:K, m in -K:K
        (n > 0 || m > 0) || continue
        coeff[K + 1 + m, n + 1] = 2 * _energy_far_coefficient(kernel, cache, K + 1 + m, K + 1 + n)
    end
    return coeff
end

# Σ_k 2c_k |T(k)|² over the half plane for the valid closed contours of each
# `(contours, weight)` group, their PV jumps scaled by `weight`.
function _parseval_energy(coeff::Matrix{T}, cache::EwaldCache{T}, groups) where {T}
    K = size(coeff, 2) - 1
    tx = zeros(Complex{T}, 2K + 1, K + 1)
    ty = zeros(Complex{T}, 2K + 1, K + 1)
    ex = Vector{Complex{T}}(undef, 2K + 1)
    ey = Vector{Complex{T}}(undef, K + 1)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    for (contours, weight) in groups
        for c in contours
            _valid_energy_contour(c) || continue
            q = weight * c.pv
            for j in 1:nnodes(c)
                seg = _contour_energy_segment(c, j)
                for p in 1:3
                    s = seg.mid + g_nodes[p] * seg.half_ds
                    w = q * g_weights[p] / 2
                    _add_structure_factor_point!(tx, ty, ex, ey, K,
                                                 cache.dkx * s[1], cache.dky * s[2],
                                                 w * seg.ds[1], w * seg.ds[2])
                end
            end
        end
    end
    energy = zero(T)
    @inbounds for i in eachindex(coeff)
        energy += coeff[i] * (abs2(tx[i]) + abs2(ty[i]))
    end
    return energy
end
