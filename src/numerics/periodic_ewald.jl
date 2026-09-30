# Scalar periodic Ewald kernels shared by the CPU and KA paths.
#
# Notation: A = 4LxLy is the cell area, κ = 1/Ld, s0 = 1/(4α²) is the Ewald
# splitting scale and x = κ²s0. With the unbounded kernels normalized as
# -(1/2π)log r (Euler) and K₀(κr)/(2π) (QG), the zero-mean periodic Green's
# functions are G̃_E = (1/A) Σ_{k≠0} e^{ik·r}/k² and
# G̃_QG = (1/A) Σ_{k≠0} e^{ik·r}/(k²+κ²). Their difference is κ²Ĝ, where the
# scaled correction
#
#     Ĝ(r) = -(1/A) Σ_{k≠0} e^{ik·r} / (k²(k²+κ²))
#
# is also, times 4π, the periodic QG contour-energy potential (and, at κ = 0,
# the periodic Euler one). Splitting
# 1/(k²(k²+κ²)) = ∫₀^∞ s φ₁(κ²s) e^{-k²s} ds at s0 gives a Gaussian-damped
# real-space image sum plus a Gaussian-damped Fourier sum:
#
#     Ĝ(r) = (s0/4π) Σ_R P(|r+R|²/(4s0), x) + Σ_{k≠0} ĉ_k cos(k·r) + s0²ψ₂(x)/A,
#     ĉ_k  = -e^{-k²s0} (1 + k²s0 φ₁(x)) / (A k²(k²+κ²)),
#
# so both sums converge at the Euler truncation. When κ is large compared with
# α the real-space kernel P loses accuracy, but K₀ then decays within a few
# periods, and the QG kernel is summed directly over images instead.

# Largest x = κ²/(4α²) for the Ewald split of the QG correction; beyond it the
# QG kernel is summed directly over images (see `_qg_uses_direct_images`).
const _QG_EWALD_X_MAX = 4

# Real-space Ewald terms bounded by E₁(u) ≤ e^{-u}/u are below rounding once
# u exceeds -log(eps) + 5; such images are skipped.
@inline _ewald_real_cutoff(::Type{T}) where {T} = -log(eps(T)) + 5

# Indices p of the images r - 2Lp that can lie within `reach` of the origin
# (along one axis), clipped to the configured block -n_images:n_images.
@inline function _ewald_image_range(r::T, L::T, reach::T, n_images::Int) where {T}
    lo = clamp((r - reach) / (2 * L), T(-n_images), T(n_images + 1))
    hi = clamp((r + reach) / (2 * L), T(-n_images - 1), T(n_images))
    return ceil(Int, lo):floor(Int, hi)
end

# Fourier part Σ_{k≠0} c_k cos(k·r) of every periodic Ewald kernel. The
# coefficients depend on |k| only, so they are even in kx and in ky, the
# sin(kx·x)sin(ky·y) halves of cos(k·r) cancel, and the sum folds onto the
# modes m, n ≥ 0 (the `EwaldCache` cosine tables, weights included):
#
#     Σ_{m,n≥0} w[m+1, n+1] cos(mθx) cos(nθy),   θ = (dkx·rx, dky·ry).
#
# cos(mθ) follows from the Chebyshev recurrence
# cos((m+1)θ) = 2cosθ·cos(mθ) - cos((m-1)θ), so a point costs two cosines
# rather than two trigonometric calls per mode.
@inline function _ewald_cosine_sum(w, dkx::T, dky::T, rx::T, ry::T) where {T}
    isempty(w) && return zero(T)
    cθx = cos(dkx * rx)
    cθy = cos(dky * ry)
    total = zero(T)
    cx, cx_prev = one(T), cθx
    @inbounds for m in 1:size(w, 1)
        row = zero(T)
        cy, cy_prev = one(T), cθy
        for n in 1:size(w, 2)
            row += w[m, n] * cy
            cy, cy_prev = 2 * cθy * cy - cy_prev, cy
        end
        total += cx * row
        cx, cx_prev = 2 * cθx * cx - cx_prev, cx
    end
    return total
end

@inline _ewald_qg_x(kappa2::T, α::T) where {T} = kappa2 / (4 * α * α)
@inline _qg_uses_direct_images(kappa2::T, α::T) where {T} = _ewald_qg_x(kappa2, α) > _QG_EWALD_X_MAX

# φ₁(x) = (1 - e^{-x})/x and ψ₂(x) = (x - 1 + e^{-x})/x², both evaluated
# without cancellation near x = 0.
@inline _ewald_phi1(x::T) where {T} = iszero(x) ? one(T) : -expm1(-x) / x

@inline function _ewald_psi2(x::T) where {T}
    if x < T(0.1)
        # Σ_{n≥0} (-x)^n/(n+2)!
        s = zero(T)
        term = one(T) / 2
        for n in 0:20
            s += term
            abs(term) <= eps(T) * abs(s) && break
            term *= -x / T(n + 3)
        end
        return s
    end
    return (x + expm1(-x)) / (x * x)
end

# Ein(z) = E₁(z) + log(z) + γ, the entire regular part of E₁.
@inline function _ewald_ein(z::T) where {T}
    if z < T(2)
        s = zero(T)
        term = one(T)
        for n in 1:80
            term *= -z / T(n)
            s -= term / T(n)
            abs(term) <= eps(T) * abs(s) && break
        end
        return s
    end
    return _expint_e1(z) + log(z) + T(Base.MathConstants.eulergamma)
end

# Real-space kernel of the scaled correction per unit s0:
#     P(u, x) = Σ_{n≥1} (-1)ⁿ x^{n-1}/n! E_{n+1}(u),   u = ρ²/(4s0).
# At x = 0 this is -E₂(u), the periodic Euler energy kernel. The forward
# recurrence E_{n+1} = (e^{-u} - u Eₙ)/n loses at most a factor of about
# e^x/x relative to eps for x ≤ _QG_EWALD_X_MAX.
@inline function _ewald_qg_real_kernel(u::T, x::T) where {T}
    s = zero(T)
    a = -one(T)                  # (-1)ⁿ x^{n-1}/n! at n = 1
    if iszero(u)
        for n in 1:120           # E_{n+1}(0) = 1/n
            t = a / T(n)
            s += t
            n >= x && abs(t) <= eps(T) * abs(s) && break
            a *= -x / T(n + 1)
        end
        return s
    end
    u > _ewald_real_cutoff(T) && return zero(T)   # |P| ≤ E₁(u)
    emu = exp(-u)
    E = _expint_e1(u)
    for n in 1:120
        E = (emu - u * E) / T(n)
        t = a * E
        s += t
        n >= x && abs(t) <= eps(T) * abs(s) && break
        a *= -x / T(n + 1)
    end
    return s
end

# Ewald evaluation of `scale · Ĝ(r)` given a cosine table of Fourier
# coefficients already multiplied by `scale` (scale · ĉ_k). The velocity uses
# scale = κ² and the energy potential scale = 4π.
@inline function _scaled_qg_correction_scalar(rx::T, ry::T, scale::T, α::T, x::T,
                                              Lx::T, Ly::T, n_images::Int,
                                              dkx::T, dky::T, table) where {T}
    s0 = one(T) / (4 * α * α)
    real_sum = zero(T)
    reach = sqrt(_ewald_real_cutoff(T)) / α   # u = α²ρ² ≤ cutoff
    for px in _ewald_image_range(rx, Lx, reach, n_images)
        sx = rx - 2 * Lx * T(px)
        for py in _ewald_image_range(ry, Ly, reach, n_images)
            sy = ry - 2 * Ly * T(py)
            real_sum += _ewald_qg_real_kernel((sx * sx + sy * sy) / (4 * s0), x)
        end
    end

    fourier = _ewald_cosine_sum(table, dkx, dky, rx, ry)
    area = 4 * Lx * Ly
    return scale * (s0 / (4 * T(π)) * real_sum + s0 * s0 * _ewald_psi2(x) / area) + fourier
end

# Fourier coefficient `scale · ĉ_k` of the scaled correction.
@inline function _scaled_qg_correction_coefficient(k2::T, kappa2::T, α::T,
                                                   area::T, scale::T) where {T}
    s0 = one(T) / (4 * α * α)
    x = kappa2 * s0
    return -scale * exp(-k2 * s0) * (1 + k2 * s0 * _ewald_phi1(x)) /
           (area * k2 * (k2 + kappa2))
end

# Number of periodic images per direction for the direct QG image sum: K₀(κd)
# is below eps once κd exceeds -log(eps). The nearest omitted image is at least
# (2n + 1)L away from a target in the central cell.
@inline function _qg_direct_image_count(kappa::T, L::T) where {T}
    decay = -log(eps(T))
    return max(0, ceil(Int, (decay / (kappa * L) - 1) / 2))
end

# Scaled correction Ĝ(r) for the direct-image regime, from the QG image sum
# and the periodic Euler Ewald correction (`euler_corr` = G̃_E - G̃_E,unbounded
# at the minimum image `r`). The logarithms of the central images cancel.
@inline function _direct_scaled_qg_correction_scalar(rx::T, ry::T, kappa2::T, Ld::T,
                                                     Lx::T, Ly::T, euler_corr::T) where {T}
    kappa = sqrt(kappa2)
    inv2pi = one(T) / (2 * T(π))
    nx = _qg_direct_image_count(kappa, Lx)
    ny = _qg_direct_image_count(kappa, Ly)
    r2 = rx * rx + ry * ry
    center = if r2 < eps(T)^2
        log(2 * Ld) - T(Base.MathConstants.eulergamma)
    else
        r = sqrt(r2)
        _qg_smooth_correction_scalar(r / Ld, r, Ld)
    end
    images = zero(T)
    for px in -nx:nx, py in -ny:ny
        (px == 0 && py == 0) && continue
        sx = rx - 2 * Lx * T(px)
        sy = ry - 2 * Ly * T(py)
        images += _besselk0_scalar(sqrt(sx * sx + sy * sy) / Ld)
    end
    area = 4 * Lx * Ly
    return (inv2pi * (center + images) - 1 / (area * kappa2) - euler_corr) / kappa2
end

# Neutralized real-space potential for the periodic SQG energy, per image.
# Laplacian: erfc(αρ)/ρ + (1/r_δ - 1/ρ), each with its net charge removed by a
# Gaussian of width 1/α that is carried in Fourier space instead, and with the
# -δ²/(2ρ) tail of the softening part moved to Fourier space through
# erf(αρ)/ρ. What remains decays like a Gaussian plus δ⁴/(24ρ³), so a finite
# block of images around the minimum image converges and is periodic.
@inline function _sqg_neutral_real_potential(ρ::T, α::T, δ::T) where {T}
    ar = α * ρ
    if ar * ar > _ewald_real_cutoff(T)
        # Past the cutoff the Gaussian-damped terms are below rounding and the
        # softening reduces to δ²/(r_δ+ρ) + δ²/(2ρ) - δ·asinh(δ/ρ) ≈ δ⁴/(24ρ³).
        r_δ = sqrt(ρ * ρ + δ * δ)
        return δ * δ / (r_δ + ρ) + δ * δ / (2 * ρ) - δ * asinh(δ / ρ)
    end
    inv_α_sqrtpi = one(T) / (α * sqrt(T(π)))
    unsoftened = ρ * erfc(ar) - exp(-ar * ar) * inv_α_sqrtpi
    iszero(δ) && return unsoftened
    r_δ = sqrt(ρ * ρ + δ * δ)
    γ = T(Base.MathConstants.eulergamma)
    softening = δ * δ / (r_δ + ρ) - δ * log(δ + r_δ) +
                δ / 2 * (_ewald_ein(ar * ar) - γ) - δ * log(α) +
                δ * δ / 2 * _sqg_erf_over_r(α, ρ * ρ)
    return unsoftened + softening
end

# Fourier coefficient of the periodic SQG energy potential (per the shared
# energy normalization, which doubles the SQG potential) at k ≠ 0.
@inline function _sqg_energy_fourier_coefficient(k2::T, α::T, δ::T, area::T) where {T}
    k = sqrt(k2)
    y = k / (2 * α)
    gauss = exp(-y * y)
    h = erfc(y) + 2 * y / sqrt(T(π)) * gauss
    return 2 * (2 * T(π) / area) * (-h / (k2 * k) + δ * gauss / k2 - δ * δ / 2 * erfc(y) / k)
end

# Constant making the SQG energy potential zero-mean: minus the cell average
# of the real-space image sum.
@inline function _sqg_energy_zero_mean_constant(α::T, δ::T, area::T) where {T}
    sqrtpi = sqrt(T(π))
    real_integral = -sqrtpi / (3 * α^3) + T(π) * δ^3 / 3 +
                    T(π) * δ / (2 * α^2) - sqrtpi * δ^2 / α
    return -2 * real_integral / area
end

# Periodic SQG contour-energy potential at separation (dx, dy), wrapped to the
# minimum image; valid because the neutralized real-space terms decay.
@inline function _sqg_periodic_energy_potential_scalar(dx::T, dy::T, α::T,
                                                       Lx::T, Ly::T, δ::T,
                                                       n_images::Int, dkx::T, dky::T,
                                                       energy_table) where {T}
    Lx2 = 2 * Lx
    Ly2 = 2 * Ly
    rx = dx - round(dx / Lx2) * Lx2
    ry = dy - round(dy / Ly2) * Ly2
    phi = zero(T)
    for px in -n_images:n_images
        sx = rx - 2 * Lx * T(px)
        for py in -n_images:n_images
            sy = ry - 2 * Ly * T(py)
            phi += 2 * _sqg_neutral_real_potential(sqrt(sx * sx + sy * sy), α, δ)
        end
    end
    phi += _ewald_cosine_sum(energy_table, dkx, dky, rx, ry)
    return phi + _sqg_energy_zero_mean_constant(α, δ, 4 * Lx * Ly)
end

# Periodic Euler (κ² = 0) or QG contour-energy potential 4πĜ at separation
# (dx, dy), wrapped to the minimum image so the finite image block is centred.
# The QG direct-image regime needs the Euler velocity table for the periodic
# Euler correction; the Ewald regime uses the cache's energy table.
@inline function _periodic_energy_potential_scalar(dx::T, dy::T, kappa2::T, α::T,
                                                   Lx::T, Ly::T, n_images::Int,
                                                   dkx::T, dky::T, fourier_table,
                                                   energy_table) where {T}
    Lx2 = 2 * Lx
    Ly2 = 2 * Ly
    rx = dx - round(dx / Lx2) * Lx2
    ry = dy - round(dy / Ly2) * Ly2
    fourpi = 4 * T(π)
    if _qg_uses_direct_images(kappa2, α)
        euler_corr = _periodic_euler_green_correction_scalar(
            rx, ry, zero(T), zero(T), α, Lx, Ly, n_images, dkx, dky, fourier_table,
            one(T) / fourpi, T(Base.MathConstants.eulergamma))
        return fourpi * _direct_scaled_qg_correction_scalar(
            rx, ry, kappa2, one(T) / sqrt(kappa2), Lx, Ly, euler_corr)
    end
    return _scaled_qg_correction_scalar(rx, ry, fourpi, α, _ewald_qg_x(kappa2, α),
                                        Lx, Ly, n_images, dkx, dky, energy_table)
end
