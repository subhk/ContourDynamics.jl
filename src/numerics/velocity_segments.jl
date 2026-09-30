# Pure per-segment formulas shared by CPU adapters and KA kernels.
# These functions accept scalars and return tuples; they allocate no arrays.

@inline function _straight_euler_contribution_scalar(xi::T, yi::T,
                                                     ax::T, ay::T, bx::T, by::T,
                                                     pv::T, inv4pi::T) where {T}
    # Rotate into segment-local tangent/normal coordinates and evaluate the
    # analytic antiderivative of log(r^2) along the straight segment.
    dsx = bx - ax
    dsy = by - ay
    ds_len_sq = dsx^2 + dsy^2
    ds_len = sqrt(ds_len_sq)
    ds_len < eps(T) && return zero(T), zero(T)

    tx = dsx / ds_len
    ty = dsy / ds_len
    nx = -ty
    ny = tx

    r0x = xi - ax
    r0y = yi - ay
    u_a = r0x * tx + r0y * ty
    h = r0x * nx + r0y * ny
    integral = _euler_segment_log_integral(u_a, h, ds_len)
    contrib = -inv4pi * pv * integral
    return contrib * tx, contrib * ty
end

@inline function _curved_euler_contribution_scalar(xi::T, yi::T,
                                                   ax::T, ay::T, bx::T, by::T,
                                                   pv::T, κa::T, κb::T,
                                                   inv4pi::T) where {T}
    # Curved segments use Dritschel's cubic normal displacement. The straight
    # analytic path is retained for nearly flat segments to avoid unnecessary
    # quadrature and roundoff.
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    ds_len < eps(T) && return zero(T), zero(T)

    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return _straight_euler_contribution_scalar(xi, yi, ax, ay, bx, by, pv, inv4pi)

    vx, vy = _straight_euler_contribution_scalar(
        xi, yi, ax, ay, bx, by, pv, inv4pi)
    g_nodes, g_weights = _gl5_nodes_weights(T)

    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        sx, sy, tangent_x, tangent_y = _cubic_point_tangent_scalar(
            ax, ay, bx, by, κa, κb, p)
        line_x = ax + p * dsx
        line_y = ay + p * dsy
        rx_curve = xi - sx
        ry_curve = yi - sy
        rx_line = xi - line_x
        ry_line = yi - line_y
        log_curve = log(max(rx_curve * rx_curve + ry_curve * ry_curve, eps(T)^2))
        log_line = log(max(rx_line * rx_line + ry_line * ry_line, eps(T)^2))
        coeff = -inv4pi * pv * (g_weights[q] / T(2))
        vx += coeff * (log_curve * tangent_x - log_line * dsx)
        vy += coeff * (log_curve * tangent_y - log_line * dsy)
    end

    return vx, vy
end

@inline function _qg_smooth_correction_scalar(rr::T, r::T, Ld::T) where {T}
    # QG = Euler logarithmic kernel plus a smooth finite deformation-radius
    # correction K₀(r/Ld) + log r, summed as a series without cancellation
    # where K₀ is logarithmic.
    rr <= 2 && return _besselk0_correction(rr) + log(2 * Ld) - T(Base.MathConstants.eulergamma)
    return _besselk0_scalar(rr) + log(r)
end

# K₀(rr) and the smooth QG correction K₀(rr) + log r (rr = r/Ld), sharing one
# evaluation.
@inline function _qg_k0_and_smooth(rr::T, r::T, Ld::T) where {T}
    if rr <= 2
        c = _besselk0_correction(rr)
        γ = T(Base.MathConstants.eulergamma)
        return c - log(rr / 2) - γ, c + log(2 * Ld) - γ
    end
    k0 = _besselk0_scalar(rr)
    return k0, k0 + log(r)
end

@inline function _curved_qg_contribution_scalar(xi::T, yi::T,
                                                ax::T, ay::T, bx::T, by::T,
                                                pv::T, κa::T, κb::T,
                                                Ld::T, inv2pi::T, inv4pi::T) where {T}
    # Reuse the Euler contribution and add only the QG smooth correction. This
    # keeps singular handling identical between Euler and QG velocity paths.
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    ds_len < eps(T) && return zero(T), zero(T)

    if max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T))
        r0x = xi - ax
        r0y = yi - ay
        p_near = clamp((r0x * dsx + r0y * dsy) / (ds_len * ds_len),
                       zero(T), one(T))
        near_x = r0x - p_near * dsx
        near_y = r0y - p_near * dsy
        min_r = sqrt(near_x * near_x + near_y * near_y)
        if min_r > T(4) * max(Ld, ds_len)
            g_nodes, g_weights = _gl5_nodes_weights(T)
            half_dsx = dsx / T(2)
            half_dsy = dsy / T(2)
            direct = zero(T)
            @inbounds for q in 1:5
                sx = (ax + bx) / T(2) + g_nodes[q] * half_dsx
                sy = (ay + by) / T(2) + g_nodes[q] * half_dsy
                rx = sx - xi
                ry = sy - yi
                direct += g_weights[q] *
                          _besselk0_scalar(sqrt(rx * rx + ry * ry) / Ld)
            end
            coeff = inv2pi * pv * direct / T(2)
            return coeff * dsx, coeff * dsy
        end

        vx, vy = _straight_euler_contribution_scalar(xi, yi, ax, ay, bx, by, pv, inv4pi)
        g_nodes, g_weights = _gl5_nodes_weights(T)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)
        @inbounds for q in 1:5
            sx = (ax + bx) / T(2) + g_nodes[q] * half_dsx
            sy = (ay + by) / T(2) + g_nodes[q] * half_dsy
            rx = sx - xi
            ry = sy - yi
            r2 = rx * rx + ry * ry
            if r2 < eps(T)^2
                corr_integral += g_weights[q] * (log(T(2) * Ld) - T(Base.MathConstants.eulergamma))
            else
                r = sqrt(r2)
                corr_integral += g_weights[q] * _qg_smooth_correction_scalar(r / Ld, r, Ld)
            end
        end
        corr = inv2pi * pv * T(0.5) * corr_integral
        return vx + corr * dsx, vy + corr * dsy
    end

    # One K₀ per quadrature point serves both the direct far-field sum and
    # the smooth correction of the near-field split.
    g_nodes, g_weights = _gl5_nodes_weights(T)
    cvx = zero(T)
    cvy = zero(T)
    direct_x = zero(T)
    direct_y = zero(T)
    min_r = T(Inf)
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        sx, sy, tx, ty = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, p)
        rx = sx - xi
        ry = sy - yi
        r2 = rx * rx + ry * ry
        r = sqrt(r2)
        min_r = min(min_r, r)
        weight = inv2pi * pv * (g_weights[q] / T(2))
        if r2 < eps(T)^2
            val = log(T(2) * Ld) - T(Base.MathConstants.eulergamma)
        else
            k0, val = _qg_k0_and_smooth(r / Ld, r, Ld)
            direct_x += weight * k0 * tx
            direct_y += weight * k0 * ty
        end
        cvx += weight * val * tx
        cvy += weight * val * ty
    end
    min_r > T(4) * max(Ld, ds_len) && return direct_x, direct_y
    evx, evy = _curved_euler_contribution_scalar(xi, yi, ax, ay, bx, by, pv, κa, κb, inv4pi)
    return evx + cvx, evy + cvy
end

# Velocity of a straight regularized SQG panel a→b: the analytic integral of
# t̂/√(r² + δ²) along the panel.
@inline function _straight_sqg_contribution_scalar(xi::T, yi::T,
                                                   ax::T, ay::T, bx::T, by::T,
                                                   pv::T, δ_sq::T, inv2pi::T) where {T}
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    iszero(ds_len) && return zero(T), zero(T)
    tx = dsx / ds_len
    ty = dsy / ds_len
    r0x = xi - ax
    r0y = yi - ay
    u_a = r0x * tx + r0y * ty
    h = -r0x * ty + r0y * tx
    h_eff = sqrt(h * h + δ_sq)
    F_diff = _sqg_asinh_difference(u_a, u_a - ds_len, h_eff, ds_len)
    contrib = inv2pi * pv * F_diff
    return contrib * tx, contrib * ty
end

@inline function _curved_sqg_contribution_scalar(xi::T, yi::T,
                                                 ax::T, ay::T, bx::T, by::T,
                                                 pv::T, κa::T, κb::T,
                                                 δ::T, inv2pi::T) where {T}
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    ds_len < eps(T) && return zero(T), zero(T)
    δ_sq = δ * δ

    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return _straight_sqg_contribution_scalar(xi, yi, ax, ay, bx, by, pv, δ_sq, inv2pi)

    # Singular subtraction: integrate a straight model panel m(p) analytically
    # and only the difference to the cubic c(p) with 5-point Gauss-Legendre.
    # Near an endpoint the model is the tangent line there, m(p) = a + p c'(0)
    # (or b + (p - 1) c'(1)): the chord would leave a θ/p remainder for a target
    # at the endpoint (θ the tangent-chord angle), whose log(ds/δ) integral
    # the quadrature cannot resolve. Elsewhere the chord is the model.
    da2 = (xi - ax)^2 + (yi - ay)^2
    db2 = (xi - bx)^2 + (yi - by)^2
    near2 = ds_len * ds_len / 16
    model = da2 <= near2 && da2 <= db2 ? 1 : db2 <= near2 ? 2 : 0
    m0x, m0y, mtx, mty = if model == 1
        _, _, t0x, t0y = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, zero(T))
        ax, ay, t0x, t0y
    elseif model == 2
        _, _, t1x, t1y = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, one(T))
        bx - t1x, by - t1y, t1x, t1y
    else
        ax, ay, dsx, dsy
    end
    vx, vy = _straight_sqg_contribution_scalar(
        xi, yi, m0x, m0y, m0x + mtx, m0y + mty, pv, δ_sq, inv2pi)

    g_nodes, g_weights = _gl5_nodes_weights(T)
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        sx, sy, tx, ty = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, p)
        rx = xi - sx
        ry = yi - sy
        mx = xi - (m0x + p * mtx)
        my = yi - (m0y + p * mty)
        inv_curve = one(T) / sqrt(rx * rx + ry * ry + δ_sq)
        inv_model = one(T) / sqrt(mx * mx + my * my + δ_sq)
        coeff = inv2pi * pv * (g_weights[q] / T(2))
        vx += coeff * (tx * inv_curve - mtx * inv_model)
        vy += coeff * (ty * inv_curve - mty * inv_model)
    end
    return vx, vy
end

@inline function _nearest_periodic_segment_image_scalar(xi::T, yi::T,
                                                        ax::T, ay::T,
                                                        bx::T, by::T,
                                                        Lx::T, Ly::T) where {T}
    Lx2 = T(2) * Lx
    Ly2 = T(2) * Ly
    midx = (ax + bx) / T(2)
    midy = (ay + by) / T(2)
    shiftx = round((xi - midx) / Lx2) * Lx2
    shifty = round((yi - midy) / Ly2) * Ly2
    return ax + shiftx, ay + shifty, bx + shiftx, by + shifty
end

@inline function _periodic_euler_zero_mode_scalar(α::T, Lx::T, Ly::T) where {T}
    area = T(4) * Lx * Ly
    return one(T) / (T(4) * α^2 * area)
end

@inline function _periodic_euler_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                         α::T, Lx::T, Ly::T,
                                                         n_images::Int,
                                                         dkx::T, dky::T, fourier_table,
                                                         inv4pi::T,
                                                         γ_euler::T) where {T}
    r0x = xi - sx
    r0y = yi - sy
    cutoff = _ewald_real_cutoff(T)

    # Central image: E₁(α²r²) + log r², whose r → 0 limit is -γ - 2 log α.
    r2 = r0x * r0x + r0y * r0y
    G_corr = if r2 <= eps(T)
        inv4pi * (-γ_euler - T(2) * log(α))
    elseif α^2 * r2 <= cutoff
        inv4pi * (_expint_e1(α^2 * r2) + log(r2))
    else
        inv4pi * log(r2)
    end

    reach = sqrt(cutoff) / α
    for px in _ewald_image_range(r0x, Lx, reach, n_images)
        rx = r0x - T(2) * Lx * T(px)
        for py in _ewald_image_range(r0y, Ly, reach, n_images)
            (px == 0 && py == 0) && continue
            ry = r0y - T(2) * Ly * T(py)
            r2 = rx * rx + ry * ry
            if r2 > eps(T) && α^2 * r2 <= cutoff
                G_corr += inv4pi * _expint_e1(α^2 * r2)
            end
        end
    end

    G_corr += _ewald_cosine_sum(fourier_table, dkx, dky, r0x, r0y)
    return G_corr - _periodic_euler_zero_mode_scalar(α, Lx, Ly)
end

# Smooth periodic QG-minus-Euler correction G̃_QG - G̃_E = κ²Ĝ at one
# quadrature point, via the Ewald split of numerics/periodic_ewald.jl.
# `corr_table` is the cache's cosine table of κ²ĉ_k.
@inline function _periodic_qg_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                      kappa2::T, α::T, Lx::T, Ly::T,
                                                      n_images::Int, dkx::T, dky::T,
                                                      corr_table) where {T}
    return _scaled_qg_correction_scalar(xi - sx, yi - sy, kappa2, α,
                                        _ewald_qg_x(kappa2, α), Lx, Ly,
                                        n_images, dkx, dky, corr_table)
end

# Periodic QG segment velocity by direct summation over periodic images, used
# when K₀ decays within a few periods. `a`/`b` must already be the segment's
# nearest image to the target. The zero-mean Green's function removes the
# k = 0 term 1/(Aκ²) of the image sum, which contributes along the chord.
@inline function _periodic_qg_direct_contribution_scalar(xi::T, yi::T,
                                                         ax::T, ay::T, bx::T, by::T,
                                                         pv::T, κa::T, κb::T,
                                                         Ld::T, Lx::T, Ly::T,
                                                         inv2pi::T, inv4pi::T) where {T}
    kappa = one(T) / Ld
    nx = _qg_direct_image_count(kappa, Lx)
    ny = _qg_direct_image_count(kappa, Ly)
    vx = zero(T)
    vy = zero(T)
    for px in -nx:nx
        shx = 2 * Lx * T(px)
        for py in -ny:ny
            shy = 2 * Ly * T(py)
            dvx, dvy = _curved_qg_contribution_scalar(
                xi, yi, ax + shx, ay + shy, bx + shx, by + shy,
                pv, κa, κb, Ld, inv2pi, inv4pi)
            vx += dvx
            vy += dvy
        end
    end
    zero_mode = pv * Ld * Ld / (4 * Lx * Ly)
    return vx - zero_mode * (bx - ax), vy - zero_mode * (by - ay)
end

# Smooth periodic SQG correction G_per - G_unbounded at one quadrature point
# for the regularized kernel 1/(2π r_δ), r_δ = √(r² + δ²). The quasi-2-D Ewald
# split of 1/r_δ (the Coulomb potential of a charge at height δ) treats the
# softening exactly: real-space terms erfc(α r_δ)/r_δ and Fourier coefficients
# (`fourier_table`, see `_ewald_fourier_coefficient`) both decay like
# Gaussians. The zero-mean inversion removes the k = 0 content of the real-space
# sum, 2π[e^{-α²δ²}/(α√π) - δ erfc(αδ)]/A; without it spanning contours with
# Σ pv·wrap ≠ 0 drift with a uniform velocity that depends on α.
@inline function _periodic_sqg_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                       α::T, δ_sq::T,
                                                       Lx::T, Ly::T, n_images::Int,
                                                       dkx::T, dky::T, fourier_table,
                                                       inv2pi::T) where {T}
    r0x = xi - sx
    r0y = yi - sy
    cutoff = _ewald_real_cutoff(T)

    # Central image: erfc(α r_δ)/r_δ - 1/r_δ, finite for δ = 0 as well.
    G_corr = -inv2pi * _sqg_erf_over_r(α, r0x * r0x + r0y * r0y + δ_sq)

    # Other images: erfc(α r_δ)/r_δ ≤ e^{-α²r_δ²}/(α√π r_δ²), negligible past
    # the cutoff.
    reach = sqrt(cutoff) / α
    for px in _ewald_image_range(r0x, Lx, reach, n_images)
        rx = r0x - T(2) * Lx * T(px)
        for py in _ewald_image_range(r0y, Ly, reach, n_images)
            (px == 0 && py == 0) && continue
            ry = r0y - T(2) * Ly * T(py)
            r2_δ = rx * rx + ry * ry + δ_sq
            if α^2 * r2_δ <= cutoff
                r_δ = sqrt(r2_δ)
                G_corr += inv2pi * erfc(α * r_δ) / r_δ
            end
        end
    end

    G_corr += inv2pi * _ewald_cosine_sum(fourier_table, dkx, dky, r0x, r0y)
    δ = sqrt(δ_sq)
    area = T(4) * Lx * Ly
    zero_mode = (exp(-α * α * δ_sq) / (α * sqrt(T(π))) - δ * erfc(α * δ)) / area
    return G_corr - zero_mode
end
