# KA velocity kernels.
#
# Scalar per-segment contribution helpers (no SVector, GPU-friendly) and the
# `@kernel` velocity kernels built on top of them: unbounded and periodic Euler,
# QG, and SQG. These run on `KernelAbstractions.CPU()` (test validation) and on
# CUDA via the package extension. Data layout (`SegmentData`, workspace) lives in
# `packing.jl`; launch/dispatch wrappers live in `velocity.jl`.

# Inline Euler antiderivative — scalar version for GPU (no SVector).
# F(u; h, h_sq) = u*log(u² + h²) - 2u + 2h*arctan(u/h)
@inline function _euler_antideriv_scalar(u::T, h::T, h_sq::T) where {T}
    r2 = u * u + h_sq
    if r2 < eps(T)^2
        return zero(T)
    end
    val = u * log(r2) - 2 * u
    if abs(h) > eps(T)
        val += 2 * h * atan(u / h)
    end
    return val
end

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
    u_b = u_a - ds_len
    h_sq = h * h

    F_diff = _euler_antideriv_scalar(u_a, h, h_sq) - _euler_antideriv_scalar(u_b, h, h_sq)
    contrib = -inv4pi * pv * F_diff
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

    nx = -dsy
    ny = dsx
    α = -ds_len * (T(2) * κa + κb) / T(6)
    β = ds_len * κa / T(2)
    γ = ds_len * (κb - κa) / T(6)
    g_nodes, g_weights = _gl5_nodes_weights(T)
    vx = zero(T)
    vy = zero(T)

    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        η = p * (α + p * (β + p * γ))
        η′ = α + T(2) * β * p + T(3) * γ * p^2
        sx = ax + p * dsx + η * nx
        sy = ay + p * dsy + η * ny
        tangent_x = dsx + η′ * nx
        tangent_y = dsy + η′ * ny
        rx = xi - sx
        ry = yi - sy
        r2 = max(rx * rx + ry * ry, eps(T)^2)
        coeff = -inv4pi * pv * (g_weights[q] / T(2)) * log(r2)
        vx += coeff * tangent_x
        vy += coeff * tangent_y
    end

    return vx, vy
end

@inline function _qg_smooth_correction_scalar(rr::T, r::T, Ld::T) where {T}
    # QG = Euler logarithmic kernel plus a smooth finite deformation-radius
    # correction. The small-r branch uses the same regularized limit as CPU code.
    if rr < T(0.5)
        return _besselk0_correction(rr) + log(T(2) * Ld) - T(Base.MathConstants.eulergamma)
    end
    return _besselk0_approx_scalar(rr) + log(r)
end

@inline function _cubic_point_tangent_scalar(ax::T, ay::T, bx::T, by::T,
                                             κa::T, κb::T, p::T) where {T}
    # Return both point and tangent on the cubic segment in scalar form, avoiding
    # SVector allocation inside GPU kernels.
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    nx = -dsy
    ny = dsx
    α = -ds_len * (T(2) * κa + κb) / T(6)
    β = ds_len * κa / T(2)
    γ = ds_len * (κb - κa) / T(6)
    η = p * (α + p * (β + p * γ))
    η′ = α + T(2) * β * p + T(3) * γ * p^2
    return ax + p * dsx + η * nx,
           ay + p * dsy + η * ny,
           dsx + η′ * nx,
           dsy + η′ * ny
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

    evx, evy = _curved_euler_contribution_scalar(xi, yi, ax, ay, bx, by, pv, κa, κb, inv4pi)
    g_nodes, g_weights = _gl5_nodes_weights(T)
    cvx = zero(T)
    cvy = zero(T)
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        sx, sy, tx, ty = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, p)
        rx = sx - xi
        ry = sy - yi
        r2 = rx * rx + ry * ry
        val = if r2 < eps(T)^2
            log(T(2) * Ld) - T(Base.MathConstants.eulergamma)
        else
            r = sqrt(r2)
            _qg_smooth_correction_scalar(r / Ld, r, Ld)
        end
        coeff = inv2pi * pv * (g_weights[q] / T(2)) * val
        cvx += coeff * tx
        cvy += coeff * ty
    end
    return evx + cvx, evy + cvy
end

@inline function _curved_sqg_contribution_scalar(xi::T, yi::T,
                                                 ax::T, ay::T, bx::T, by::T,
                                                 pv::T, κa::T, κb::T,
                                                 delta::T, inv2pi::T) where {T}
    dsx = bx - ax
    dsy = by - ay
    ds_len = sqrt(dsx^2 + dsy^2)
    ds_len < eps(T) && return zero(T), zero(T)
    delta_sq = delta * delta

    if max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T))
        tx = dsx / ds_len
        ty = dsy / ds_len
        nx = -ty
        ny = tx
        r0x = xi - ax
        r0y = yi - ay
        u_a = r0x * tx + r0y * ty
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len
        h_eff = sqrt(h * h + delta_sq)
        F_diff = asinh(u_a / h_eff) - asinh(u_b / h_eff)
        contrib = inv2pi * pv * F_diff
        return contrib * tx, contrib * ty
    end

    g_nodes, g_weights = _gl5_nodes_weights(T)
    vx = zero(T)
    vy = zero(T)
    @inbounds for q in 1:5
        p = (one(T) + g_nodes[q]) / T(2)
        sx, sy, tx, ty = _cubic_point_tangent_scalar(ax, ay, bx, by, κa, κb, p)
        rx = xi - sx
        ry = yi - sy
        rreg = sqrt(rx * rx + ry * ry + delta_sq)
        coeff = inv2pi * pv * (g_weights[q] / T(2)) / rreg
        vx += coeff * tx
        vy += coeff * ty
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

@inline function _periodic_euler_zero_mode_scalar(alpha::T, Lx::T, Ly::T) where {T}
    area = T(4) * Lx * Ly
    return one(T) / (T(4) * alpha^2 * area)
end

@inline function _periodic_euler_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                         alpha::T, Lx::T, Ly::T,
                                                         n_images::Int,
                                                         kx, ky, fourier_coeffs,
                                                         inv4pi::T,
                                                         gamma_euler::T) where {T}
    r0x = xi - sx
    r0y = yi - sy
    G_corr = zero(T)

    for px in -n_images:n_images
        shiftx = T(2) * Lx * T(px)
        for py in -n_images:n_images
            shifty = T(2) * Ly * T(py)
            rx = r0x - shiftx
            ry = r0y - shifty
            r2 = rx * rx + ry * ry

            if px == 0 && py == 0
                if r2 > eps(T)
                    G_corr += inv4pi * (_expint_e1(alpha^2 * r2) + log(r2))
                else
                    G_corr += inv4pi * (-gamma_euler - T(2) * log(alpha))
                end
            elseif r2 > eps(T)
                G_corr += inv4pi * _expint_e1(alpha^2 * r2)
            end
        end
    end

    nkx = length(kx)
    nky = length(ky)
    for mi in 1:nkx
        kxi = kx[mi]
        cx = cos(kxi * r0x)
        sx_trig = sin(kxi * r0x)
        for ni in 1:nky
            coeff = fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            kyi = ky[ni]
            G_corr += coeff * (cx * cos(kyi * r0y) - sx_trig * sin(kyi * r0y))
        end
    end

    return G_corr - _periodic_euler_zero_mode_scalar(alpha, Lx, Ly)
end

@inline function _periodic_qg_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                      kappa2::T, area::T,
                                                      kx, ky) where {T}
    rx = xi - sx
    ry = yi - sy
    G_corr = zero(T)
    nkx = length(kx)
    nky = length(ky)

    for mi in 1:nkx
        kxi = kx[mi]
        cx = cos(kxi * rx)
        sx_trig = sin(kxi * rx)
        for ni in 1:nky
            kyi = ky[ni]
            k2 = kxi^2 + kyi^2
            k2 < eps(T) && continue
            coeff = -kappa2 / (k2 * (k2 + kappa2) * area)
            G_corr += coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
        end
    end

    return G_corr
end

@inline function _periodic_sqg_green_correction_scalar(xi::T, yi::T, sx::T, sy::T,
                                                       alpha::T, delta_sq::T,
                                                       Lx::T, Ly::T, n_images::Int,
                                                       kx, ky, fourier_coeffs,
                                                       inv2pi::T) where {T}
    r0x = xi - sx
    r0y = yi - sy
    G_corr = zero(T)

    for px in -n_images:n_images
        shiftx = T(2) * Lx * T(px)
        for py in -n_images:n_images
            shifty = T(2) * Ly * T(py)
            rx = r0x - shiftx
            ry = r0y - shifty
            r2 = rx * rx + ry * ry

            if px == 0 && py == 0
                r_reg = sqrt(r2 + delta_sq)
                G_corr -= inv2pi * erf(alpha * r_reg) / r_reg
            elseif r2 > eps(T)
                r = sqrt(r2)
                G_corr += inv2pi * erfc(alpha * r) / r
            end
        end
    end

    nkx = length(kx)
    nky = length(ky)
    for mi in 1:nkx
        kxi = kx[mi]
        cx = cos(kxi * r0x)
        sx_trig = sin(kxi * r0x)
        for ni in 1:nky
            coeff = fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            kyi = ky[ni]
            G_corr += inv2pi * coeff * (cx * cos(kyi * r0y) - sx_trig * sin(kyi * r0y))
        end
    end

    return G_corr
end

"""KernelAbstractions kernel: each workitem computes velocity at one target node."""
@kernel function _euler_velocity_ka!(vel_x, vel_y,
                                      target_x, target_y,
                                      seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                      seg_ka, seg_kb,
                                      n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv4pi = one(T) / (4 * T(π))

    @inbounds for j in 1:n_seg
        dvx, dvy = _curved_euler_contribution_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j],
            seg_pv[j], seg_ka[j], seg_kb[j], inv4pi)
        vx += dvx
        vy += dvy
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _periodic_euler_velocity_ka!(vel_x, vel_y,
                                              target_x, target_y,
                                              seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                              seg_ka, seg_kb,
                                              alpha, Lx, Ly, n_images,
                                              kx, ky, fourier_coeffs,
                                              n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv4pi = one(T) / (T(4) * T(pi))
    gamma_euler = T(Base.MathConstants.eulergamma)
    g_nodes, g_weights = _gl5_nodes_weights(T)

    @inbounds for j in 1:n_seg
        ax, ay, bx, by = _nearest_periodic_segment_image_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j], Lx, Ly)
        dsx = bx - ax
        dsy = by - ay
        ds_len_sq = dsx^2 + dsy^2
        ds_len = sqrt(ds_len_sq)
        ds_len < eps(T) && continue

        if max(abs(seg_ka[j]), abs(seg_kb[j])) * ds_len > sqrt(eps(T))
            dvx, dvy = _curved_euler_contribution_scalar(
                xi, yi, ax, ay, bx, by,
                seg_pv[j], seg_ka[j], seg_kb[j], inv4pi)
            vx += dvx
            vy += dvy

            g5_nodes, g5_weights = _gl5_nodes_weights(T)
            @inbounds for q in 1:5
                p = (one(T) + g5_nodes[q]) / T(2)
                sx, sy, tx_curve, ty_curve = _cubic_point_tangent_scalar(
                    ax, ay, bx, by,
                    seg_ka[j], seg_kb[j], p)
                G_corr = _periodic_euler_green_correction_scalar(
                    xi, yi, sx, sy, alpha, Lx, Ly, n_images,
                    kx, ky, fourier_coeffs, inv4pi, gamma_euler)
                coeff = seg_pv[j] * (g5_weights[q] / T(2)) * G_corr
                vx += coeff * tx_curve
                vy += coeff * ty_curve
            end
            continue
        end

        tx = dsx / ds_len
        ty = dsy / ds_len
        nx = -ty
        ny = tx

        r0x = xi - ax
        r0y = yi - ay
        u_a = r0x * tx + r0y * ty
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len
        h_sq = h * h

        F_diff = _euler_antideriv_scalar(u_a, h, h_sq) - _euler_antideriv_scalar(u_b, h, h_sq)
        contrib = -inv4pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty

        mid_x = (ax + bx) / T(2)
        mid_y = (ay + by) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in eachindex(g_nodes)
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy

            r0x = xi - sx
            r0y = yi - sy
            G_corr = zero(T)

            for px in -n_images:n_images
                shiftx = T(2) * Lx * T(px)
                for py in -n_images:n_images
                    shifty = T(2) * Ly * T(py)
                    rx = r0x - shiftx
                    ry = r0y - shifty
                    r2 = rx * rx + ry * ry

                    if px == 0 && py == 0
                        if r2 > eps(T)
                            G_corr += inv4pi * (_expint_e1(alpha^2 * r2) + log(r2))
                        else
                            G_corr += inv4pi * (-gamma_euler - T(2) * log(alpha))
                        end
                    elseif r2 > eps(T)
                        G_corr += inv4pi * _expint_e1(alpha^2 * r2)
                    end
                end
            end

            rx = r0x
            ry = r0y
            nkx = length(kx)
            nky = length(ky)
            for mi in 1:nkx
                kxi = kx[mi]
                cx = cos(kxi * rx)
                sx_trig = sin(kxi * rx)
                for ni in 1:nky
                    coeff = fourier_coeffs[mi, ni]
                    abs(coeff) < eps(T) && continue
                    kyi = ky[ni]
                    G_corr += coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
                end
            end

            corr_integral += g_weights[q] * (G_corr - _periodic_euler_zero_mode_scalar(alpha, Lx, Ly))
        end

        vx += seg_pv[j] * half_dsx * corr_integral
        vy += seg_pv[j] * half_dsy * corr_integral
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _qg_velocity_ka!(vel_x, vel_y,
                                  target_x, target_y,
                                  seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                  seg_ka, seg_kb,
                                  Ld, n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv2pi = one(T) / (T(2) * T(pi))
    inv4pi = one(T) / (T(4) * T(pi))

    @inbounds for j in 1:n_seg
        dvx, dvy = _curved_qg_contribution_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j],
            seg_pv[j], seg_ka[j], seg_kb[j], Ld, inv2pi, inv4pi)
        vx += dvx
        vy += dvy
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _periodic_qg_correction_ka!(vel_x, vel_y,
                                             target_x, target_y,
                                             seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                             seg_ka, seg_kb,
                                             Ld, Lx, Ly, kx, ky,
                                             n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    kappa2 = one(T) / (Ld * Ld)
    area = T(4) * Lx * Ly
    g_nodes, g_weights = _gl5_nodes_weights(T)
    vx = vel_x[i]
    vy = vel_y[i]

    @inbounds for j in 1:n_seg
        ax, ay, bx, by = _nearest_periodic_segment_image_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j], Lx, Ly)
        dsx = bx - ax
        dsy = by - ay
        ds_len_sq = dsx^2 + dsy^2
        ds_len = sqrt(ds_len_sq)
        ds_len < eps(T) && continue

        if max(abs(seg_ka[j]), abs(seg_kb[j])) * ds_len > sqrt(eps(T))
            g5_nodes, g5_weights = _gl5_nodes_weights(T)
            @inbounds for q in 1:5
                p = (one(T) + g5_nodes[q]) / T(2)
                sx, sy, tx_curve, ty_curve = _cubic_point_tangent_scalar(
                    ax, ay, bx, by,
                    seg_ka[j], seg_kb[j], p)
                G_corr = _periodic_qg_green_correction_scalar(
                    xi, yi, sx, sy, kappa2, area, kx, ky)
                coeff = seg_pv[j] * (g5_weights[q] / T(2)) * G_corr
                vx += coeff * tx_curve
                vy += coeff * ty_curve
            end
            continue
        end

        mid_x = (ax + bx) / T(2)
        mid_y = (ay + by) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in eachindex(g_nodes)
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy
            rx = xi - sx
            ry = yi - sy
            G_corr = zero(T)

            nkx = length(kx)
            nky = length(ky)
            for mi in 1:nkx
                kxi = kx[mi]
                cx = cos(kxi * rx)
                sx_trig = sin(kxi * rx)
                for ni in 1:nky
                    kyi = ky[ni]
                    k2 = kxi^2 + kyi^2
                    k2 < eps(T) && continue
                    coeff = -kappa2 / (k2 * (k2 + kappa2) * area)
                    G_corr += coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
                end
            end

            corr_integral += g_weights[q] * G_corr
        end

        vx += seg_pv[j] * half_dsx * corr_integral
        vy += seg_pv[j] * half_dsy * corr_integral
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _periodic_sqg_velocity_ka!(vel_x, vel_y,
                                            target_x, target_y,
                                            seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                            seg_ka, seg_kb,
                                            alpha, delta, Lx, Ly, n_images,
                                            kx, ky, fourier_coeffs,
                                            n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    delta_sq = delta * delta
    inv2pi = one(T) / (T(2) * T(pi))
    g_nodes, g_weights = _gl5_nodes_weights(T)
    vx = zero(T)
    vy = zero(T)

    @inbounds for j in 1:n_seg
        ax, ay, bx, by = _nearest_periodic_segment_image_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j], Lx, Ly)
        dsx = bx - ax
        dsy = by - ay
        ds_len_sq = dsx^2 + dsy^2
        ds_len = sqrt(ds_len_sq)
        ds_len < eps(T) && continue

        if max(abs(seg_ka[j]), abs(seg_kb[j])) * ds_len > sqrt(eps(T))
            dvx, dvy = _curved_sqg_contribution_scalar(
                xi, yi, ax, ay, bx, by,
                seg_pv[j], seg_ka[j], seg_kb[j], delta, inv2pi)
            vx += dvx
            vy += dvy

            g5_nodes, g5_weights = _gl5_nodes_weights(T)
            @inbounds for q in 1:5
                p = (one(T) + g5_nodes[q]) / T(2)
                sx, sy, tx_curve, ty_curve = _cubic_point_tangent_scalar(
                    ax, ay, bx, by,
                    seg_ka[j], seg_kb[j], p)
                G_corr = _periodic_sqg_green_correction_scalar(
                    xi, yi, sx, sy, alpha, delta_sq, Lx, Ly, n_images,
                    kx, ky, fourier_coeffs, inv2pi)
                coeff = seg_pv[j] * (g5_weights[q] / T(2)) * G_corr
                vx += coeff * tx_curve
                vy += coeff * ty_curve
            end
            continue
        end

        tx = dsx / ds_len
        ty = dsy / ds_len
        nx = -ty
        ny = tx

        r0x = xi - ax
        r0y = yi - ay
        u_a = r0x * tx + r0y * ty
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len

        h_eff = sqrt(h * h + delta_sq)
        F_diff = asinh(u_a / h_eff) - asinh(u_b / h_eff)
        contrib = inv2pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty

        mid_x = (ax + bx) / T(2)
        mid_y = (ay + by) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in eachindex(g_nodes)
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy

            r0x = xi - sx
            r0y = yi - sy
            G_corr = zero(T)

            for px in -n_images:n_images
                shiftx = T(2) * Lx * T(px)
                for py in -n_images:n_images
                    shifty = T(2) * Ly * T(py)
                    rx = r0x - shiftx
                    ry = r0y - shifty
                    r2 = rx * rx + ry * ry

                    if px == 0 && py == 0
                        r_reg = sqrt(r2 + delta_sq)
                        G_corr -= inv2pi * erf(alpha * r_reg) / r_reg
                    elseif r2 > eps(T)
                        r = sqrt(r2)
                        G_corr += inv2pi * erfc(alpha * r) / r
                    end
                end
            end

            rx = r0x
            ry = r0y
            nkx = length(kx)
            nky = length(ky)
            for mi in 1:nkx
                kxi = kx[mi]
                cx = cos(kxi * rx)
                sx_trig = sin(kxi * rx)
                for ni in 1:nky
                    coeff = fourier_coeffs[mi, ni]
                    abs(coeff) < eps(T) && continue
                    kyi = ky[ni]
                    G_corr += inv2pi * coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
                end
            end

            corr_integral += g_weights[q] * G_corr
        end

        vx += seg_pv[j] * half_dsx * corr_integral
        vy += seg_pv[j] * half_dsy * corr_integral
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _sqg_velocity_ka!(vel_x, vel_y,
                                   target_x, target_y,
                                   seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                   seg_ka, seg_kb,
                                   delta, n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv2pi = one(T) / (2 * T(pi))

    @inbounds for j in 1:n_seg
        dvx, dvy = _curved_sqg_contribution_scalar(
            xi, yi, seg_ax[j], seg_ay[j], seg_bx[j], seg_by[j],
            seg_pv[j], seg_ka[j], seg_kb[j], delta, inv2pi)
        vx += dvx
        vy += dvy
    end

    vel_x[i] = vx
    vel_y[i] = vy
end


@kernel function _beta_sawtooth_add_ka!(vel_x, y, beta, kappa, dy, Ly, total)
    # Analytic zonal velocity of `reference staircase - beta*y`, added on top
    # of the contour-integral velocity (mirrors _beta_plane_sawtooth_velocity).
    i = @index(Global)
    if i <= total
        ξ = mod(y[i] + Ly + dy / 2, dy) - dy / 2
        # Shared with the CPU evaluator so the two cannot drift apart; see
        # `_beta_sawtooth_u` for why the small-κ·dy branch is required.
        vel_x[i] += _beta_sawtooth_u(beta, kappa, dy, ξ)
    end
end
