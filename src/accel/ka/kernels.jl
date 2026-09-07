# KernelAbstractions velocity loops. Scalar formulas live in numerics/velocity_segments.jl.

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
                                              α, Lx, Ly, n_images,
                                              kx, ky, fourier_coeffs,
                                              n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv4pi = one(T) / (T(4) * T(pi))
    γ_euler = T(Base.MathConstants.eulergamma)
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
                    xi, yi, sx, sy, α, Lx, Ly, n_images,
                    kx, ky, fourier_coeffs, inv4pi, γ_euler)
                coeff = seg_pv[j] * (g5_weights[q] / T(2)) * G_corr
                vx += coeff * tx_curve
                vy += coeff * ty_curve
            end
            continue
        end

        dvx, dvy = _straight_euler_contribution_scalar(
            xi, yi, ax, ay, bx, by, seg_pv[j], inv4pi)
        vx += dvx
        vy += dvy

        mid_x = (ax + bx) / T(2)
        mid_y = (ay + by) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in eachindex(g_nodes)
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy
            G_corr = _periodic_euler_green_correction_scalar(
                xi, yi, sx, sy, α, Lx, Ly, n_images,
                kx, ky, fourier_coeffs, inv4pi, γ_euler)
            corr_integral += g_weights[q] * G_corr
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
            G_corr = _periodic_qg_green_correction_scalar(
                xi, yi, sx, sy, kappa2, area, kx, ky)
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
                                            α, δ, Lx, Ly, n_images,
                                            kx, ky, fourier_coeffs,
                                            n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    δ_sq = δ * δ
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
                seg_pv[j], seg_ka[j], seg_kb[j], δ, inv2pi)
            vx += dvx
            vy += dvy

            g5_nodes, g5_weights = _gl5_nodes_weights(T)
            @inbounds for q in 1:5
                p = (one(T) + g5_nodes[q]) / T(2)
                sx, sy, tx_curve, ty_curve = _cubic_point_tangent_scalar(
                    ax, ay, bx, by,
                    seg_ka[j], seg_kb[j], p)
                G_corr = _periodic_sqg_green_correction_scalar(
                    xi, yi, sx, sy, α, δ_sq, Lx, Ly, n_images,
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

        h_eff = sqrt(h * h + δ_sq)
        F_diff = _sqg_asinh_difference(u_a, u_b, h_eff, ds_len)
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
            G_corr = _periodic_sqg_green_correction_scalar(
                xi, yi, sx, sy, α, δ_sq, Lx, Ly, n_images,
                kx, ky, fourier_coeffs, inv2pi)
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
                                   δ, n_seg)
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
            seg_pv[j], seg_ka[j], seg_kb[j], δ, inv2pi)
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
