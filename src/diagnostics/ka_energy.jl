# KernelAbstractions-backed single-layer energy reductions.
#
# Energy diagnostics are double contour integrals. The accelerated path packs
# every valid contour segment into structure-of-arrays buffers, evaluates the
# pairwise Green's-function contribution on the selected backend, and reduces
# the result back to a scalar. Segment identity metadata is kept alongside the
# geometry so kernels can handle same-segment singular limits correctly.

"""
    EnergySegmentData

Packed segment geometry and identity metadata for KernelAbstractions energy
reductions. `seg` holds endpoint/PV arrays; `contour_id` and `local_index`
allow device kernels to detect self-segment singular cases without consulting
the original contour objects.
"""
struct EnergySegmentData{A<:AbstractVector, I<:AbstractVector}
    seg::SegmentData{A}
    # `contour_id` is compacted over energy-valid contours only; `local_index`
    # is the original segment index within that contour.
    contour_id::I
    local_index::I
end

function _valid_energy_segment_count(contours)
    # Spanning contours are excluded from scalar energy diagnostics; they have
    # zero enclosed area and do not represent closed vortex patches.
    n = 0
    for c in contours
        _valid_energy_contour(c) || continue
        n += nnodes(c)
    end
    return n
end

function _fill_energy_segment_bufs!(ax, ay, bx, by, pv, contour_id, local_index, contours)
    idx = 1
    cid = 0
    for c in contours
        _valid_energy_contour(c) || continue
        cid += 1
        nc = nnodes(c)
        @inbounds for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            ax[idx] = a[1]
            ay[idx] = a[2]
            bx[idx] = b[1]
            by[idx] = b[2]
            pv[idx] = c.pv
            contour_id[idx] = cid
            local_index[idx] = j
            idx += 1
        end
    end
    return idx - 1
end

function _pack_energy_segments(contours, dev::AbstractDevice, ::Type{T}) where {T}
    # Build CPU buffers first, then move once to the requested device. This keeps
    # packing simple and avoids scalar mutation of device arrays.
    n = _valid_energy_segment_count(contours)
    ax = Vector{T}(undef, n)
    ay = Vector{T}(undef, n)
    bx = Vector{T}(undef, n)
    by = Vector{T}(undef, n)
    pv = Vector{T}(undef, n)
    ka = zeros(T, n)
    kb = zeros(T, n)
    contour_id = Vector{Int}(undef, n)
    local_index = Vector{Int}(undef, n)
    _fill_energy_segment_bufs!(ax, ay, bx, by, pv, contour_id, local_index, contours)
    return EnergySegmentData(
        SegmentData(to_device(dev, ax), to_device(dev, ay), to_device(dev, bx),
                    to_device(dev, by), to_device(dev, pv),
                    to_device(dev, ka), to_device(dev, kb)),
        to_device(dev, contour_id),
        to_device(dev, local_index),
    )
end

@kernel function _state_energy_valid_kernel!(valid, lengths, wrapx, wrapy, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        valid[ci] = lengths[ci] >= 3 && iszero(wrapx[ci]) && iszero(wrapy[ci]) ?
                    UInt8(1) : UInt8(0)
    end
end

@kernel function _state_energy_lengths_kernel!(out_lengths, source_contour,
                                               valid_slots, valid, lengths,
                                               ncontours)
    ci = @index(Global)
    if ci <= ncontours && !iszero(valid[ci])
        slot = valid_slots[ci]
        out_lengths[slot] = lengths[ci]
        source_contour[slot] = ci
    end
end

@kernel function _state_energy_segments_kernel!(ax, ay, bx, by, out_pv, ka, kb,
                                                contour_id, local_index,
                                                out_offsets, source_contour,
                                                x, y, pv, wrapx, wrapy,
                                                in_offsets, in_lengths, nvalid)
    out_ci = @index(Global)
    if out_ci <= nvalid
        ci = source_contour[out_ci]
        out_off = out_offsets[out_ci]
        in_off = in_offsets[ci]
        n = in_lengths[ci]
        @inbounds for li in 1:n
            out_g = out_off + li - 1
            in_g = in_off + li - 1
            ax[out_g] = x[in_g]
            ay[out_g] = y[in_g]
            if li < n
                bx[out_g] = x[in_g + 1]
                by[out_g] = y[in_g + 1]
            else
                bx[out_g] = x[in_off] + wrapx[ci]
                by[out_g] = y[in_off] + wrapy[ci]
            end
            out_pv[out_g] = pv[ci]
            ka[out_g] = zero(eltype(ka))
            kb[out_g] = zero(eltype(kb))
            contour_id[out_g] = out_ci
            local_index[out_g] = li
        end
    end
end

function _pack_energy_segments(state::DeviceContourState{T}, dev::AbstractDevice,
                               ::Type{T}) where {T}
    ncontours = length(state.lengths)
    valid = device_zeros(dev, UInt8, ncontours)
    valid_slots = device_zeros(dev, Int, ncontours)
    valid_count = device_zeros(dev, Int, 1)
    if ncontours > 0
        @_ka_launch dev ncontours _state_energy_valid_kernel!(
            valid, state.lengths, state.wrapx, state.wrapy, ncontours)
        @_ka_launch dev ncontours _prefix_u8_kernel!(
            valid_slots, valid_count, valid, ncontours)
    end
    nvalid = ncontours == 0 ? 0 : to_cpu(valid_count)[1]

    out_lengths = device_zeros(dev, Int, nvalid)
    out_offsets = device_zeros(dev, Int, nvalid)
    source_contour = device_zeros(dev, Int, nvalid)
    if nvalid > 0
        @_ka_launch dev ncontours _state_energy_lengths_kernel!(
            out_lengths, source_contour, valid_slots, valid, state.lengths,
            ncontours)
    end

    total_store = device_zeros(dev, Int, 1)
    if nvalid > 0
        @_ka_launch dev nvalid _prefix_lengths_kernel!(
            out_offsets, total_store, out_lengths, nvalid)
    end
    n = nvalid == 0 ? 0 : to_cpu(total_store)[1]

    ax = device_zeros(dev, T, n)
    ay = device_zeros(dev, T, n)
    bx = device_zeros(dev, T, n)
    by = device_zeros(dev, T, n)
    pv = device_zeros(dev, T, n)
    ka = device_zeros(dev, T, n)
    kb = device_zeros(dev, T, n)
    contour_id = device_zeros(dev, Int, n)
    local_index = device_zeros(dev, Int, n)
    if nvalid > 0 && n > 0
        @_ka_launch dev nvalid _state_energy_segments_kernel!(
            ax, ay, bx, by, pv, ka, kb, contour_id, local_index,
            out_offsets, source_contour, state.x, state.y, state.pv,
            state.wrapx, state.wrapy, state.offsets, state.lengths, nvalid)
    end

    return EnergySegmentData(SegmentData(ax, ay, bx, by, pv, ka, kb),
                             contour_id, local_index)
end

@inline function _same_energy_segment(contour_id, local_index, i, j)
    return contour_id[i] == contour_id[j] && local_index[i] == local_index[j]
end

@inline function _energy_segment_geometry(ax, ay, bx, by, i, ::Type{T}) where {T}
    ax_i = ax[i]
    ay_i = ay[i]
    bx_i = bx[i]
    by_i = by[i]
    dsx = bx_i - ax_i
    dsy = by_i - ay_i
    midx = (ax_i + bx_i) / T(2)
    midy = (ay_i + by_i) / T(2)
    half_dsx = dsx / T(2)
    half_dsy = dsy / T(2)
    return dsx, dsy, midx, midy, half_dsx, half_dsy
end

@inline function _energy_ewald_greens_scalar(rx::T, ry::T, alpha::T, Lx::T, Ly::T,
                                             n_images::Int, kx, ky, fourier_coeffs) where {T}
    inv4pi = one(T) / (T(4) * T(pi))
    G_val = zero(T)

    for px in -n_images:n_images
        shiftx = T(2) * Lx * T(px)
        for py in -n_images:n_images
            shifty = T(2) * Ly * T(py)
            sx = rx - shiftx
            sy = ry - shifty
            r2 = sx * sx + sy * sy
            r2 > eps(T) && (G_val += inv4pi * _expint_e1(alpha * alpha * r2))
        end
    end

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
            G_val += coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
        end
    end

    return G_val
end

@inline function _sqg_ewald_real_potential_scalar(r::T, alpha::T) where {T}
    r <= zero(T) && return zero(T)

    inv_alpha_sqrtpi = one(T) / (alpha * sqrt(T(pi)))
    ar = alpha * r
    z = ar * ar
    gamma_euler = T(Base.MathConstants.eulergamma)

    log_plus_half_e1 = if z < T(0.25)
        s = -log(alpha) - gamma_euler / T(2)
        term = one(T)
        for n in 1:80
            term *= -z / T(n)
            incr = -term / (T(2) * T(n))
            s += incr
            abs(incr) < eps(T) * max(one(T), abs(s)) && break
        end
        s
    else
        log(r) + _expint_e1(z) / T(2)
    end

    zero_limit = (-one(T) - gamma_euler / T(2) - log(alpha)) * inv_alpha_sqrtpi
    return r * erfc(ar) - exp(-z) * inv_alpha_sqrtpi +
        log_plus_half_e1 * inv_alpha_sqrtpi - zero_limit
end

@inline function _sqg_regularized_energy_potential_scalar(r2::T, delta::T) where {T}
    r_delta = sqrt(r2 + delta * delta)
    return r_delta - delta * log(delta + r_delta)
end

@inline function _sqg_periodic_energy_potential_scalar(rx::T, ry::T, alpha::T,
                                                       Lx::T, Ly::T, delta::T,
                                                       n_images::Int, kx, ky,
                                                       fourier_coeffs) where {T}
    delta_sq = delta * delta
    phi = zero(T)

    for px in -n_images:n_images
        shiftx = T(2) * Lx * T(px)
        for py in -n_images:n_images
            shifty = T(2) * Ly * T(py)
            sx = rx - shiftx
            sy = ry - shifty
            r2 = sx * sx + sy * sy
            r = (px == 0 && py == 0) ? sqrt(r2 + delta_sq) : sqrt(r2)
            phi += _sqg_ewald_real_potential_scalar(r, alpha)
        end
    end

    nkx = length(kx)
    nky = length(ky)
    for mi in 1:nkx
        kxi = kx[mi]
        cx = cos(kxi * rx)
        sx_trig = sin(kxi * rx)
        for ni in 1:nky
            kyi = ky[ni]
            k2 = kxi * kxi + kyi * kyi
            k2 < eps(T) && continue
            coeff = fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            phi -= coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry)) / k2
        end
    end

    return phi
end

@kernel function _euler_energy_ka!(partial, ax, ay, bx, by, pv, contour_id, local_index, n_seg)
    # Each work item owns one source segment i and sums interactions with all
    # target segments j. The final host-side sum over partial therefore performs
    # the same double contour integral as the CPU diagnostic path.
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    self_seg_const = T(4) * log(T(2)) - T(6)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy

        quad = zero(T)
        if _same_energy_segment(contour_id, local_index, i, j)
            half_ds_len = sqrt(half_dsix * half_dsix + half_dsiy * half_dsiy)
            quad = half_ds_len > eps(T) ? self_seg_const + T(4) * log(half_ds_len) : zero(T)
        else
            for qi in 1:3
                pix = midix + g_nodes[qi] * half_dsix
                piy = midiy + g_nodes[qi] * half_dsiy
                for qj in 1:3
                    pjx = midjx + g_nodes[qj] * half_dsjx
                    pjy = midjy + g_nodes[qj] * half_dsjy
                    dx = pix - pjx
                    dy = piy - pjy
                    r2 = max(dx * dx + dy * dy, eps(T))
                    quad += g_weights[qi] * g_weights[qj] * log(r2) / T(2)
                end
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _sqg_energy_ka!(partial, ax, ay, bx, by, pv, contour_id, local_index,
                                 delta, n_seg)
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    delta_sq = delta * delta
    g_nodes, g_weights = _gl3_nodes_weights(T)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy
        quad = zero(T)

        for qi in 1:3
            pix = midix + g_nodes[qi] * half_dsix
            piy = midiy + g_nodes[qi] * half_dsiy
            for qj in 1:3
                pjx = midjx + g_nodes[qj] * half_dsjx
                pjy = midjy + g_nodes[qj] * half_dsjy
                dx = pix - pjx
                dy = piy - pjy
                quad += g_weights[qi] * g_weights[qj] *
                    _sqg_regularized_energy_potential_scalar(dx * dx + dy * dy, delta)
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _qg_energy_ka!(partial, ax, ay, bx, by, pv, contour_id, local_index,
                                Ld, n_seg)
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    self_seg_const = T(4) * log(T(2)) - T(6)
    k0_smooth_at_zero = log(T(2) * Ld) - T(Base.MathConstants.eulergamma)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy
        quad = zero(T)

        if _same_energy_segment(contour_id, local_index, i, j)
            half_ds_len = sqrt(half_dsix * half_dsix + half_dsiy * half_dsiy)
            quad_log = half_ds_len > eps(T) ? self_seg_const + T(4) * log(half_ds_len) : zero(T)
            quad_smooth = zero(T)
            for qi in 1:3
                pix = midix + g_nodes[qi] * half_dsix
                piy = midiy + g_nodes[qi] * half_dsiy
                for qj in 1:3
                    pjx = midjx + g_nodes[qj] * half_dsjx
                    pjy = midjy + g_nodes[qj] * half_dsjy
                    dx = pix - pjx
                    dy = piy - pjy
                    r2 = dx * dx + dy * dy
                    if r2 < eps(T)^2
                        quad_smooth += g_weights[qi] * g_weights[qj] * k0_smooth_at_zero
                    else
                        r = sqrt(r2)
                        quad_smooth += g_weights[qi] * g_weights[qj] *
                            (_besselk0_approx_scalar(r / Ld) + log(r))
                    end
                end
            end
            quad = -quad_log + quad_smooth
        else
            for qi in 1:3
                pix = midix + g_nodes[qi] * half_dsix
                piy = midiy + g_nodes[qi] * half_dsiy
                for qj in 1:3
                    pjx = midjx + g_nodes[qj] * half_dsjx
                    pjy = midjy + g_nodes[qj] * half_dsjy
                    dx = pix - pjx
                    dy = piy - pjy
                    r = sqrt(dx * dx + dy * dy)
                    r < eps(T) * Ld && continue
                    quad += g_weights[qi] * g_weights[qj] * _besselk0_approx_scalar(r / Ld)
                end
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _periodic_euler_energy_ka!(partial, ax, ay, bx, by, pv,
                                            contour_id, local_index,
                                            alpha, Lx, Ly, n_images, kx, ky,
                                            fourier_coeffs, corr_at_zero, n_seg)
    # Periodic Euler keeps the same singular split as the CPU path: the
    # self-segment logarithmic term is analytical, and the smooth Ewald
    # correction is evaluated by quadrature.
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    Lx2, Ly2 = _period_lengths(Lx, Ly)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    self_seg_const = T(4) * log(T(2)) - T(6)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy
        quad = zero(T)

        if _same_energy_segment(contour_id, local_index, i, j)
            half_ds_len = sqrt(half_dsix * half_dsix + half_dsiy * half_dsiy)
            quad_analytical = half_ds_len > eps(T) ? self_seg_const + T(4) * log(half_ds_len) : zero(T)
            quad_corr = zero(T)
            for qi in 1:3
                pix = midix + g_nodes[qi] * half_dsix
                piy = midiy + g_nodes[qi] * half_dsiy
                for qj in 1:3
                    pjx = midjx + g_nodes[qj] * half_dsjx
                    pjy = midjy + g_nodes[qj] * half_dsjy
                    rx = pix - pjx
                    ry = piy - pjy
                    r2 = rx * rx + ry * ry
                    if r2 > eps(T)
                        G_per = _energy_ewald_greens_scalar(rx, ry, alpha, Lx, Ly, n_images,
                                                            kx, ky, fourier_coeffs)
                        quad_corr += g_weights[qi] * g_weights[qj] *
                            (-T(2) * T(pi) * G_per - log(r2) / T(2))
                    else
                        quad_corr += g_weights[qi] * g_weights[qj] * corr_at_zero
                    end
                end
            end
            quad = quad_analytical + quad_corr
        else
            for qi in 1:3
                pix = midix + g_nodes[qi] * half_dsix
                piy = midiy + g_nodes[qi] * half_dsiy
                for qj in 1:3
                    pjx = midjx + g_nodes[qj] * half_dsjx
                    pjy = midjy + g_nodes[qj] * half_dsjy
                    rx_raw = pix - pjx
                    ry_raw = piy - pjy
                    rx = rx_raw - round(rx_raw / Lx2) * Lx2
                    ry = ry_raw - round(ry_raw / Ly2) * Ly2
                    G_per = _energy_ewald_greens_scalar(rx, ry, alpha, Lx, Ly, n_images,
                                                        kx, ky, fourier_coeffs)
                    quad += g_weights[qi] * g_weights[qj] * (-T(2) * T(pi) * G_per)
                end
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _periodic_qg_correction_energy_ka!(partial, ax, ay, bx, by, pv,
                                                    contour_id, local_index,
                                                    kappa2, area, kx, ky, n_seg)
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy
        quad = zero(T)

        for qi in 1:3
            pix = midix + g_nodes[qi] * half_dsix
            piy = midiy + g_nodes[qi] * half_dsiy
            for qj in 1:3
                pjx = midjx + g_nodes[qj] * half_dsjx
                pjy = midjy + g_nodes[qj] * half_dsjy
                dx = pix - pjx
                dy = piy - pjy
                G_corr = zero(T)
                nkx = length(kx)
                nky = length(ky)
                for mi in 1:nkx
                    kxi = kx[mi]
                    cx = cos(kxi * dx)
                    sx_trig = sin(kxi * dx)
                    for ni in 1:nky
                        kyi = ky[ni]
                        k2 = kxi * kxi + kyi * kyi
                        k2 < eps(T) && continue
                        coeff = kappa2 / (k2 * (k2 + kappa2) * area)
                        G_corr -= coeff * (cx * cos(kyi * dy) - sx_trig * sin(kyi * dy))
                    end
                end
                quad += g_weights[qi] * g_weights[qj] * (-T(2) * T(pi) * G_corr)
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _periodic_sqg_energy_ka!(partial, ax, ay, bx, by, pv,
                                          contour_id, local_index,
                                          alpha, delta, Lx, Ly, n_images,
                                          kx, ky, fourier_coeffs, n_seg)
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    Lx2, Ly2 = _period_lengths(Lx, Ly)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    local_s = zero(T)

    @inbounds for j in 1:n_seg
        dsjx, dsjy, midjx, midjy, half_dsjx, half_dsjy =
            _energy_segment_geometry(ax, ay, bx, by, j, T)
        dot_ds = dsix * dsjx + dsiy * dsjy
        quad = zero(T)

        for qi in 1:3
            pix = midix + g_nodes[qi] * half_dsix
            piy = midiy + g_nodes[qi] * half_dsiy
            for qj in 1:3
                pjx = midjx + g_nodes[qj] * half_dsjx
                pjy = midjy + g_nodes[qj] * half_dsjy
                rx_raw = pix - pjx
                ry_raw = piy - pjy
                rx = rx_raw - round(rx_raw / Lx2) * Lx2
                ry = ry_raw - round(ry_raw / Ly2) * Ly2
                phi = _sqg_periodic_energy_potential_scalar(rx, ry, alpha, Lx, Ly, delta,
                                                            n_images, kx, ky, fourier_coeffs)
                quad += g_weights[qi] * g_weights[qj] * phi
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

function _ka_energy_raw_with_segments!(kernel!, data::EnergySegmentData, dev::AbstractDevice,
                                       ::Type{T}, args...) where {T}
    # Launch one contribution per packed segment, then reduce on the host. This
    # avoids assuming a portable parallel reduction primitive across KA backends.
    n = length(data.seg.ax)
    n == 0 && return zero(T)
    partial = device_zeros(dev, T, n)
    @_ka_launch dev n kernel!(partial, data.seg.ax, data.seg.ay, data.seg.bx,
                              data.seg.by, data.seg.pv, data.contour_id,
                              data.local_index, args..., n)
    return sum(to_cpu(partial))
end

function _ka_energy_raw(kernel!, contours, dev::AbstractDevice, ::Type{T}, args...) where {T}
    data = _pack_energy_segments(contours, dev, T)
    return _ka_energy_raw_with_segments!(kernel!, data, dev, T, args...)
end

function _periodic_euler_corr_at_zero(cache::EwaldCache{T}, domain::PeriodicDomain{T}) where {T}
    alpha = cache.alpha
    Lx, Ly = domain.Lx, domain.Ly
    corr = (T(Base.MathConstants.eulergamma) + T(2) * log(alpha)) / T(2)
    for px in -cache.n_images:cache.n_images
        for py in -cache.n_images:cache.n_images
            (px == 0 && py == 0) && continue
            shift_r2 = (T(2) * Lx * T(px))^2 + (T(2) * Ly * T(py))^2
            corr -= _expint_e1(alpha^2 * shift_r2) / T(2)
        end
    end
    for (mi, kxi) in enumerate(cache.kx)
        for (ni, _kyi) in enumerate(cache.ky)
            coeff = cache.fourier_coeffs[mi, ni]
            abs(coeff) < eps(T) && continue
            corr -= T(2) * T(π) * coeff
        end
    end
    return corr
end

function _ka_energy(prob::ContourProblem{EulerKernel, UnboundedDomain, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    raw = _ka_energy_raw(_euler_energy_ka!, prob.contours, dev, T)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy(prob::ContourProblem{SQGKernel{T}, UnboundedDomain, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    raw = _ka_energy_raw(_sqg_energy_ka!, prob.contours, dev, T, prob.kernel.delta)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy(prob::ContourProblem{QGKernel{T}, UnboundedDomain, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    raw = _ka_energy_raw(_qg_energy_ka!, prob.contours, dev, T, prob.kernel.Ld)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy(prob::ContourProblem{EulerKernel, PeriodicDomain{T}, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    data = _pack_energy_segments(prob.contours, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        corr0 = _periodic_euler_corr_at_zero(cache, prob.domain)
        _ka_energy_raw_with_segments!(_periodic_euler_energy_ka!, data, dev, T,
                                      cache.alpha, prob.domain.Lx, prob.domain.Ly,
                                      cache.n_images, kx, ky, fourier, corr0)
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy(prob::ContourProblem{QGKernel{T}, PeriodicDomain{T}, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    cache = _get_ewald_cache(prob.domain, EulerKernel())
    data = _pack_energy_segments(prob.contours, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        corr0 = _periodic_euler_corr_at_zero(cache, prob.domain)
        raw_euler = _ka_energy_raw_with_segments!(_periodic_euler_energy_ka!, data, dev, T,
                                                  cache.alpha, prob.domain.Lx, prob.domain.Ly,
                                                  cache.n_images, kx, ky, fourier, corr0)
        kappa2 = one(T) / (prob.kernel.Ld * prob.kernel.Ld)
        area = T(4) * prob.domain.Lx * prob.domain.Ly
        raw_corr = _ka_energy_raw_with_segments!(_periodic_qg_correction_energy_ka!, data, dev, T,
                                                 kappa2, area, kx, ky)
        raw_euler + raw_corr
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy(prob::ContourProblem{SQGKernel{T}, PeriodicDomain{T}, T}, dev::AbstractDevice) where {T}
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    cache = _get_ewald_cache(prob.domain, prob.kernel)
    data = _pack_energy_segments(prob.contours, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        _ka_energy_raw_with_segments!(_periodic_sqg_energy_ka!, data, dev, T,
                                      cache.alpha, prob.kernel.delta,
                                      prob.domain.Lx, prob.domain.Ly,
                                      cache.n_images, kx, ky, fourier)
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, ::EulerKernel,
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    raw = _ka_energy_raw(_euler_energy_ka!, state, dev, T)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel::SQGKernel{T},
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    raw = _ka_energy_raw(_sqg_energy_ka!, state, dev, T, kernel.delta)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel::QGKernel{T},
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    raw = _ka_energy_raw(_qg_energy_ka!, state, dev, T, kernel.Ld)
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel::EulerKernel,
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(state, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        corr0 = _periodic_euler_corr_at_zero(cache, domain)
        _ka_energy_raw_with_segments!(_periodic_euler_energy_ka!, data, dev, T,
                                      cache.alpha, domain.Lx, domain.Ly,
                                      cache.n_images, kx, ky, fourier, corr0)
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel::QGKernel{T},
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, EulerKernel())
    data = _pack_energy_segments(state, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        corr0 = _periodic_euler_corr_at_zero(cache, domain)
        raw_euler = _ka_energy_raw_with_segments!(_periodic_euler_energy_ka!, data, dev, T,
                                                  cache.alpha, domain.Lx, domain.Ly,
                                                  cache.n_images, kx, ky, fourier, corr0)
        kappa2 = one(T) / (kernel.Ld * kernel.Ld)
        area = T(4) * domain.Lx * domain.Ly
        raw_corr = _ka_energy_raw_with_segments!(_periodic_qg_correction_energy_ka!, data, dev, T,
                                                 kappa2, area, kx, ky)
        raw_euler + raw_corr
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel::SQGKernel{T},
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(state, dev, T)
    raw = if length(data.seg.ax) == 0
        zero(T)
    else
        kx = to_device(dev, cache.kx)
        ky = to_device(dev, cache.ky)
        fourier = to_device(dev, cache.fourier_coeffs)
        _ka_energy_raw_with_segments!(_periodic_sqg_energy_ka!, data, dev, T,
                                      cache.alpha, kernel.delta,
                                      domain.Lx, domain.Ly,
                                      cache.n_images, kx, ky, fourier)
    end
    return -(one(T) / (T(4) * T(π))) * raw / T(2)
end
