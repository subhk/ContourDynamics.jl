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
                                                in_offsets, in_lengths,
                                                output_offset, contour_offset,
                                                nvalid)
    out_ci = @index(Global)
    if out_ci <= nvalid
        ci = source_contour[out_ci]
        out_off = out_offsets[out_ci]
        in_off = in_offsets[ci]
        n = in_lengths[ci]
        @inbounds for li in 1:n
            out_g = output_offset + out_off + li - 1
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
            contour_id[out_g] = contour_offset + out_ci
            local_index[out_g] = li
        end
    end
end

# Task-local buffers for repeated energy diagnostics on device-resident state.
# The workspace is rebuilt only when surgery changes the number of contours or
# nodes; validity, compacted topology, segment geometry, reduction storage, and
# periodic Ewald tables are otherwise refilled/reused in place.
mutable struct _EnergyWorkspace{T, DA<:AbstractVector{T}, IA<:AbstractVector{Int},
                                BA<:AbstractVector{UInt8}, DMA<:AbstractMatrix{T}}
    valid::BA
    valid_slots::IA
    valid_count::IA
    scan_a::IA
    scan_b::IA
    out_lengths::IA
    out_offsets::IA
    source_contour::IA
    contour_id::IA
    local_index::IA
    total_store::IA
    ax::DA; ay::DA; bx::DA; by::DA; pv::DA; ka::DA; kb::DA
    partial::DA
    host_count::Vector{Int}
    host_partial::Vector{T}
    dev_ewald_kx::DA
    dev_ewald_ky::DA
    dev_ewald_fourier::DMA
    last_ewald::Union{Nothing,EwaldCache{T}}
    ncontours::Int
    total_nodes::Int
end

function _create_energy_workspace(dev::AbstractDevice, ::Type{T},
                                  ncontours::Int, total_nodes::Int) where {T}
    da = device_zeros(dev, T, total_nodes)
    ia = device_zeros(dev, Int, ncontours)
    ba = device_zeros(dev, UInt8, ncontours)
    dma = device_zeros(dev, T, 0, 0)
    DA, IA, BA, DMA = typeof(da), typeof(ia), typeof(ba), typeof(dma)
    mk_t() = device_zeros(dev, T, total_nodes)
    mk_i_contours() = device_zeros(dev, Int, ncontours)
    mk_i_nodes() = device_zeros(dev, Int, total_nodes)
    _EnergyWorkspace{T,DA,IA,BA,DMA}(
        ba, ia, device_zeros(dev, Int, 1), mk_i_contours(), mk_i_contours(),
        mk_i_contours(), mk_i_contours(), mk_i_contours(),
        mk_i_nodes(), mk_i_nodes(), device_zeros(dev, Int, 1),
        da, mk_t(), mk_t(), mk_t(), mk_t(), mk_t(), mk_t(), mk_t(),
        zeros(Int, 1), Vector{T}(undef, total_nodes),
        device_zeros(dev, T, 0), device_zeros(dev, T, 0), dma, nothing,
        ncontours, total_nodes)
end

const _ENERGY_WS_TLS_KEY = :contourdynamics_energy_workspace

function _get_energy_workspace(dev::AbstractDevice, ::Type{T},
                               ncontours::Int, total_nodes::Int) where {T}
    store = task_local_storage()
    key = (_ENERGY_WS_TLS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    if ws === nothing || (ws::_EnergyWorkspace).ncontours != ncontours ||
       ws.total_nodes != total_nodes
        ws = _create_energy_workspace(dev, T, ncontours, total_nodes)
        store[key] = ws
    end
    return ws
end

function _pack_energy_workspace!(ws::_EnergyWorkspace{T},
                                 state::DeviceContourState{T},
                                 dev::AbstractDevice;
                                 output_offset::Int=0,
                                 contour_offset::Int=0) where {T}
    ncontours = length(state.lengths)
    ncontours <= ws.ncontours || throw(DimensionMismatch(
        "energy workspace contour capacity ($(ws.ncontours)) < state contours ($ncontours)"))
    output_offset + length(state.x) <= ws.total_nodes || throw(DimensionMismatch(
        "energy workspace node capacity ($(ws.total_nodes)) < requested packed extent ($(output_offset + length(state.x)))"))
    ncontours == 0 && return 0
    @_ka_launch dev ncontours _state_energy_valid_kernel!(
        ws.valid, state.lengths, state.wrapx, state.wrapy, ncontours)
    _device_compact_scan!(ws.valid_slots, ws.valid_count, ws.valid,
                          ncontours, dev, ws.scan_a, ws.scan_b)
    copyto!(ws.host_count, ws.valid_count)
    nvalid = ws.host_count[1]
    nvalid == 0 && return 0

    @_ka_launch dev ncontours _state_energy_lengths_kernel!(
        ws.out_lengths, ws.source_contour, ws.valid_slots, ws.valid,
        state.lengths, ncontours)
    @_ka_launch dev nvalid _prefix_lengths_kernel!(
        ws.out_offsets, ws.total_store, ws.out_lengths, nvalid)
    copyto!(ws.host_count, ws.total_store)
    n = ws.host_count[1]
    n == 0 && return 0

    @_ka_launch dev nvalid _state_energy_segments_kernel!(
        ws.ax, ws.ay, ws.bx, ws.by, ws.pv, ws.ka, ws.kb,
        ws.contour_id, ws.local_index, ws.out_offsets, ws.source_contour,
        state.x, state.y, state.pv, state.wrapx, state.wrapy,
        state.offsets, state.lengths, output_offset, contour_offset, nvalid)
    return n
end

function _ensure_energy_ewald!(ws::_EnergyWorkspace{T}, cache::EwaldCache{T},
                               dev::AbstractDevice) where {T}
    if ws.last_ewald !== cache
        ws.dev_ewald_kx = to_device(dev, cache.kx)
        ws.dev_ewald_ky = to_device(dev, cache.ky)
        ws.dev_ewald_fourier = to_device(dev, cache.fourier_coeffs)
        ws.last_ewald = cache
    end
    return ws.dev_ewald_kx, ws.dev_ewald_ky, ws.dev_ewald_fourier
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
        _device_compact_scan!(valid_slots, valid_count, valid, ncontours, dev)
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
            state.wrapx, state.wrapy, state.offsets, state.lengths,
            0, 0, nvalid)
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
    # If phi_delta = r_delta - delta*log(delta + r_delta), then
    # Delta phi_delta = 1/r_delta. The shared energy normalization is
    # -raw/(8pi), so SQG uses 2phi_delta to recover the physical Hamiltonian.
    return T(2) * (r_delta - delta * log(delta + r_delta))
end

@inline function _sqg_periodic_energy_potential_scalar(rx::T, ry::T, alpha::T,
                                                       Lx::T, Ly::T, delta::T,
                                                       n_images::Int, kx, ky,
                                                       fourier_coeffs) where {T}
    phi = zero(T)

    for px in -n_images:n_images
        shiftx = T(2) * Lx * T(px)
        for py in -n_images:n_images
            shifty = T(2) * Ly * T(py)
            sx = rx - shiftx
            sy = ry - shifty
            r2 = sx * sx + sy * sy
            r = sqrt(r2)
            # The regularized potential is already doubled for the shared
            # energy normalization; scale every other Ewald piece likewise.
            phi += T(2) * _sqg_ewald_real_potential_scalar(r, alpha) +
                   _sqg_regularized_energy_potential_scalar(r2, delta) - T(2) * r
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
            phi -= T(2) * coeff *
                   (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry)) / k2
        end
    end

    return phi
end

@kernel function _euler_energy_ka!(partial, ax, ay, bx, by, pv, contour_id, local_index, n_seg)
    # Smooth contour form of the unbounded Euler Hamiltonian.  No self-panel
    # special case is needed because r²(2-log(r²))/4 tends to zero at r=0.
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
                quad += g_weights[qi] * g_weights[qj] *
                        _euler_energy_potential_scalar(dx * dx + dy * dy)
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _log_green_energy_ka!(partial, ax, ay, bx, by, pv,
                                       contour_id, local_index, n_seg)
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
            quad = half_ds_len > eps(T) ?
                   self_seg_const + T(4) * log(half_ds_len) : zero(T)
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
                        _qg_energy_potential_scalar(dx * dx + dy * dy, Ld)
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _periodic_green_energy_ka!(partial, ax, ay, bx, by, pv,
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

@kernel function _periodic_euler_energy_ka!(partial, ax, ay, bx, by, pv,
                                            contour_id, local_index,
                                            alpha, Lx, Ly, n_images, kx, ky,
                                            fourier_coeffs, corr_at_zero, n_seg)
    # Periodic Euler Hamiltonian in contour form.  For G_k=1/(A|k|²), the
    # smooth contour potential required by the shared normalization is
    # -4π cos(k·r)/(A|k|⁴).
    i = @index(Global)
    T = eltype(partial)
    dsix, dsiy, midix, midiy, half_dsix, half_dsiy =
        _energy_segment_geometry(ax, ay, bx, by, i, T)
    g_nodes, g_weights = _gl3_nodes_weights(T)
    area = T(4) * Lx * Ly
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
                phi = zero(T)
                for mi in 1:length(kx)
                    kxi = kx[mi]
                    cx = cos(kxi * dx)
                    sx = sin(kxi * dx)
                    for ni in 1:length(ky)
                        kyi = ky[ni]
                        k2 = kxi * kxi + kyi * kyi
                        k2 < eps(T) && continue
                        phase_cos = cx * cos(kyi * dy) - sx * sin(kyi * dy)
                        phi -= T(4) * T(pi) * phase_cos / (area * k2 * k2)
                    end
                end
                quad += g_weights[qi] * g_weights[qj] * phi
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    partial[i] = pv[i] * local_s
end

@kernel function _periodic_qg_energy_ka!(partial, ax, ay, bx, by, pv,
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
                phi = zero(T)
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
                        phase_cos = cx * cos(kyi * dy) - sx_trig * sin(kyi * dy)
                        phi -= T(4) * T(pi) * phase_cos /
                               (area * k2 * (k2 + kappa2))
                    end
                end
                quad += g_weights[qi] * g_weights[qj] * phi
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

function _ka_energy_raw_with_workspace!(kernel!, ws::_EnergyWorkspace{T}, n::Int,
                                        dev::AbstractDevice, args...) where {T}
    n == 0 && return zero(T)
    @_ka_launch dev n kernel!(ws.partial, ws.ax, ws.ay, ws.bx, ws.by, ws.pv,
                              ws.contour_id, ws.local_index, args..., n)
    copyto!(ws.host_partial, 1, ws.partial, 1, n)
    total = zero(T)
    @inbounds for i in 1:n
        total += ws.host_partial[i]
    end
    return total
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

"""
    _EnergySource{T}

Anything the energy path can pack into segments: a host contour vector or a
flat device state. `_pack_energy_segments` has a method for each, so the
kernel/domain implementations below are written once and serve both the
CPU-device path (`prob.contours`) and the GPU path (`prob.device_state`).
"""
const _EnergySource{T} = Union{DeviceContourState{T}, Vector{PVContour{T}}}

@inline function _energy_contour_circulation(contours::Vector{PVContour{T}}) where {T}
    gamma = zero(T)
    for c in contours
        _valid_energy_contour(c) || continue
        gamma += c.pv * vortex_area(c)
    end
    return gamma
end

# The 2-D Ewald split of the softened SQG kernel retains a spatially constant
# coefficient even though the fractional-Laplacian inverse is defined only for
# nonzero Fourier modes. The unregularized real-space term contributes
# 1/(A*alpha*sqrt(pi)); softening contributes -delta/A.
@inline function _sqg_periodic_ewald_zero_mode(cache::EwaldCache{T},
                                                domain::PeriodicDomain{T},
                                                delta::T) where {T}
    area = T(4) * domain.Lx * domain.Ly
    return (inv(cache.alpha * sqrt(T(pi))) - delta) / area
end

# Upload the Ewald tables once per call — every periodic energy kernel takes
# the same (kx, ky, fourier_coeffs) triple.
@inline function _device_ewald_tables(cache::EwaldCache, dev::AbstractDevice)
    return (to_device(dev, cache.kx), to_device(dev, cache.ky),
            to_device(dev, cache.fourier_coeffs))
end

# GPU problems keep their nodes in `device_state`, CPU-device problems in the
# host `contours` vector; both are valid `_EnergySource`s.
function _ka_energy(prob::ContourProblem, dev::AbstractDevice)
    prob.dev isa GPU && return _ka_energy_from_state(prob.device_state, prob.kernel, prob.domain, dev)
    return _ka_energy_from_state(prob.contours, prob.kernel, prob.domain, dev)
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, ::EulerKernel,
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    return _normalize_energy(_ka_energy_raw(_euler_energy_ka!, src, dev, T))
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, kernel::SQGKernel{T},
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    return _normalize_energy(_ka_energy_raw(_sqg_energy_ka!, src, dev, T, kernel.delta))
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, kernel::QGKernel{T},
                               ::UnboundedDomain, dev::AbstractDevice) where {T}
    return _normalize_energy(_ka_energy_raw(_qg_energy_ka!, src, dev, T, kernel.Ld))
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, kernel::EulerKernel,
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(src, dev, T)
    length(data.seg.ax) == 0 && return _normalize_energy(zero(T))
    kx, ky, fourier = _device_ewald_tables(cache, dev)
    corr0 = _periodic_euler_corr_at_zero(cache, domain)
    raw = _ka_energy_raw_with_segments!(_periodic_euler_energy_ka!, data, dev, T,
                                        cache.alpha, domain.Lx, domain.Ly,
                                        cache.n_images, kx, ky, fourier, corr0)
    return _normalize_energy(raw)
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, kernel::QGKernel{T},
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(src, dev, T)
    length(data.seg.ax) == 0 && return zero(T)
    kx, ky, fourier = _device_ewald_tables(cache, dev)
    kappa2 = one(T) / (kernel.Ld * kernel.Ld)
    area = T(4) * domain.Lx * domain.Ly
    raw = _ka_energy_raw_with_segments!(
        _periodic_qg_energy_ka!, data, dev, T, kappa2, area, kx, ky)
    gamma = _energy_contour_circulation(src)
    return _normalize_energy(raw) + gamma * gamma / (T(2) * area * kappa2)
end

function _ka_energy_from_state(src::Vector{PVContour{T}}, kernel::SQGKernel{T},
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(src, dev, T)
    length(data.seg.ax) == 0 && return _normalize_energy(zero(T))
    kx, ky, fourier = _device_ewald_tables(cache, dev)
    raw = _ka_energy_raw_with_segments!(_periodic_sqg_energy_ka!, data, dev, T,
                                        cache.alpha, kernel.delta,
                                        domain.Lx, domain.Ly,
                                        cache.n_images, kx, ky, fourier)
    gamma = _energy_contour_circulation(src)
    zero_mode = _sqg_periodic_ewald_zero_mode(cache, domain, kernel.delta)
    return _normalize_energy(raw) - zero_mode * gamma * gamma / T(2)
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel,
                               domain::AbstractDomain,
                               dev::AbstractDevice) where {T}
    ws = _get_energy_workspace(dev, T, length(state.lengths), length(state.x))
    return _ka_energy_state_with_ws(state, kernel, domain, dev, ws)
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, ::EulerKernel,
                                  ::UnboundedDomain, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    n = _pack_energy_workspace!(ws, state, dev)
    return _normalize_energy(
        _ka_energy_raw_with_workspace!(_euler_energy_ka!, ws, n, dev))
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, kernel::SQGKernel{T},
                                  ::UnboundedDomain, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    n = _pack_energy_workspace!(ws, state, dev)
    return _normalize_energy(
        _ka_energy_raw_with_workspace!(_sqg_energy_ka!, ws, n, dev, kernel.delta))
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, kernel::QGKernel{T},
                                  ::UnboundedDomain, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    n = _pack_energy_workspace!(ws, state, dev)
    return _normalize_energy(
        _ka_energy_raw_with_workspace!(_qg_energy_ka!, ws, n, dev, kernel.Ld))
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, kernel::EulerKernel,
                                  domain::PeriodicDomain{T}, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    n = _pack_energy_workspace!(ws, state, dev)
    n == 0 && return _normalize_energy(zero(T))
    kx, ky, fourier = _ensure_energy_ewald!(ws, cache, dev)
    corr0 = _periodic_euler_corr_at_zero(cache, domain)
    raw = _ka_energy_raw_with_workspace!(
        _periodic_euler_energy_ka!, ws, n, dev, cache.alpha,
        domain.Lx, domain.Ly, cache.n_images, kx, ky, fourier, corr0)
    return _normalize_energy(raw)
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, kernel::QGKernel{T},
                                  domain::PeriodicDomain{T}, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    n = _pack_energy_workspace!(ws, state, dev)
    n == 0 && return zero(T)
    kx, ky, fourier = _ensure_energy_ewald!(ws, cache, dev)
    kappa2 = one(T) / (kernel.Ld * kernel.Ld)
    area = T(4) * domain.Lx * domain.Ly
    raw = _ka_energy_raw_with_workspace!(
        _periodic_qg_energy_ka!, ws, n, dev, kappa2, area, kx, ky)
    gamma = _state_circulation(state, dev)
    return _normalize_energy(raw) + gamma * gamma / (T(2) * area * kappa2)
end

function _ka_energy_state_with_ws(state::DeviceContourState{T}, kernel::SQGKernel{T},
                                  domain::PeriodicDomain{T}, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    n = _pack_energy_workspace!(ws, state, dev)
    n == 0 && return _normalize_energy(zero(T))
    kx, ky, fourier = _ensure_energy_ewald!(ws, cache, dev)
    raw = _ka_energy_raw_with_workspace!(
        _periodic_sqg_energy_ka!, ws, n, dev, cache.alpha, kernel.delta,
        domain.Lx, domain.Ly, cache.n_images, kx, ky, fourier)
    gamma = _state_circulation(state, dev)
    zero_mode = _sqg_periodic_ewald_zero_mode(cache, domain, kernel.delta)
    return _normalize_energy(raw) - zero_mode * gamma * gamma / T(2)
end

# ── Multi-layer modal energy ─────────────────────────────────────────────

mutable struct _MultilayerEnergyWorkspace{
        T, EW<:_EnergyWorkspace{T}, DA<:AbstractVector{T}}
    energy::EW
    base_pv::DA
end

const _MULTILAYER_ENERGY_WS_TLS_KEY = :contourdynamics_multilayer_energy_workspace

function _create_multilayer_energy_workspace(dev::AbstractDevice, ::Type{T},
                                             max_contours::Int,
                                             total_nodes::Int) where {T}
    energy = _create_energy_workspace(dev, T, max_contours, total_nodes)
    base_pv = device_zeros(dev, T, total_nodes)
    return _MultilayerEnergyWorkspace{T,typeof(energy),typeof(base_pv)}(
        energy, base_pv)
end

function _get_multilayer_energy_workspace(
        states::NTuple{N, <:DeviceContourState{T}}, dev::AbstractDevice) where {N, T}
    max_contours = maximum(s -> length(s.lengths), states; init=0)
    total_nodes = sum(s -> length(s.x), states; init=0)
    store = task_local_storage()
    key = (_MULTILAYER_ENERGY_WS_TLS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    if ws === nothing ||
       (ws::_MultilayerEnergyWorkspace).energy.ncontours != max_contours ||
       ws.energy.total_nodes != total_nodes
        ws = _create_multilayer_energy_workspace(
            dev, T, max_contours, total_nodes)
        store[key] = ws
    end
    return ws
end

@kernel function _copy_multilayer_base_pv_kernel!(base_pv, pv, output_offset, n)
    i = @index(Global)
    if i <= n
        g = output_offset + i
        base_pv[g] = pv[g]
    end
end

@kernel function _apply_multilayer_modal_pv_kernel!(pv, base_pv, weight,
                                                    output_offset, n)
    i = @index(Global)
    if i <= n
        g = output_offset + i
        pv[g] = weight * base_pv[g]
    end
end

function _pack_multilayer_energy_workspace!(
        ws::_MultilayerEnergyWorkspace{T},
        states::NTuple{N, <:DeviceContourState{T}},
        dev::AbstractDevice) where {N, T}
    layer_lengths = MVector{N,Int}(undef)
    output_offset = 0
    contour_offset = 0
    for layer in 1:N
        state = states[layer]
        n = _pack_energy_workspace!(
            ws.energy, state, dev;
            output_offset, contour_offset)
        layer_lengths[layer] = n
        if n > 0
            @_ka_launch dev n _copy_multilayer_base_pv_kernel!(
                ws.base_pv, ws.energy.pv, output_offset, n)
        end
        output_offset += n
        # Full layer contour counts keep compacted ids distinct across layers.
        contour_offset += length(state.lengths)
    end
    return SVector{N,Int}(layer_lengths), output_offset
end

function _apply_multilayer_modal_pv!(
        ws::_MultilayerEnergyWorkspace{T}, layer_lengths::SVector{N,Int},
        weights::NTuple{N,T}, dev::AbstractDevice) where {N,T}
    output_offset = 0
    for layer in 1:N
        n = layer_lengths[layer]
        if n > 0
            @_ka_launch dev n _apply_multilayer_modal_pv_kernel!(
                ws.energy.pv, ws.base_pv, weights[layer], output_offset, n)
        end
        output_offset += n
    end
    return ws.energy
end

"""
    _ka_multilayer_energy_from_states(states, kernel, domain, dev)

Device-resident multi-layer QG energy: diagonalize the vertical coupling and
evaluate energy mode-by-mode, exactly mirroring the CPU modal decomposition.
The barotropic (λ≈0) mode uses the Euler energy kernel; nonzero modes use the
QG kernel with modal deformation radius 1/√|λ|.
"""
function _ka_multilayer_energy_from_states(states::NTuple{N, <:DeviceContourState{T}},
                                           kernel::MultiLayerQGKernel{N},
                                           domain::UnboundedDomain,
                                           dev::AbstractDevice) where {N, T}
    ws = _get_multilayer_energy_workspace(states, dev)
    return _ka_multilayer_energy_with_ws(states, kernel, domain, dev, ws)
end

function _ka_multilayer_energy_with_ws(
        states::NTuple{N, <:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, ::UnboundedDomain,
        dev::AbstractDevice,
        ws::_MultilayerEnergyWorkspace{T}) where {N,T}
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    layer_lengths, total = _pack_multilayer_energy_workspace!(ws, states, dev)
    total == 0 && return zero(T)
    raw = zero(T)
    for mode in 1:N
        weights = ntuple(ℓ -> T(P_inv[mode, ℓ]), Val(N))
        energy_ws = _apply_multilayer_modal_pv!(ws, layer_lengths, weights, dev)
        lam = evals[mode]
        raw += if _is_barotropic_mode(kernel, lam)
            _ka_energy_raw_with_workspace!(
                _euler_energy_ka!, energy_ws, total, dev)
        else
            _ka_energy_raw_with_workspace!(
                _qg_energy_ka!, energy_ws, total, dev,
                one(T) / sqrt(abs(lam)))
        end
    end
    return _normalize_energy(raw)
end

function _ka_multilayer_energy_from_states(states::NTuple{N, <:DeviceContourState{T}},
                                           kernel::MultiLayerQGKernel{N},
                                           domain::PeriodicDomain{T},
                                           dev::AbstractDevice) where {N, T}
    ws = _get_multilayer_energy_workspace(states, dev)
    return _ka_multilayer_energy_with_ws(states, kernel, domain, dev, ws)
end

function _ka_multilayer_energy_with_ws(
        states::NTuple{N, <:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, domain::PeriodicDomain{T},
        dev::AbstractDevice,
        ws::_MultilayerEnergyWorkspace{T}) where {N,T}
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    # Every mode uses the same Fourier grid; its kernel differs only through
    # the modal eigenvalue in the denominator.
    cache = _get_ewald_cache(domain, EulerKernel())
    kx, ky, fourier = _ensure_energy_ewald!(ws.energy, cache, dev)
    corr0 = _periodic_euler_corr_at_zero(cache, domain)
    area = T(4) * domain.Lx * domain.Ly
    layer_lengths, total = _pack_multilayer_energy_workspace!(ws, states, dev)
    total == 0 && return zero(T)
    raw = zero(T)
    E_zero = zero(T)
    layer_circulation = ntuple(ℓ -> _state_circulation(states[ℓ], dev), Val(N))
    for mode in 1:N
        weights = ntuple(ℓ -> T(P_inv[mode, ℓ]), Val(N))
        energy_ws = _apply_multilayer_modal_pv!(ws, layer_lengths, weights, dev)
        lam = evals[mode]
        is_euler_mode = _is_barotropic_mode(kernel, lam)
        raw_mode = if is_euler_mode
            _ka_energy_raw_with_workspace!(
                _periodic_euler_energy_ka!, energy_ws, total, dev,
                cache.alpha, domain.Lx, domain.Ly,
                cache.n_images, kx, ky, fourier, corr0)
        else
            gamma_mode = sum(P_inv[mode, ℓ] * layer_circulation[ℓ] for ℓ in 1:N)
            E_zero += gamma_mode * gamma_mode / (T(2) * area * abs(lam))
            _ka_energy_raw_with_workspace!(
                _periodic_qg_energy_ka!, energy_ws, total, dev,
                T(abs(lam)), area, kx, ky)
        end
        raw += raw_mode
    end
    return _normalize_energy(raw) + E_zero
end
