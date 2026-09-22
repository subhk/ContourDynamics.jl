# KernelAbstractions-backed single-layer energy reductions.
#
# Energy diagnostics are double contour integrals. The accelerated path packs
# every valid contour segment into structure-of-arrays buffers, evaluates the
# pairwise Green's-function contribution on the selected backend, and reduces
# the result back to a scalar. The smooth energy potentials only need segment
# endpoints and PV jumps, including at coincident quadrature points.

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

function _fill_energy_segment_bufs!(ax, ay, bx, by, pv, contours)
    idx = 1
    for c in contours
        _valid_energy_contour(c) || continue
        nc = nnodes(c)
        @inbounds for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            ax[idx] = a[1]
            ay[idx] = a[2]
            bx[idx] = b[1]
            by[idx] = b[2]
            pv[idx] = c.pv
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
    _fill_energy_segment_bufs!(ax, ay, bx, by, pv, contours)
    return (; ax=to_device(dev, ax), ay=to_device(dev, ay),
              bx=to_device(dev, bx), by=to_device(dev, by), pv=to_device(dev, pv))
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

@kernel function _state_energy_segments_kernel!(ax, ay, bx, by, out_pv,
                                                out_offsets, source_contour,
                                                x, y, pv, wrapx, wrapy,
                                                in_offsets, in_lengths,
                                                output_offset, nvalid)
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
        end
    end
end

# ExecutionWorkspace-owned buffers for repeated device energy diagnostics.
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
    total_store::IA
    ax::DA; ay::DA; bx::DA; by::DA; pv::DA
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
    _EnergyWorkspace{T,DA,IA,BA,DMA}(
        ba, ia, device_zeros(dev, Int, 1), mk_i_contours(), mk_i_contours(),
        mk_i_contours(), mk_i_contours(), mk_i_contours(),
        device_zeros(dev, Int, 1),
        da, mk_t(), mk_t(), mk_t(), mk_t(), mk_t(),
        zeros(Int, 1), Vector{T}(undef, total_nodes),
        device_zeros(dev, T, 0), device_zeros(dev, T, 0), dma, nothing,
        ncontours, total_nodes)
end

const _ENERGY_WS_KEY = :contourdynamics_energy_workspace

function _get_energy_workspace(dev::AbstractDevice, ::Type{T},
                               ncontours::Int, total_nodes::Int; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    store = workspace.buffers
    key = (_ENERGY_WS_KEY, T, typeof(dev))
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
                                 output_offset::Int=0) where {T}
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
        ws.ax, ws.ay, ws.bx, ws.by, ws.pv, ws.out_offsets, ws.source_contour,
        state.x, state.y, state.pv, state.wrapx, state.wrapy,
        state.offsets, state.lengths, output_offset, nvalid)
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

@inline function _sqg_regularized_energy_potential_scalar(r2::T, δ::T) where {T}
    r_δ = sqrt(r2 + δ * δ)
    # If phi_δ = r_δ - δ*log(δ + r_δ), then
    # Delta phi_δ = 1/r_δ. The shared energy normalization is
    # -raw/(8pi), so SQG uses 2phi_δ to recover the physical Hamiltonian.
    return T(2) * (r_δ - δ * log(δ + r_δ))
end

@inline function _sqg_periodic_energy_potential_scalar(rx::T, ry::T, α::T,
                                                       Lx::T, Ly::T, δ::T,
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
            phi += T(2) * _sqg_ewald_real_potential(r, α) +
                   _sqg_regularized_energy_potential_scalar(r2, δ) - T(2) * r
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
            iszero(k2) && continue
            coeff = fourier_coeffs[mi, ni]
            iszero(coeff) && continue
            phi -= T(2) * coeff *
                   (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry)) / k2
        end
    end

    return phi
end

# All six kernels use the same straight-segment 3×3 Gauss–Legendre rule.
@inline function _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg,
                                      potential::F) where {F}
    T = eltype(pv)
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
                        potential(dx, dy)
            end
        end
        local_s += pv[j] * quad * dot_ds / T(4)
    end

    return pv[i] * local_s
end

@kernel function _euler_energy_ka!(partial, ax, ay, bx, by, pv, n_seg)
    i = @index(Global)
    potential = (dx, dy) -> _euler_energy_potential_scalar(dx * dx + dy * dy)
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

@kernel function _sqg_energy_ka!(partial, ax, ay, bx, by, pv, δ, n_seg)
    i = @index(Global)
    potential = (dx, dy) ->
        _sqg_regularized_energy_potential_scalar(dx * dx + dy * dy, δ)
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

@kernel function _qg_energy_ka!(partial, ax, ay, bx, by, pv, Ld, n_seg)
    i = @index(Global)
    potential = (dx, dy) -> _qg_energy_potential_scalar(dx * dx + dy * dy, Ld)
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

# For G_k = 1/[A(k²+κ²)], the shared contour-energy normalization needs
# -4π cos(k·r)/[A k²(k²+κ²)]. Euler is the κ²=0 case. The k=0 energy,
# when present, is added by the problem-level caller.
@inline function _periodic_energy_potential_scalar(dx::T, dy::T, kappa2::T,
                                                   area::T, kx, ky) where {T}
    phi = zero(T)
    for mi in eachindex(kx)
        kxi = kx[mi]
        cx = cos(kxi * dx)
        sx = sin(kxi * dx)
        for ni in eachindex(ky)
            kyi = ky[ni]
            k2 = kxi * kxi + kyi * kyi
            iszero(k2) && continue
            phase_cos = cx * cos(kyi * dy) - sx * sin(kyi * dy)
            phi -= T(4) * T(pi) * phase_cos / (area * k2 * (k2 + kappa2))
        end
    end
    return phi
end

@kernel function _periodic_euler_energy_ka!(partial, ax, ay, bx, by, pv,
                                            Lx, Ly, kx, ky, n_seg)
    i = @index(Global)
    T = eltype(partial)
    area = T(4) * Lx * Ly
    potential = (dx, dy) ->
        _periodic_energy_potential_scalar(dx, dy, zero(T), area, kx, ky)
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

@kernel function _periodic_qg_energy_ka!(partial, ax, ay, bx, by, pv,
                                         kappa2, area, kx, ky, n_seg)
    i = @index(Global)
    potential = (dx, dy) ->
        _periodic_energy_potential_scalar(dx, dy, kappa2, area, kx, ky)
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

@kernel function _periodic_sqg_energy_ka!(partial, ax, ay, bx, by, pv,
                                          α, δ, Lx, Ly, n_images,
                                          kx, ky, fourier_coeffs, n_seg)
    i = @index(Global)
    Lx2, Ly2 = _period_lengths(Lx, Ly)
    potential = (dx, dy) -> begin
        rx = dx - round(dx / Lx2) * Lx2
        ry = dy - round(dy / Ly2) * Ly2
        _sqg_periodic_energy_potential_scalar(rx, ry, α, Lx, Ly, δ,
                                               n_images, kx, ky, fourier_coeffs)
    end
    partial[i] = _energy_segment_sum(i, ax, ay, bx, by, pv, n_seg, potential)
end

function _ka_energy_raw_with_segments!(kernel!, data::NamedTuple, dev::AbstractDevice,
                                       ::Type{T}, args...) where {T}
    # Launch one contribution per packed segment, then reduce on the host. This
    # avoids assuming a portable parallel reduction primitive across KA backends.
    n = length(data.ax)
    n == 0 && return zero(T)
    partial = device_zeros(dev, T, n)
    @_ka_launch dev n kernel!(partial, data.ax, data.ay, data.bx,
                              data.by, data.pv, args..., n)
    return sum(to_cpu(partial))
end

function _ka_energy_raw_with_workspace!(kernel!, ws::_EnergyWorkspace{T}, n::Int,
                                        dev::AbstractDevice, args...) where {T}
    n == 0 && return zero(T)
    @_ka_launch dev n kernel!(ws.partial, ws.ax, ws.ay, ws.bx, ws.by, ws.pv, args..., n)
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

@inline function _energy_contour_circulation(contours::Vector{PVContour{T}}) where {T}
    γ = zero(T)
    for c in contours
        _valid_energy_contour(c) || continue
        γ += c.pv * vortex_area(c)
    end
    return γ
end

# The 2-D Ewald split of the softened SQG kernel retains a spatially constant
# coefficient even though the fractional-Laplacian inverse is defined only for
# nonzero Fourier modes. The unregularized real-space term contributes
# 1/(A*α*sqrt(pi)); softening contributes -δ/A.
@inline function _sqg_periodic_ewald_zero_mode(cache::EwaldCache{T},
                                                domain::PeriodicDomain{T},
                                                δ::T) where {T}
    area = T(4) * domain.Lx * domain.Ly
    return (inv(cache.α * sqrt(T(pi))) - δ) / area
end

# Upload the Ewald tables once per call — every periodic energy kernel takes
# the same (kx, ky, fourier_coeffs) triple.
@inline function _device_ewald_tables(cache::EwaldCache, dev::AbstractDevice)
    return (to_device(dev, cache.kx), to_device(dev, cache.ky),
            to_device(dev, cache.fourier_coeffs))
end

# GPU problems keep their nodes in `device_state`, CPU-device problems in the
# host `contours` vector. Each uses its corresponding packing path.
function _ka_energy(prob::ContourProblem, dev::AbstractDevice)
    return _ka_energy_from_state(_storage_data(_active_storage(prob)), prob.kernel,
                                prob.domain, dev; workspace=execution_workspace(prob))
end

# The kernel/domain-specific pieces of the single-layer device energy live in
# three small traits so the launch-argument lists are written exactly once and
# shared by both energy sources (host contours and device workspace):
#   * `_unbounded_energy_recipe(kernel)`  -> (kernel!, trailing args)
#   * `_periodic_energy_recipe(...)`      -> (kernel!, trailing args)
#   * `_periodic_energy_zero_mode(...)`   -> k=0 adjustment after normalization
@inline _unbounded_energy_recipe(::EulerKernel) = (_euler_energy_ka!, ())
@inline _unbounded_energy_recipe(kernel::QGKernel) = (_qg_energy_ka!, (kernel.Ld,))
@inline _unbounded_energy_recipe(kernel::SQGKernel) = (_sqg_energy_ka!, (kernel.δ,))

@inline _periodic_energy_recipe(::EulerKernel, domain::PeriodicDomain{T},
                                cache::EwaldCache{T}, tables) where {T} =
    (_periodic_euler_energy_ka!, (domain.Lx, domain.Ly, tables[1], tables[2]))
@inline function _periodic_energy_recipe(kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                         cache::EwaldCache{T}, tables) where {T}
    kappa2 = one(T) / (kernel.Ld * kernel.Ld)
    area = T(4) * domain.Lx * domain.Ly
    return (_periodic_qg_energy_ka!, (kappa2, area, tables[1], tables[2]))
end
@inline _periodic_energy_recipe(kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                cache::EwaldCache{T}, tables) where {T} =
    (_periodic_sqg_energy_ka!, (cache.α, kernel.δ, domain.Lx, domain.Ly,
                                cache.n_images, tables[1], tables[2], tables[3]))

# `circulation_fn` is a thunk so only the kernels whose zero mode needs the
# circulation pay for it (on the GPU path it is a device reduction).
@inline _periodic_energy_zero_mode(::EulerKernel, domain::PeriodicDomain{T},
                                   cache, circulation_fn::F) where {T, F} = zero(T)
@inline function _periodic_energy_zero_mode(kernel::QGKernel{T},
                                            domain::PeriodicDomain{T},
                                            cache, circulation_fn::F) where {T, F}
    γ = circulation_fn()
    kappa2 = one(T) / (kernel.Ld * kernel.Ld)
    area = T(4) * domain.Lx * domain.Ly
    return γ * γ / (T(2) * area * kappa2)
end
@inline function _periodic_energy_zero_mode(kernel::SQGKernel{T},
                                            domain::PeriodicDomain{T},
                                            cache::EwaldCache{T},
                                            circulation_fn::F) where {T, F}
    γ = circulation_fn()
    return -_sqg_periodic_ewald_zero_mode(cache, domain, kernel.δ) * γ * γ / T(2)
end

function _ka_energy_from_state(src::Vector{PVContour{T}},
                               kernel::_PeriodicPointKernel{T},
                               ::UnboundedDomain, dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    kernel!, args = _unbounded_energy_recipe(kernel)
    return _normalize_energy(_ka_energy_raw(kernel!, src, dev, T, args...))
end

function _ka_energy_from_state(src::Vector{PVContour{T}},
                               kernel::_PeriodicPointKernel{T},
                               domain::PeriodicDomain{T},
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    cache = _get_ewald_cache(domain, kernel)
    data = _pack_energy_segments(src, dev, T)
    length(data.ax) == 0 && return zero(T)
    tables = _device_ewald_tables(cache, dev)
    kernel!, args = _periodic_energy_recipe(kernel, domain, cache, tables)
    raw = _ka_energy_raw_with_segments!(kernel!, data, dev, T, args...)
    return _normalize_energy(raw) + _periodic_energy_zero_mode(
        kernel, domain, cache, () -> _energy_contour_circulation(src))
end

function _ka_energy_from_state(state::DeviceContourState{T}, kernel,
                               domain::AbstractDomain,
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    ws = _get_energy_workspace(dev, T, length(state.lengths), length(state.x); workspace=workspace)
    return _ka_energy_state_with_ws(state, kernel, domain, dev, ws)
end

function _ka_energy_state_with_ws(state::DeviceContourState{T},
                                  kernel::_PeriodicPointKernel{T},
                                  ::UnboundedDomain, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    kernel!, args = _unbounded_energy_recipe(kernel)
    n = _pack_energy_workspace!(ws, state, dev)
    return _normalize_energy(
        _ka_energy_raw_with_workspace!(kernel!, ws, n, dev, args...))
end

function _ka_energy_state_with_ws(state::DeviceContourState{T},
                                  kernel::_PeriodicPointKernel{T},
                                  domain::PeriodicDomain{T}, dev::AbstractDevice,
                                  ws::_EnergyWorkspace{T}) where {T}
    cache = _get_ewald_cache(domain, kernel)
    n = _pack_energy_workspace!(ws, state, dev)
    n == 0 && return zero(T)
    tables = _ensure_energy_ewald!(ws, cache, dev)
    kernel!, args = _periodic_energy_recipe(kernel, domain, cache, tables)
    raw = _ka_energy_raw_with_workspace!(kernel!, ws, n, dev, args...)
    return _normalize_energy(raw) + _periodic_energy_zero_mode(
        kernel, domain, cache, () -> _state_circulation(state, dev))
end

# ── Multi-layer modal energy ─────────────────────────────────────────────

mutable struct _MultilayerEnergyWorkspace{
        T, EW<:_EnergyWorkspace{T}, DA<:AbstractVector{T}}
    energy::EW
    base_pv::DA
end

const _MULTILAYER_ENERGY_WS_KEY = :contourdynamics_multilayer_energy_workspace

function _create_multilayer_energy_workspace(dev::AbstractDevice, ::Type{T},
                                             max_contours::Int,
                                             total_nodes::Int) where {T}
    energy = _create_energy_workspace(dev, T, max_contours, total_nodes)
    base_pv = device_zeros(dev, T, total_nodes)
    return _MultilayerEnergyWorkspace{T,typeof(energy),typeof(base_pv)}(
        energy, base_pv)
end

function _get_multilayer_energy_workspace(
        states::NTuple{N, <:DeviceContourState{T}}, dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N, T}
    max_contours = maximum(s -> length(s.lengths), states; init=0)
    total_nodes = sum(s -> length(s.x), states; init=0)
    store = workspace.buffers
    key = (_MULTILAYER_ENERGY_WS_KEY, T, typeof(dev))
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
    for layer in 1:N
        state = states[layer]
        n = _pack_energy_workspace!(
            ws.energy, state, dev;
            output_offset)
        layer_lengths[layer] = n
        if n > 0
            @_ka_launch dev n _copy_multilayer_base_pv_kernel!(
                ws.base_pv, ws.energy.pv, output_offset, n)
        end
        output_offset += n
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
                                           domain::Union{UnboundedDomain,PeriodicDomain{T}},
                                           dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N, T}
    ws = _get_multilayer_energy_workspace(states, dev; workspace=workspace)
    return _ka_multilayer_energy_with_ws(states, kernel, domain, dev, ws)
end

@inline function _ka_unbounded_modal_energy(mode_kernel, energy_ws, total, dev)
    kernel!, args = _unbounded_energy_recipe(mode_kernel)
    return _ka_energy_raw_with_workspace!(kernel!, energy_ws, total, dev, args...)
end

function _ka_multilayer_energy_with_ws(
        states::NTuple{N, <:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, ::UnboundedDomain,
        dev::AbstractDevice,
        ws::_MultilayerEnergyWorkspace{T}) where {N,T}
    evals = kernel.eigenvalues
    to_modal = kernel.physical_to_modal
    layer_lengths, total = _pack_multilayer_energy_workspace!(ws, states, dev)
    total == 0 && return zero(T)
    raw = zero(T)
    for mode in 1:N
        weights = ntuple(layer -> T(to_modal[mode, layer]), Val(N))
        energy_ws = _apply_multilayer_modal_pv!(ws, layer_lengths, weights, dev)
        lam = evals[mode]
        raw += _dispatch_qg_mode(
            _ka_unbounded_modal_energy, kernel, lam, energy_ws, total, dev)
    end
    return _normalize_energy(raw)
end

@inline function _ka_periodic_modal_energy(
        ::EulerKernel, energy_ws, total, dev, cache, domain,
        kx, ky, to_modal, mode, layer_circulation, area)
    raw = _ka_energy_raw_with_workspace!(
        _periodic_euler_energy_ka!, energy_ws, total, dev,
        domain.Lx, domain.Ly, kx, ky)
    return raw, zero(area)
end

@inline function _ka_periodic_modal_energy(
        mode_kernel::QGKernel{T}, energy_ws, total, dev, cache, domain,
        kx, ky, to_modal, mode,
        layer_circulation, area) where {T}
    kappa2 = inv(mode_kernel.Ld * mode_kernel.Ld)
    raw = _ka_energy_raw_with_workspace!(
        _periodic_qg_energy_ka!, energy_ws, total, dev,
        kappa2, area, kx, ky)
    zero_energy = _periodic_modal_zero_energy(
        mode_kernel, to_modal, mode, layer_circulation, area)
    return raw, zero_energy
end

function _ka_multilayer_energy_with_ws(
        states::NTuple{N, <:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, domain::PeriodicDomain{T},
        dev::AbstractDevice,
        ws::_MultilayerEnergyWorkspace{T}) where {N,T}
    evals = kernel.eigenvalues
    to_modal = kernel.physical_to_modal
    # Every mode uses the same Fourier grid; its kernel differs only through
    # the modal eigenvalue in the denominator.
    cache = _get_ewald_cache(domain, EulerKernel())
    kx, ky, _ = _ensure_energy_ewald!(ws.energy, cache, dev)
    area = T(4) * domain.Lx * domain.Ly
    layer_lengths, total = _pack_multilayer_energy_workspace!(ws, states, dev)
    total == 0 && return zero(T)
    raw = zero(T)
    zero_energy = zero(T)
    layer_circulation = ntuple(
        layer -> _state_circulation(states[layer], dev), Val(N))
    for mode in 1:N
        weights = ntuple(layer -> T(to_modal[mode, layer]), Val(N))
        energy_ws = _apply_multilayer_modal_pv!(ws, layer_lengths, weights, dev)
        lam = evals[mode]
        raw_mode, mode_zero = _dispatch_qg_mode(
            _ka_periodic_modal_energy, kernel, lam, energy_ws, total, dev,
            cache, domain, kx, ky, to_modal, mode,
            layer_circulation, area)
        raw += raw_mode
        zero_energy += mode_zero
    end
    return _normalize_energy(raw) + zero_energy
end
