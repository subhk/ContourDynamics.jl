# KA velocity: data packing and workspace.
#
# This is the entry file for the KernelAbstractions (KA) velocity subsystem,
# split across three files:
#   - packing.jl  (this file) — flat data layout and workspace buffers
#   - kernels.jl              — scalar helpers and the `@kernel` velocity kernels
#   - velocity.jl             — launch wrappers, dispatch, and `_ka_velocity!`
#
# Before a KA kernel can run, contours are packed into a flat segment
# structure-of-arrays layout (`SegmentData`) plus flat target-node arrays;
# kernels compute one target velocity per work item, then the result is copied
# back into the caller's velocity vectors. The same code runs on
# `KernelAbstractions.CPU()` (used to validate the kernels in tests) and on CUDA
# via the package extension. On CPU the public `velocity!` uses the direct
# reference path; this subsystem is the single implementation reused by the GPU
# backend.

"""
    SegmentData{A}

Structure-of-Arrays layout for contour segments, suitable for GPU coalesced access.
Each field is a flat vector of length `total_segments`.
"""
struct SegmentData{A<:AbstractVector}
    ax::A   # segment start x
    ay::A   # segment start y
    bx::A   # segment end x
    by::A   # segment end y
    pv::A   # PV jump for this segment
    ka::A   # signed curvature at segment start
    kb::A   # signed curvature at segment end
end

@inline _state_norm2(dx::T, dy::T) where {T} = sqrt(dx * dx + dy * dy)

@inline function _state_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                              lengths, corners, ci::Int, li::Int)
    T = eltype(x)
    n = lengths[ci]
    n < 3 && return zero(T)

    off = offsets[ci]
    prev_i = li == 1 ? n : li - 1
    next_i = li == n ? 1 : li + 1
    (!iszero(corners[off + prev_i - 1]) ||
     !iszero(corners[off + li - 1]) ||
     !iszero(corners[off + next_i - 1])) && return zero(T)

    g = off + li - 1
    prev_x = li == 1 ? x[off + n - 1] - wrapx[ci] : x[g - 1]
    prev_y = li == 1 ? y[off + n - 1] - wrapy[ci] : y[g - 1]
    curr_x = x[g]
    curr_y = y[g]
    next_x = li == n ? x[off] + wrapx[ci] : x[g + 1]
    next_y = li == n ? y[off] + wrapy[ci] : y[g + 1]

    ax = curr_x - prev_x
    ay = curr_y - prev_y
    bx = next_x - curr_x
    by = next_y - curr_y
    chord_x = next_x - prev_x
    chord_y = next_y - prev_y
    denom = _state_norm2(ax, ay) * _state_norm2(bx, by) *
            _state_norm2(chord_x, chord_y)
    denom <= eps(T) && return zero(T)
    return T(2) * (ax * by - ay * bx) / denom
end

@kernel function _state_segment_data_kernel!(ax, ay, bx, by, seg_pv, ka, kb,
                                             x, y, pv, wrapx, wrapy, offsets,
                                             lengths, corners, contour_of_node,
                                             local_index, pv_scale, total_nodes)
    g = @index(Global)
    if g <= total_nodes
        T = eltype(x)
        ci = contour_of_node[g]
        li = local_index[g]
        n = lengths[ci]
        off = offsets[ci]

        ax[g] = x[g]
        ay[g] = y[g]
        seg_pv[g] = pv_scale * pv[ci]

        if n < 2
            bx[g] = x[g]
            by[g] = y[g]
            ka[g] = zero(T)
            kb[g] = zero(T)
        else
            if li < n
                bx[g] = x[g + 1]
                by[g] = y[g + 1]
            else
                bx[g] = x[off] + wrapx[ci]
                by[g] = y[off] + wrapy[ci]
            end
            ka[g] = _state_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                                 lengths, corners, ci, li)
            next_li = li == n ? 1 : li + 1
            kb[g] = _state_signed_node_curvature(x, y, wrapx, wrapy, offsets,
                                                 lengths, corners, ci, next_li)
        end
    end
end

function _state_segment_data(state::DeviceContourState{T},
                             dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    ax = device_zeros(dev, T, N)
    ay = device_zeros(dev, T, N)
    bx = device_zeros(dev, T, N)
    by = device_zeros(dev, T, N)
    pv = device_zeros(dev, T, N)
    ka = device_zeros(dev, T, N)
    kb = device_zeros(dev, T, N)
    N == 0 && return SegmentData(ax, ay, bx, by, pv, ka, kb)

    @_ka_launch dev N _state_segment_data_kernel!(
        ax, ay, bx, by, pv, ka, kb,
        state.x, state.y, state.pv, state.wrapx, state.wrapy,
        state.offsets, state.lengths, state.corners,
        state.contour_of_node, state.local_index, one(T), N)
    return SegmentData(ax, ay, bx, by, pv, ka, kb)
end

@kernel function _copy_velocity_svector_kernel!(vel, vel_x, vel_y, n)
    i = @index(Global)
    if i <= n
        T = eltype(vel_x)
        vel[i] = SVector{2,T}(vel_x[i], vel_y[i])
    end
end

@kernel function _modal_accumulate_ka!(vel_x, vel_y, mode_vx, mode_vy, w, n)
    i = @index(Global)
    if i <= n
        vel_x[i] += w * mode_vx[i]
        vel_y[i] += w * mode_vy[i]
    end
end

# ── KA workspace (CPU-backend packing buffers) ───────────────────
# Bundles the CPU packing buffers and device arrays for one packed problem size.
# Used by the CPU KA velocity path, which exists to validate the KA kernels
# against the scalar evaluator in tests; the GPU path packs from its device-
# resident state instead.

mutable struct _GPUWorkspace{T, DA<:AbstractVector{T}, DMA<:AbstractMatrix{T}}
    # CPU packing buffers (filled each call, then copied to device)
    cpu_ax::Vector{T}; cpu_ay::Vector{T}
    cpu_bx::Vector{T}; cpu_by::Vector{T}
    cpu_pv::Vector{T}
    cpu_ka::Vector{T}; cpu_kb::Vector{T}
    cpu_tx::Vector{T}; cpu_ty::Vector{T}
    # Device arrays — parameterized so field access is type-stable
    # (Vector{T} on CPU, CuVector{T} on GPU via the CUDA extension).
    dev_ax::DA; dev_ay::DA
    dev_bx::DA; dev_by::DA
    dev_pv::DA
    dev_ka::DA; dev_kb::DA
    dev_tx::DA; dev_ty::DA
    dev_vel_x::DA; dev_vel_y::DA
    dev_ewald_kx::DA; dev_ewald_ky::DA
    dev_ewald_fourier::DMA
    # CPU copy-back buffers
    cpu_vx::Vector{T}; cpu_vy::Vector{T}
    last_ewald::Union{Nothing, EwaldCache{T}}
    n::Int
end

function _create_gpu_workspace(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    # Allocate all CPU and device buffers required for one packed problem size.
    # A probe allocation captures the concrete device array type, making the
    # mutable workspace fields concrete and type-stable.
    da = device_zeros(dev, T, N)  # probe device array type
    DA = typeof(da)
    dev_fourier = device_zeros(dev, T, 0, 0)
    DMA = typeof(dev_fourier)
    _GPUWorkspace{T, DA, DMA}(
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_ax, cpu_ay
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_bx, cpu_by
        Vector{T}(undef, N),                        # cpu_pv
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_ka, cpu_kb
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_tx, cpu_ty
        da, device_zeros(dev, T, N),                # dev_ax, dev_ay
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_bx, dev_by
        device_zeros(dev, T, N),                            # dev_pv
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_ka, dev_kb
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_tx, dev_ty
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_vel_x, dev_vel_y
        device_zeros(dev, T, 0), device_zeros(dev, T, 0),  # dev_ewald_kx, dev_ewald_ky
        dev_fourier,                                        # dev_ewald_fourier
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_vx, cpu_vy
        nothing,                                   # last_ewald
        N,
    )
end

# Reused workspace for the device-resident MULTI-LAYER modal velocity path.
# Mirrors `_GPUWorkspace`/`_get_state_workspace` but is sized to the concatenated
# node count `total` and carries the extra per-mode velocity scratch
# (`mode_vx`/`mode_vy`). Reusing it across RK stages avoids reallocating 13 device
# arrays every modal-velocity evaluation (≥4×/RK4 step). Held in TASK-LOCAL
# storage by `_get_multilayer_workspace`, like the single-layer workspace.
mutable struct _MultilayerWorkspace{T, DA<:AbstractVector{T}}
    ax::DA; ay::DA; bx::DA; by::DA; pv::DA; ka::DA; kb::DA  # 7 segment buffers
    tx::DA; ty::DA                                          # concatenated targets
    mode_vx::DA; mode_vy::DA                                # per-mode velocity scratch
    vel_x::DA; vel_y::DA                                    # accumulated flat output
    n::Int
end

function _create_multilayer_workspace(dev::AbstractDevice, ::Type{T}, total::Int) where {T}
    # Probe allocation captures the concrete device array type so every field is
    # concrete and type-stable (Vector{T} on CPU, CuVector{T} on GPU).
    da = device_zeros(dev, T, total)
    DA = typeof(da)
    mk() = device_zeros(dev, T, total)
    _MultilayerWorkspace{T, DA}(da, mk(), mk(), mk(), mk(), mk(), mk(),
                                mk(), mk(), mk(), mk(), mk(), mk(), total)
end

function _ensure_device_ewald!(ws::_GPUWorkspace{T}, cache::EwaldCache{T},
                               dev::AbstractDevice) where {T}
    # Fourier/Ewald data depends only on the domain/kernel cache. Reuse the
    # previous device copy when the same cache object is used across RK stages.
    if ws.last_ewald !== cache
        ws.dev_ewald_kx = to_device(dev, cache.kx)
        ws.dev_ewald_ky = to_device(dev, cache.ky)
        ws.dev_ewald_fourier = to_device(dev, cache.fourier_coeffs)
        ws.last_ewald = cache
    end
    return ws.dev_ewald_kx, ws.dev_ewald_ky, ws.dev_ewald_fourier
end

@inline function _periodic_ewald_data(::Nothing, cache::EwaldCache{T},
                                      dev::AbstractDevice) where {T}
    return to_device(dev, cache.kx),
           to_device(dev, cache.ky),
           to_device(dev, cache.fourier_coeffs)
end

@inline _periodic_ewald_data(ws, cache::EwaldCache{T}, dev::AbstractDevice) where {T} =
    _ensure_device_ewald!(ws, cache, dev)

@inline function _periodic_ewald_vectors(maybe_ws, cache::EwaldCache{T},
                                         dev::AbstractDevice) where {T}
    dev_kx, dev_ky, _ = _periodic_ewald_data(maybe_ws, cache, dev)
    return dev_kx, dev_ky
end

# In-place segment packing into a reused workspace's device buffers (avoids the
# per-evaluation allocation of `_state_segment_data`). The workspace must be
# sized for `state`'s node count. `pv_scale` rescales the per-segment PV weight
# (1 for single-layer; the modal weight for a multi-layer mode).
function _state_segment_data!(ws::_GPUWorkspace{T}, state::DeviceContourState{T},
                              dev::AbstractDevice, pv_scale::T=one(T)) where {T}
    N = _device_state_nnodes(state)
    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by,
                      ws.dev_pv, ws.dev_ka, ws.dev_kb)
    N == 0 && return seg
    @_ka_launch dev N _state_segment_data_kernel!(
        ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv, ws.dev_ka, ws.dev_kb,
        state.x, state.y, state.pv, state.wrapx, state.wrapy,
        state.offsets, state.lengths, state.corners,
        state.contour_of_node, state.local_index, pv_scale, N)
    return seg
end

@inline function _launch_ka_segment_kernel!(kernel_builder,
                                            vel_x, vel_y, target_x, target_y,
                                            seg::SegmentData,
                                            dev::AbstractDevice,
                                            args...)
    # Common launch wrapper used by all single-layer velocity kernels. Keeping
    # launch/synchronize logic here avoids subtle backend differences elsewhere.
    @_ka_launch dev length(target_x) kernel_builder(vel_x, vel_y, target_x, target_y,
                                                    seg.ax, seg.ay, seg.bx, seg.by,
                                                    seg.pv, seg.ka, seg.kb, args...,
                                                    Int32(length(seg.ax)))
    return nothing
end

"""
    pack_segments(prob::ContourProblem, dev::AbstractDevice)

Pack all contour segments into SoA layout. Built on CPU, then transferred to `dev`.
"""
function pack_segments(prob::ContourProblem{K,D,T}, dev::AbstractDevice) where {K,D,T}
    N = total_nodes(prob)
    ax = Vector{T}(undef, N)
    ay = Vector{T}(undef, N)
    bx = Vector{T}(undef, N)
    by = Vector{T}(undef, N)
    pv_vec = Vector{T}(undef, N)
    ka = Vector{T}(undef, N)
    kb = Vector{T}(undef, N)
    _fill_segment_bufs!(ax, ay, bx, by, pv_vec, ka, kb, prob)
    SegmentData(
        to_device(dev, ax), to_device(dev, ay),
        to_device(dev, bx), to_device(dev, by),
        to_device(dev, pv_vec),
        to_device(dev, ka), to_device(dev, kb)
    )
end

# Shared logic for filling CPU segment buffers.
function _fill_segment_bufs!(ax, ay, bx, by, pv_vec, ka, kb, prob)
    # Packing is intentionally CPU-side: contour objects are ragged and pointer
    # rich, while device kernels expect flat SoA arrays with one entry per
    # source segment.
    idx = 1
    for c in prob.contours
        nc = nnodes(c)
        if nc < 2
            # Single-node contours produce a degenerate zero-length segment.
            # Include them to keep segment count aligned with total_nodes.
            for j in 1:nc
                ax[idx] = c.nodes[j][1]; ay[idx] = c.nodes[j][2]
                bx[idx] = c.nodes[j][1]; by[idx] = c.nodes[j][2]
                pv_vec[idx] = c.pv
                ka[idx] = zero(eltype(ka)); kb[idx] = zero(eltype(kb))
                idx += 1
            end
        else
            κ_first = _signed_node_curvature(c, 1)
            κj = κ_first
            for j in 1:nc
                a = c.nodes[j]
                b = next_node(c, j)
                κnext = j == nc ? κ_first : _signed_node_curvature(c, j + 1)
                ax[idx] = a[1]; ay[idx] = a[2]
                bx[idx] = b[1]; by[idx] = b[2]
                pv_vec[idx] = c.pv
                ka[idx] = κj
                kb[idx] = κnext
                κj = κnext
                idx += 1
            end
        end
    end
    return idx - 1
end

"""
    pack_targets(prob::ContourProblem, dev::AbstractDevice)

Pack all target node positions into flat x/y arrays on the given device.
"""
function pack_targets(prob::ContourProblem{K,D,T}, dev::AbstractDevice) where {K,D,T}
    N = total_nodes(prob)
    tx = Vector{T}(undef, N)
    ty = Vector{T}(undef, N)
    _fill_target_bufs!(tx, ty, prob)
    (to_device(dev, tx), to_device(dev, ty))
end

# Shared logic for filling CPU target buffers.
function _fill_target_bufs!(tx, ty, prob)
    # Target nodes are packed in the same contour order used by velocity! copy
    # back, so the flat kernel output can be written directly into vel.
    idx = 1
    for c in prob.contours
        for j in 1:nnodes(c)
            tx[idx] = c.nodes[j][1]
            ty[idx] = c.nodes[j][2]
            idx += 1
        end
    end
    return idx - 1
end

