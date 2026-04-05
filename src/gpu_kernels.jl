# GPU-portable velocity kernels using KernelAbstractions.jl.
# These work on both CPU (KernelAbstractions.CPU backend) and
# GPU (CUDABackend via the CUDA extension).

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
end

# ── GPU workspace for buffer reuse ───────────────────
# Avoids repeated allocation of CPU packing buffers and device arrays
# across the 4 velocity evaluations per RK4 step.

mutable struct _GPUWorkspace{T, DA<:AbstractVector{T}}
    # CPU packing buffers (filled each call, then copied to device)
    cpu_ax::Vector{T}; cpu_ay::Vector{T}
    cpu_bx::Vector{T}; cpu_by::Vector{T}
    cpu_pv::Vector{T}
    cpu_tx::Vector{T}; cpu_ty::Vector{T}
    # Device arrays — parameterized so field access is type-stable
    # (Vector{T} on CPU, CuVector{T} on GPU via the CUDA extension).
    dev_ax::DA; dev_ay::DA
    dev_bx::DA; dev_by::DA
    dev_pv::DA
    dev_tx::DA; dev_ty::DA
    dev_vel_x::DA; dev_vel_y::DA
    # CPU copy-back buffers
    cpu_vx::Vector{T}; cpu_vy::Vector{T}
    n::Int
end

function _create_gpu_workspace(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    da = device_zeros(dev, T, N)  # probe device array type
    DA = typeof(da)
    _GPUWorkspace{T, DA}(
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_ax, cpu_ay
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_bx, cpu_by
        Vector{T}(undef, N),                        # cpu_pv
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_tx, cpu_ty
        da, device_zeros(dev, T, N),                # dev_ax, dev_ay
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_bx, dev_by
        device_zeros(dev, T, N),                            # dev_pv
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_tx, dev_ty
        device_zeros(dev, T, N), device_zeros(dev, T, N),  # dev_vel_x, dev_vel_y
        Vector{T}(undef, N), Vector{T}(undef, N),  # cpu_vx, cpu_vy
        N,
    )
end

# Module-level workspace cache (one workspace at a time).
const _gpu_ws_ref = Ref{Union{Nothing, _GPUWorkspace}}(nothing)
const _gpu_ws_lock = ReentrantLock()

function _get_gpu_workspace!(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    lock(_gpu_ws_lock) do
        ws = _gpu_ws_ref[]
        if ws isa _GPUWorkspace{T, <:AbstractVector{T}} && ws.n == N
            return ws::_GPUWorkspace{T}
        end
        new_ws = _create_gpu_workspace(dev, T, N)
        _gpu_ws_ref[] = new_ws
        return new_ws
    end
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
    _fill_segment_bufs!(ax, ay, bx, by, pv_vec, prob)
    SegmentData(
        to_device(dev, ax), to_device(dev, ay),
        to_device(dev, bx), to_device(dev, by),
        to_device(dev, pv_vec)
    )
end

# Shared logic for filling CPU segment buffers.
function _fill_segment_bufs!(ax, ay, bx, by, pv_vec, prob)
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
                idx += 1
            end
        else
            for j in 1:nc
                a = c.nodes[j]
                b = next_node(c, j)
                ax[idx] = a[1]; ay[idx] = a[2]
                bx[idx] = b[1]; by[idx] = b[2]
                pv_vec[idx] = c.pv
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

"""KernelAbstractions kernel: each workitem computes velocity at one target node."""
@kernel function _euler_velocity_ka!(vel_x, vel_y,
                                      target_x, target_y,
                                      seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                      n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv4pi = one(T) / (4 * T(π))

    for j in 1:n_seg
        dsx = seg_bx[j] - seg_ax[j]
        dsy = seg_by[j] - seg_ay[j]
        ds_len_sq = dsx^2 + dsy^2
        ds_len = sqrt(ds_len_sq)
        ds_len < eps(T) && continue

        tx = dsx / ds_len
        ty = dsy / ds_len
        nx = -ty
        ny = tx

        r0x = xi - seg_ax[j]
        r0y = yi - seg_ay[j]
        u_a = r0x * tx + r0y * ty
        h   = r0x * nx + r0y * ny
        u_b = u_a - ds_len
        h_sq = h * h

        F_diff = _euler_antideriv_scalar(u_a, h, h_sq) - _euler_antideriv_scalar(u_b, h, h_sq)
        contrib = -inv4pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

"""
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev)

Launch the KA Euler velocity kernel on the given device.
"""
function _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev::AbstractDevice)
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    kernel = _euler_velocity_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _gpu_velocity!(vel_x, vel_y, prob::ContourProblem, dev::AbstractDevice)

Full GPU velocity evaluation: pack segments, pack targets, launch kernel.
Allocates fresh buffers each call (use `_gpu_velocity_ws!` for buffer reuse).
"""
function _gpu_velocity!(vel_x, vel_y, prob::ContourProblem, dev::AbstractDevice)
    seg = pack_segments(prob, dev)
    target_x, target_y = pack_targets(prob, dev)
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, dev)
    return nothing
end

"""
    _gpu_velocity_ws!(ws::_GPUWorkspace, prob::ContourProblem, dev::AbstractDevice)

GPU velocity evaluation using pre-allocated workspace buffers.
Fills CPU packing buffers, copies to device via `copyto!`, launches kernel,
and copies results back — all without allocating new arrays.
"""
function _gpu_velocity_ws!(ws::_GPUWorkspace{T}, prob::ContourProblem{K,D,T}, dev::AbstractDevice) where {K,D,T}
    N = total_nodes(prob)

    # Fill CPU packing buffers (no allocation)
    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    # Transfer to device (copyto! reuses pre-allocated device arrays)
    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    # Launch kernel (no allocation)
    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    _ka_euler_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty, seg, dev)

    # Copy results back to CPU (no allocation)
    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end
