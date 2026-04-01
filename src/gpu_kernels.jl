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
    idx = 1
    for c in prob.contours
        nc = nnodes(c)
        nc < 2 && continue
        for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            ax[idx] = a[1]; ay[idx] = a[2]
            bx[idx] = b[1]; by[idx] = b[2]
            pv_vec[idx] = c.pv
            idx += 1
        end
    end
    n_seg = idx - 1
    SegmentData(
        to_device(dev, ax[1:n_seg]), to_device(dev, ay[1:n_seg]),
        to_device(dev, bx[1:n_seg]), to_device(dev, by[1:n_seg]),
        to_device(dev, pv_vec[1:n_seg])
    )
end

"""
    pack_targets(prob::ContourProblem, dev::AbstractDevice)

Pack all target node positions into flat x/y arrays on the given device.
"""
function pack_targets(prob::ContourProblem{K,D,T}, dev::AbstractDevice) where {K,D,T}
    N = total_nodes(prob)
    tx = Vector{T}(undef, N)
    ty = Vector{T}(undef, N)
    idx = 1
    for c in prob.contours
        nc = nnodes(c)
        nc < 2 && continue
        for j in 1:nc
            tx[idx] = c.nodes[j][1]
            ty[idx] = c.nodes[j][2]
            idx += 1
        end
    end
    n_targets = idx - 1
    (to_device(dev, tx[1:n_targets]), to_device(dev, ty[1:n_targets]))
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
"""
function _gpu_velocity!(vel_x, vel_y, prob::ContourProblem, dev::AbstractDevice)
    seg = pack_segments(prob, dev)
    target_x, target_y = pack_targets(prob, dev)
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, dev)
    return nothing
end
