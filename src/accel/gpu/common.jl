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

# Module-level workspace cache keyed by (T, N, Dev) so problems with different
# sizes or devices don't collide.  Each entry carries its own lock so that
# concurrent velocity evaluations on the *same* workspace block rather than
# racing on shared buffers.
struct _WSEntry{T, DA<:AbstractVector{T}}
    ws::_GPUWorkspace{T, DA}
    lock::ReentrantLock
end
const _gpu_ws_cache = Dict{Tuple{DataType, DataType}, Any}()
const _gpu_ws_key_order = Dict{Tuple{DataType, DataType}, Any}()
const _gpu_ws_cache_lock = ReentrantLock()
const _GPU_WS_CACHE_MAX = 8

function _gpu_ws_dict(::Type{T}, ::Type{Dev}) where {T, Dev<:AbstractDevice}
    key = (T, Dev)
    cache = get!(_gpu_ws_cache, key) do
        Dict{Int, _WSEntry{T}}()
    end
    return cache::Dict{Int, _WSEntry{T}}
end

function _gpu_ws_order(::Type{T}, ::Type{Dev}) where {T, Dev<:AbstractDevice}
    key = (T, Dev)
    order = get!(_gpu_ws_key_order, key) do
        Int[]
    end
    return order::Vector{Int}
end

function _get_gpu_workspace!(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    Dev = typeof(dev)
    caches = _gpu_ws_dict(T, Dev)
    order = _gpu_ws_order(T, Dev)

    cached = lock(_gpu_ws_cache_lock) do
        get(caches, N, nothing)
    end
    cached !== nothing && return cached::_WSEntry{T}

    new_entry = _WSEntry(_create_gpu_workspace(dev, T, N), ReentrantLock())

    lock(_gpu_ws_cache_lock) do
        existing = get(caches, N, nothing)
        if existing !== nothing
            return existing::_WSEntry{T}
        end
        while length(caches) >= _GPU_WS_CACHE_MAX && !isempty(order)
            old_n = popfirst!(order)
            delete!(caches, old_n)
        end
        caches[N] = new_entry
        push!(order, N)
        return new_entry
    end
end

"""Build the cache key for a ContourProblem's workspace."""
_gpu_ws_key(prob::ContourProblem{K,D,T,Dev}, N::Int) where {K,D,T,Dev} = (T, Dev, N)

"""
    _evict_gpu_workspace!(key::Tuple{DataType, DataType, Int})

Remove a specific cached workspace entry by its exact key `(T, DevType, N)`.
Called after surgery changes the node count so stale entries don't leak memory.
"""
function _evict_gpu_workspace!(key::Tuple{DataType, DataType, Int})
    T, Dev, N = key
    lock(_gpu_ws_cache_lock) do
        caches = get(_gpu_ws_cache, (T, Dev), nothing)
        order = get(_gpu_ws_key_order, (T, Dev), nothing)
        caches === nothing || delete!(caches, N)
        if order !== nothing
            filter!(!=(N), order)
        end
    end
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

# Modified Bessel K₀ approximation that uses only device-friendly elementary
# functions. The small-argument branch mirrors the CPU singular-subtraction path,
# while the larger-argument branch uses the standard piecewise approximation from
# Numerical Recipes.
@inline function _besselk0_approx_scalar(x::T) where {T}
    ax = abs(x)
    ax < eps(T) && return T(Inf)

    if ax <= T(2)
        y = (ax * ax) / T(4)
        return -log(ax / T(2)) * _i0_approx_scalar(ax) +
            (-T(0.57721566) +
             y * (T(0.42278420) +
             y * (T(0.23069756) +
             y * (T(0.03488590) +
             y * (T(0.00262698) +
             y * (T(0.00010750) +
                  y * T(0.00000740)))))))
    end

    y = T(2) / ax
    poly = T(1.25331414) +
           y * (-T(0.07832358) +
           y * (T(0.02189568) +
           y * (-T(0.01062446) +
           y * (T(0.00587872) +
           y * (-T(0.00251540) +
                y * T(0.00053208))))))
    return exp(-ax) / sqrt(ax) * poly
end

@inline function _i0_approx_scalar(x::T) where {T}
    ax = abs(x)
    if ax < T(3.75)
        y = (ax / T(3.75))^2
        return one(T) +
            y * (T(3.5156229) +
            y * (T(3.0899424) +
            y * (T(1.2067492) +
            y * (T(0.2659732) +
            y * (T(0.0360768) +
                 y * T(0.0045813))))))
    end

    y = T(3.75) / ax
    poly = T(0.39894228) +
           y * (T(0.01328592) +
           y * (T(0.00225319) +
           y * (-T(0.00157565) +
           y * (T(0.00916281) +
           y * (-T(0.02057706) +
           y * (T(0.02635537) +
           y * (-T(0.01647633) +
                y * T(0.00392377))))))))
    return exp(ax) / sqrt(ax) * poly
end

@inline function _qg_smooth_correction_scalar(rr::T, r::T, Ld::T) where {T}
    if rr < T(0.5)
        return _besselk0_correction(rr) + log(T(2) * Ld) - T(Base.MathConstants.eulergamma)
    end
    return _besselk0_approx_scalar(rr) + log(r)
end

@inline function _minimum_image_scalar(dx::T, L::T) where {T}
    period = T(2) * L
    return dx - round(dx / period) * period
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

    @inbounds for j in 1:n_seg
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

@kernel function _periodic_euler_velocity_ka!(vel_x, vel_y,
                                              target_x, target_y,
                                              seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
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
    g_nodes, g_weights = _gl3_nodes_weights(T)

    @inbounds for j in 1:n_seg
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
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len
        h_sq = h * h

        F_diff = _euler_antideriv_scalar(u_a, h, h_sq) - _euler_antideriv_scalar(u_b, h, h_sq)
        contrib = -inv4pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty

        mid_x = (seg_ax[j] + seg_bx[j]) / T(2)
        mid_y = (seg_ay[j] + seg_by[j]) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in 1:3
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy

            r0x_raw = xi - sx
            r0y_raw = yi - sy
            r0x_wrap = _minimum_image_scalar(r0x_raw, Lx)
            r0y_wrap = _minimum_image_scalar(r0y_raw, Ly)
            G_corr = zero(T)

            for px in -n_images:n_images
                shiftx = T(2) * Lx * T(px)
                for py in -n_images:n_images
                    shifty = T(2) * Ly * T(py)
                    rx = r0x_wrap - shiftx
                    ry = r0y_wrap - shifty
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

            rx = r0x_wrap
            ry = r0y_wrap
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
                                  Ld, n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv2pi = one(T) / (T(2) * T(pi))
    g_nodes, g_weights = _gl5_nodes_weights(T)

    @inbounds for j in 1:n_seg
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
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len
        h_sq = h * h

        F_diff = _euler_antideriv_scalar(u_a, h, h_sq) - _euler_antideriv_scalar(u_b, h, h_sq)
        contrib = -T(0.5) * inv2pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty

        mid_x = (seg_ax[j] + seg_bx[j]) / T(2)
        mid_y = (seg_ay[j] + seg_by[j]) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in 1:5
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy
            rx = sx - xi
            ry = sy - yi
            r2 = rx * rx + ry * ry

            if r2 < eps(T)^2
                corr_integral += g_weights[q] * (log(T(2) * Ld) - T(Base.MathConstants.eulergamma))
                continue
            end

            r = sqrt(r2)
            rr = r / Ld
            corr_integral += g_weights[q] * _qg_smooth_correction_scalar(rr, r, Ld)
        end

        corr_integral *= T(0.5)
        corr = inv2pi * seg_pv[j] * corr_integral
        vx += corr * dsx
        vy += corr * dsy
    end

    vel_x[i] = vx
    vel_y[i] = vy
end

@kernel function _periodic_qg_correction_ka!(vel_x, vel_y,
                                             target_x, target_y,
                                             seg_ax, seg_ay, seg_bx, seg_by, seg_pv,
                                             Ld, Lx, Ly, kx, ky,
                                             n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    kappa2 = one(T) / (Ld * Ld)
    area = T(4) * Lx * Ly
    g_nodes, g_weights = _gl3_nodes_weights(T)
    vx = vel_x[i]
    vy = vel_y[i]

    @inbounds for j in 1:n_seg
        dsx = seg_bx[j] - seg_ax[j]
        dsy = seg_by[j] - seg_ay[j]
        ds_len_sq = dsx^2 + dsy^2
        ds_len = sqrt(ds_len_sq)
        ds_len < eps(T) && continue

        mid_x = (seg_ax[j] + seg_bx[j]) / T(2)
        mid_y = (seg_ay[j] + seg_by[j]) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in 1:3
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
                                            alpha, delta, Lx, Ly, n_images,
                                            kx, ky, fourier_coeffs,
                                            n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    delta_sq = delta * delta
    inv2pi = one(T) / (T(2) * T(pi))
    g_nodes, g_weights = _gl3_nodes_weights(T)
    vx = zero(T)
    vy = zero(T)

    @inbounds for j in 1:n_seg
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
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len

        h_eff = sqrt(h * h + delta_sq)
        F_diff = asinh(u_a / h_eff) - asinh(u_b / h_eff)
        contrib = -inv2pi * seg_pv[j] * F_diff
        vx += contrib * tx
        vy += contrib * ty

        mid_x = (seg_ax[j] + seg_bx[j]) / T(2)
        mid_y = (seg_ay[j] + seg_by[j]) / T(2)
        half_dsx = dsx / T(2)
        half_dsy = dsy / T(2)
        corr_integral = zero(T)

        for q in 1:3
            sx = mid_x + g_nodes[q] * half_dsx
            sy = mid_y + g_nodes[q] * half_dsy

            r0x_raw = xi - sx
            r0y_raw = yi - sy
            r0x_wrap = _minimum_image_scalar(r0x_raw, Lx)
            r0y_wrap = _minimum_image_scalar(r0y_raw, Ly)
            G_corr = zero(T)

            for px in -n_images:n_images
                shiftx = T(2) * Lx * T(px)
                for py in -n_images:n_images
                    shifty = T(2) * Ly * T(py)
                    rx = r0x_wrap - shiftx
                    ry = r0y_wrap - shifty
                    r2 = rx * rx + ry * ry

                    if px == 0 && py == 0
                        r_reg = sqrt(r2 + delta_sq)
                        G_corr += inv2pi * erf(alpha * r_reg) / r_reg
                    elseif r2 > eps(T)
                        r = sqrt(r2)
                        G_corr -= inv2pi * erfc(alpha * r) / r
                    end
                end
            end

            rx = r0x_wrap
            ry = r0y_wrap
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
                    G_corr -= inv2pi * coeff * (cx * cos(kyi * ry) - sx_trig * sin(kyi * ry))
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
                                   delta, n_seg)
    i = @index(Global)
    T = eltype(vel_x)
    xi = target_x[i]
    yi = target_y[i]
    vx = zero(T)
    vy = zero(T)
    inv2pi = one(T) / (2 * T(pi))

    @inbounds for j in 1:n_seg
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
        h = r0x * nx + r0y * ny
        u_b = u_a - ds_len

        h_eff = sqrt(h * h + delta * delta)
        F_diff = asinh(u_a / h_eff) - asinh(u_b / h_eff)
        contrib = -inv2pi * seg_pv[j] * F_diff
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
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, delta, dev)

Launch the KA SQG velocity kernel on the given device.
"""
function _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                           delta, dev::AbstractDevice)
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    kernel = _sqg_velocity_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           delta, Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, Ld, dev)

Launch the KA QG velocity kernel on the given device.
"""
function _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                          Ld, dev::AbstractDevice)
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    kernel = _qg_velocity_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           Ld, Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev)

Launch the KA periodic Euler velocity kernel on the given device.
"""
function _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                      domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                      dev::AbstractDevice) where {T}
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    dev_kx = to_device(dev, cache.kx)
    dev_ky = to_device(dev, cache.ky)
    dev_fourier = to_device(dev, cache.fourier_coeffs)
    kernel = _periodic_euler_velocity_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           cache.alpha, domain.Lx, domain.Ly, cache.n_images,
           dev_kx, dev_ky, dev_fourier,
           Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache, Ld, dev)

Apply the periodic QG-minus-Euler Fourier correction on top of the periodic
Euler velocity already stored in `vel_x`/`vel_y`.
"""
function _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                     Ld::T, dev::AbstractDevice) where {T}
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    dev_kx = to_device(dev, cache.kx)
    dev_ky = to_device(dev, cache.ky)
    kernel = _periodic_qg_correction_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           Ld, domain.Lx, domain.Ly, dev_kx, dev_ky,
           Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, delta, dev)

Launch the KA periodic SQG velocity kernel on the given device.
"""
function _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                    domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                    delta::T, dev::AbstractDevice) where {T}
    backend = _ka_backend(dev)
    n_targets = length(target_x)
    n_seg = length(seg.ax)
    dev_kx = to_device(dev, cache.kx)
    dev_ky = to_device(dev, cache.ky)
    dev_fourier = to_device(dev, cache.fourier_coeffs)
    kernel = _periodic_sqg_velocity_ka!(backend)
    kernel(vel_x, vel_y, target_x, target_y,
           seg.ax, seg.ay, seg.bx, seg.by, seg.pv,
           cache.alpha, delta, domain.Lx, domain.Ly, cache.n_images,
           dev_kx, dev_ky, dev_fourier,
           Int32(n_seg); ndrange=n_targets)
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg, kernel, domain, dev)

Evaluate a direct velocity subset on flat target/source arrays using the
existing KA kernels. This is used by the hybrid treecode path when a small
direct leaf interaction can be offloaded without rebuilding the full problem.
"""
function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              ::EulerKernel, ::UnboundedDomain,
                              dev::AbstractDevice)
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, dev)
end

function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              kernel::QGKernel{T}, ::UnboundedDomain,
                              dev::AbstractDevice) where {T}
    _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg, kernel.Ld, dev)
end

function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              kernel::SQGKernel{T}, ::UnboundedDomain,
                              dev::AbstractDevice) where {T}
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, kernel.delta, dev)
end

function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              ::EulerKernel, domain::PeriodicDomain{T},
                              dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, EulerKernel())
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev)
end

function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              kernel::QGKernel{T}, domain::PeriodicDomain{T},
                              dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, EulerKernel())
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev)
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache, kernel.Ld, dev)
end

function _ka_velocity_subset!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                              dev::AbstractDevice) where {T}
    cache = _get_ewald_cache(domain, kernel)
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, kernel.delta, dev)
end

"""
    _ka_velocity_ws!(ws::_GPUWorkspace, prob::ContourProblem, dev::AbstractDevice)

KernelAbstractions-based velocity evaluation using pre-allocated workspace
buffers. This supports both CPU and GPU backends through the same packing and
launch path for the kernels that already have flat direct evaluators.
"""
function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{EulerKernel,D,T},
                          dev::AbstractDevice) where {D,T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

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

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{QGKernel{T},UnboundedDomain,T},
                          dev::AbstractDevice) where {T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    _ka_qg_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                     seg, prob.kernel.Ld, dev)

    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{EulerKernel,PeriodicDomain{T},T},
                          dev::AbstractDevice) where {T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    ewald = _get_ewald_cache(prob.domain, prob.kernel)
    _ka_periodic_euler_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                                 seg, prob.domain, ewald, dev)

    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{QGKernel{T},PeriodicDomain{T},T},
                          dev::AbstractDevice) where {T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    ewald = _get_ewald_cache(prob.domain, EulerKernel())
    _ka_periodic_euler_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                                 seg, prob.domain, ewald, dev)
    _ka_periodic_qg_correction!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                                seg, prob.domain, ewald, prob.kernel.Ld, dev)

    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{SQGKernel{T},UnboundedDomain,T},
                          dev::AbstractDevice) where {T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    _ka_sqg_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                      seg, prob.kernel.delta, dev)

    # Copy results back to CPU (no allocation)
    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{SQGKernel{T},PeriodicDomain{T},T},
                          dev::AbstractDevice) where {T}
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))

    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    seg = SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv)
    ewald = _get_ewald_cache(prob.domain, prob.kernel)
    _ka_periodic_sqg_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                               seg, prob.domain, ewald, prob.kernel.delta, dev)

    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)

    return nothing
end

"""
    _ka_velocity!(vel, prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},<:AbstractDomain}, dev)

Evaluate a supported single-layer direct velocity path through the
KernelAbstractions backend selected by `dev`, then repack the flat result into
`vel`.
"""
function _ka_velocity!(vel::Vector{SVector{2,T}},
                       prob::ContourProblem{K, D, T, Dev},
                       dev::Dev) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T, Dev<:AbstractDevice}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    entry = _get_gpu_workspace!(dev, T, N)
    lock(entry.lock) do
        ws = entry.ws::_GPUWorkspace{T}
        _ka_velocity_ws!(ws, prob, dev)

        vx = ws.cpu_vx
        vy = ws.cpu_vy
        @inbounds for i in 1:N
            vel[i] = SVector{2,T}(vx[i], vy[i])
        end
    end

    return vel
end
