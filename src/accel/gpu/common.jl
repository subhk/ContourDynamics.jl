# GPU-portable velocity kernels using KernelAbstractions.jl.
#
# The public `velocity!` API works with normal contour structs. Before a device
# kernel can run, contours are packed into a flat segment structure-of-arrays
# layout (`SegmentData`) plus flat target-node arrays. Kernels then compute one
# target velocity per work item, and the result is copied back into the caller's
# velocity vectors.
#
# The same kernels can run on KernelAbstractions.CPU for testing and on CUDA via
# the package extension. CPU execution still uses the direct reference path by
# default; this file exists so device behavior has a single implementation.

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
                                             local_index, total_nodes)
    g = @index(Global)
    if g <= total_nodes
        T = eltype(x)
        ci = contour_of_node[g]
        li = local_index[g]
        n = lengths[ci]
        off = offsets[ci]

        ax[g] = x[g]
        ay[g] = y[g]
        seg_pv[g] = pv[ci]

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
        state.contour_of_node, state.local_index, N)
    return SegmentData(ax, ay, bx, by, pv, ka, kb)
end

@kernel function _copy_velocity_svector_kernel!(vel, vel_x, vel_y, n)
    i = @index(Global)
    if i <= n
        T = eltype(vel_x)
        vel[i] = SVector{2,T}(vel_x[i], vel_y[i])
    end
end

# ── GPU workspace for buffer reuse ───────────────────
# Avoids repeated allocation of CPU packing buffers and device arrays
# across the 4 velocity evaluations per RK4 step.

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

# Module-level workspace cache keyed by (T, N, Dev) so problems with different
# sizes or devices don't collide. The cache reuses mutable packing buffers and
# assumes callers do not concurrently evaluate velocity with the same cache key.
struct _WSEntry{T, DA<:AbstractVector{T}, DMA<:AbstractMatrix{T}}
    ws::_GPUWorkspace{T, DA, DMA}
end
const _gpu_ws_cache_f64_cpu = Dict{Int, _WSEntry{Float64, Vector{Float64}, Matrix{Float64}}}()
const _gpu_ws_cache_f32_cpu = Dict{Int, _WSEntry{Float32, Vector{Float32}, Matrix{Float32}}}()
const _gpu_ws_order_f64_cpu = Int[]
const _gpu_ws_order_f32_cpu = Int[]
const _gpu_ws_cache = Dict{Tuple{DataType, DataType}, Any}()
const _gpu_ws_key_order = Dict{Tuple{DataType, DataType}, Vector{Int}}()
const _GPU_WS_CACHE_MAX = 8

_gpu_ws_dict(::Type{Float64}, ::Type{CPU}) = _gpu_ws_cache_f64_cpu
_gpu_ws_dict(::Type{Float32}, ::Type{CPU}) = _gpu_ws_cache_f32_cpu
function _gpu_ws_dict(::Type{T}, ::Type{Dev}) where {T, Dev<:AbstractDevice}
    key = (T, Dev)
    cache = get!(_gpu_ws_cache, key) do
        Dict{Int, _WSEntry{T}}()
    end
    return cache::Dict{Int, _WSEntry{T}}
end

_gpu_ws_order(::Type{Float64}, ::Type{CPU}) = _gpu_ws_order_f64_cpu
_gpu_ws_order(::Type{Float32}, ::Type{CPU}) = _gpu_ws_order_f32_cpu
function _gpu_ws_order(::Type{T}, ::Type{Dev}) where {T, Dev<:AbstractDevice}
    key = (T, Dev)
    order = get!(_gpu_ws_key_order, key) do
        Int[]
    end
    return order::Vector{Int}
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

function _get_gpu_workspace!(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    Dev = typeof(dev)
    caches = _gpu_ws_dict(T, Dev)
    order = _gpu_ws_order(T, Dev)

    cached = get(caches, N, nothing)
    cached !== nothing && return cached

    # Small FIFO eviction keeps the cache bounded while preserving the common
    # case where RK stages repeatedly reuse the same node count.
    while length(caches) >= _GPU_WS_CACHE_MAX && !isempty(order)
        old_n = popfirst!(order)
        delete!(caches, old_n)
    end
    new_entry = _WSEntry(_create_gpu_workspace(dev, T, N))
    caches[N] = new_entry
    push!(order, N)
    return new_entry
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
    _evict_gpu_workspace!(T, Dev, N)
    return nothing
end

function _evict_gpu_workspace!(::Type{Float64}, ::Type{CPU}, N::Int)
    delete!(_gpu_ws_cache_f64_cpu, N)
    filter!(!=(N), _gpu_ws_order_f64_cpu)
    return nothing
end

function _evict_gpu_workspace!(::Type{Float32}, ::Type{CPU}, N::Int)
    delete!(_gpu_ws_cache_f32_cpu, N)
    filter!(!=(N), _gpu_ws_order_f32_cpu)
    return nothing
end

function _evict_gpu_workspace!(::Type{T}, ::Type{Dev}, N::Int) where {T, Dev<:AbstractDevice}
    caches = get(_gpu_ws_cache, (T, Dev), nothing)
    order = get(_gpu_ws_key_order, (T, Dev), nothing)
    caches === nothing || delete!(caches, N)
    order === nothing || filter!(!=(N), order)
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

"""
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev)

Launch the KA Euler velocity kernel on the given device.
"""
function _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev::AbstractDevice)
    return _launch_ka_segment_kernel!(_euler_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev)
end

"""
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, delta, dev)

Launch the KA SQG velocity kernel on the given device.
"""
function _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                           delta, dev::AbstractDevice)
    return _launch_ka_segment_kernel!(_sqg_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev, delta)
end

"""
    _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, Ld, dev)

Launch the KA QG velocity kernel on the given device.
"""
function _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                          Ld, dev::AbstractDevice)
    return _launch_ka_segment_kernel!(_qg_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev, Ld)
end

"""
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev)

Launch the KA periodic Euler velocity kernel on the given device.
"""
function _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                      domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                      dev::AbstractDevice, ws=nothing) where {T}
    dev_kx, dev_ky, dev_fourier = _periodic_ewald_data(ws, cache, dev)
    return _launch_ka_segment_kernel!(_periodic_euler_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      cache.alpha, domain.Lx, domain.Ly, cache.n_images,
                                      dev_kx, dev_ky, dev_fourier)
end

"""
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache, Ld, dev)

Apply the periodic QG-minus-Euler Fourier correction on top of the periodic
Euler velocity already stored in `vel_x`/`vel_y`.
"""
function _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                     Ld::T, dev::AbstractDevice, ws=nothing) where {T}
    dev_kx, dev_ky = _periodic_ewald_vectors(ws, cache, dev)
    return _launch_ka_segment_kernel!(_periodic_qg_correction_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      Ld, domain.Lx, domain.Ly, dev_kx, dev_ky)
end

"""
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, delta, dev)

Launch the KA periodic SQG velocity kernel on the given device.
"""
function _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                    domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                    delta::T, dev::AbstractDevice, ws=nothing) where {T}
    dev_kx, dev_ky, dev_fourier = _periodic_ewald_data(ws, cache, dev)
    return _launch_ka_segment_kernel!(_periodic_sqg_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      cache.alpha, delta, domain.Lx, domain.Ly, cache.n_images,
                                      dev_kx, dev_ky, dev_fourier)
end

@inline _ka_periodic_cache(domain::PeriodicDomain, kernel::EulerKernel) =
    _get_ewald_cache(domain, kernel)
@inline _ka_periodic_cache(domain::PeriodicDomain, ::QGKernel) =
    _get_ewald_cache(domain, EulerKernel())
@inline _ka_periodic_cache(domain::PeriodicDomain, kernel::SQGKernel) =
    _get_ewald_cache(domain, kernel)

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     ::EulerKernel, ::UnboundedDomain,
                                     dev::AbstractDevice, ws=nothing)
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, dev)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::QGKernel{T}, ::UnboundedDomain,
                                     dev::AbstractDevice, ws=nothing) where {T}
    _ka_qg_velocity!(vel_x, vel_y, target_x, target_y, seg, kernel.Ld, dev)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::SQGKernel{T}, ::UnboundedDomain,
                                     dev::AbstractDevice, ws=nothing) where {T}
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, kernel.delta, dev)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::EulerKernel, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev, ws)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev, ws)
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache,
                                kernel.Ld, dev, ws)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache,
                               kernel.delta, dev, ws)
end

"""
    _ka_velocity_ws!(ws::_GPUWorkspace, prob::ContourProblem, dev::AbstractDevice)

KernelAbstractions-based velocity evaluation using pre-allocated workspace
buffers. This supports both CPU and GPU backends through the same packing and
launch path for the kernels that already have flat direct evaluators.
"""
@inline function _check_workspace_size(ws::_GPUWorkspace, prob::ContourProblem)
    N = total_nodes(prob)
    N == ws.n || throw(DimensionMismatch(
        "GPU workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Call velocity! through evolve!() or manually evict stale workspaces after surgery!()."))
    return N
end

@inline function _pack_workspace!(ws::_GPUWorkspace, prob::ContourProblem)
    _fill_segment_bufs!(ws.cpu_ax, ws.cpu_ay, ws.cpu_bx, ws.cpu_by, ws.cpu_pv,
                        ws.cpu_ka, ws.cpu_kb, prob)
    _fill_target_bufs!(ws.cpu_tx, ws.cpu_ty, prob)

    copyto!(ws.dev_ax, ws.cpu_ax)
    copyto!(ws.dev_ay, ws.cpu_ay)
    copyto!(ws.dev_bx, ws.cpu_bx)
    copyto!(ws.dev_by, ws.cpu_by)
    copyto!(ws.dev_pv, ws.cpu_pv)
    copyto!(ws.dev_ka, ws.cpu_ka)
    copyto!(ws.dev_kb, ws.cpu_kb)
    copyto!(ws.dev_tx, ws.cpu_tx)
    copyto!(ws.dev_ty, ws.cpu_ty)

    return SegmentData(ws.dev_ax, ws.dev_ay, ws.dev_bx, ws.dev_by, ws.dev_pv,
                       ws.dev_ka, ws.dev_kb)
end

@inline function _copy_workspace_velocity!(ws::_GPUWorkspace)
    copyto!(ws.cpu_vx, ws.dev_vel_x)
    copyto!(ws.cpu_vy, ws.dev_vel_y)
    return nothing
end

function _with_packed_workspace!(f, ws::_GPUWorkspace{T},
                                 prob::ContourProblem{<:Any, <:Any, T},
                                 dev::AbstractDevice) where {T}
    _check_workspace_size(ws, prob)
    seg = _pack_workspace!(ws, prob)
    f(ws, seg, prob, dev)
    _copy_workspace_velocity!(ws)
    return nothing
end

@inline function _ka_workspace_launch!(ws::_GPUWorkspace,
                                       seg::SegmentData,
                                       prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},<:AbstractDomain},
                                       dev::AbstractDevice)
    _ka_apply_velocity!(ws.dev_vel_x, ws.dev_vel_y, ws.dev_tx, ws.dev_ty,
                        seg, prob.kernel, prob.domain, dev, ws)
end

function _ka_velocity_ws!(ws::_GPUWorkspace{T},
                          prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},<:AbstractDomain,T},
                          dev::AbstractDevice) where {T}
    return _with_packed_workspace!(ws, prob, dev) do ws, seg, prob, dev
        _ka_workspace_launch!(ws, seg, prob, dev)
    end
end

function _copy_velocity_output!(vel::Vector{SVector{2,T}}, vel_x, vel_y,
                                ::AbstractDevice, N::Int) where {T}
    vx = to_cpu(vel_x)
    vy = to_cpu(vel_y)
    @inbounds for i in 1:N
        vel[i] = SVector{2,T}(vx[i], vy[i])
    end
    return vel
end

function _copy_velocity_output!(vel::AbstractVector{SVector{2,T}}, vel_x, vel_y,
                                dev::AbstractDevice, N::Int) where {T}
    @_ka_launch dev N _copy_velocity_svector_kernel!(vel, vel_x, vel_y, N)
    return vel
end

function _ka_velocity_from_state!(vel::AbstractVector{SVector{2,T}},
                                  state::DeviceContourState{T},
                                  kernel::Union{EulerKernel,QGKernel{T},SQGKernel{T}},
                                  domain::AbstractDomain,
                                  dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    seg = _state_segment_data(state, dev)
    vel_x = device_zeros(dev, T, N)
    vel_y = device_zeros(dev, T, N)
    _ka_apply_velocity!(vel_x, vel_y, state.x, state.y, seg, kernel, domain, dev)
    return _copy_velocity_output!(vel, vel_x, vel_y, dev, N)
end

"""
    _ka_velocity!(vel, prob::ContourProblem{<:Union{EulerKernel,QGKernel,SQGKernel},<:AbstractDomain}, dev)

Evaluate a supported single-layer direct velocity path through the
KernelAbstractions backend selected by `dev`, then repack the flat result into
`vel`.
"""
function _ka_velocity!(vel::Vector{SVector{2,T}},
                       prob::ContourProblem{K, D, T, CPU},
                       dev::CPU) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    entry = _get_gpu_workspace!(dev, T, N)
    ws = entry.ws
    _ka_velocity_ws!(ws, prob, dev)

    vx = ws.cpu_vx
    vy = ws.cpu_vy
    @inbounds for i in 1:N
        vel[i] = SVector{2,T}(vx[i], vy[i])
    end

    return vel
end

function _ka_velocity!(vel::AbstractVector{SVector{2,T}},
                       prob::ContourProblem{K, D, T, GPU},
                       dev::GPU) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T}
    return _ka_velocity_from_state!(vel, prob.device_state, prob.kernel,
                                    prob.domain, dev)
end
