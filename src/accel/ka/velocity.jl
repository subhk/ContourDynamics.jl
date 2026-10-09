# KA velocity launch and dispatch.
#
# Thin launch wrappers around the `@kernel` definitions in `kernels.jl`, the
# kernel/domain dispatch (`_ka_apply_velocity!`), workspace orchestration, and
# the `_ka_velocity!` entry points. The CPU method builds a fresh workspace and
# exists to validate the kernels against the scalar path; the GPU method packs
# from the device-resident `DeviceContourState`.

"""
    _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev)

Launch the KA Euler velocity kernel on the given device.
"""
function _ka_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, dev::AbstractDevice)
    return _launch_ka_segment_kernel!(_euler_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev)
end

"""
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData, δ, dev)

Launch the KA SQG velocity kernel on the given device.
"""
function _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                           δ, dev::AbstractDevice)
    return _launch_ka_segment_kernel!(_sqg_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev, δ)
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

Launch the KA periodic Euler velocity kernel on the given device: the
real-space part of the Ewald sum, to which [`_ka_ewald_far_field!`](@ref) adds
the Fourier part.
"""
function _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                      domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                      dev::AbstractDevice, far=nothing) where {T}
    return _launch_ka_segment_kernel!(_periodic_euler_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      cache.α, domain.Lx, domain.Ly, cache.n_images,
                                      cache.dkx, cache.dky, _device_far_field(far, dev, T).empty)
end

"""
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache, Ld, dev)

Add the real-space part of the Ewald-split periodic QG-minus-Euler correction
to the periodic Euler velocity already stored in `vel_x`/`vel_y`.
"""
function _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                     Ld::T, dev::AbstractDevice, far=nothing) where {T}
    return _launch_ka_segment_kernel!(_periodic_qg_correction_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      Ld, cache.α, domain.Lx, domain.Ly, cache.n_images,
                                      cache.dkx, cache.dky, _device_far_field(far, dev, T).empty)
end

"""
    _ka_periodic_qg_direct_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, Ld, dev)

Launch the direct periodic-image QG velocity kernel (short deformation radius).
"""
function _ka_periodic_qg_direct_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                          domain::PeriodicDomain{T}, Ld::T,
                                          dev::AbstractDevice) where {T}
    return _launch_ka_segment_kernel!(_periodic_qg_direct_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      Ld, domain.Lx, domain.Ly)
end

"""
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, δ, dev)

Launch the KA periodic SQG velocity kernel on the given device: the real-space
part of the Ewald sum, to which [`_ka_ewald_far_field!`](@ref) adds the
Fourier part.
"""
function _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                    domain::PeriodicDomain{T}, cache::EwaldCache{T},
                                    δ::T, dev::AbstractDevice, far=nothing) where {T}
    return _launch_ka_segment_kernel!(_periodic_sqg_velocity_ka!,
                                      vel_x, vel_y, target_x, target_y, seg, dev,
                                      cache.α, δ, domain.Lx, domain.Ly, cache.n_images,
                                      cache.dkx, cache.dky, _device_far_field(far, dev, T).empty)
end

"""
    _ka_ewald_far_field!(vel_x, vel_y, target_x, target_y, seg, kernel, cache, dev, far)

Add the Fourier part of the periodic Ewald velocity: build the structure factor
of all segments, then sum it over the modes at every target (see
velocity/periodic/far_field.jl).
"""
function _ka_ewald_far_field!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                              kernel::AbstractKernel, cache::EwaldCache{T},
                              dev::AbstractDevice, far::_DeviceFarField{T}) where {T}
    n_seg = length(seg.ax)
    n_targets = length(target_x)
    coeff = _prepare_device_far_field!(far, cache, kernel, 5 * n_seg, dev)
    K = length(cache.kx) ÷ 2
    if n_seg > 0
        @_ka_launch dev n_seg _ewald_sources_ka!(
            far.src_x, far.src_y, far.src_wx, far.src_wy,
            seg.ax, seg.ay, seg.bx, seg.by, seg.pv, seg.ka, seg.kb, n_seg)
    end
    @_ka_launch dev (2K + 1) * (K + 1) _ewald_structure_factor_ka!(
        far.s_re_x, far.s_im_x, far.s_re_y, far.s_im_y,
        far.src_x, far.src_y, far.src_wx, far.src_wy,
        cache.dkx, cache.dky, K, 5 * n_seg)
    n_targets > 0 && @_ka_launch dev n_targets _ewald_far_field_ka!(
        vel_x, vel_y, target_x, target_y, coeff,
        far.s_re_x, far.s_im_x, far.s_re_y, far.s_im_y,
        cache.dkx, cache.dky, K, n_targets)
    return nothing
end

# Resolve the same registry entry as the CPU path (`_prefetch_ewald`): each
# kernel reads its own key. The QG cache carries the periodic Euler
# coefficients too, so the Euler sub-evaluation needs no separate entry.
@inline _ka_periodic_cache(domain::PeriodicDomain, kernel::AbstractKernel) =
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
    _ka_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, kernel.δ, dev)
end

# Periodic kernels: the pair kernels sum the real-space part, then the far
# field adds the Fourier part once for all segments.
@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::EulerKernel, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    far = _device_far_field(ws, dev, T)
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev, far)
    _ka_ewald_far_field!(vel_x, vel_y, target_x, target_y, seg, kernel, cache, dev, far)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::QGKernel{T}, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    _qg_uses_direct_images(inv(kernel.Ld^2), cache.α) &&
        return _ka_periodic_qg_direct_velocity!(vel_x, vel_y, target_x, target_y, seg,
                                                domain, kernel.Ld, dev)
    far = _device_far_field(ws, dev, T)
    _ka_periodic_euler_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache, dev, far)
    _ka_periodic_qg_correction!(vel_x, vel_y, target_x, target_y, seg, domain, cache,
                                kernel.Ld, dev, far)
    _ka_ewald_far_field!(vel_x, vel_y, target_x, target_y, seg, kernel, cache, dev, far)
end

@inline function _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg::SegmentData,
                                     kernel::SQGKernel{T}, domain::PeriodicDomain{T},
                                     dev::AbstractDevice, ws=nothing) where {T}
    cache = _ka_periodic_cache(domain, kernel)
    far = _device_far_field(ws, dev, T)
    _ka_periodic_sqg_velocity!(vel_x, vel_y, target_x, target_y, seg, domain, cache,
                               kernel.δ, dev, far)
    _ka_ewald_far_field!(vel_x, vel_y, target_x, target_y, seg, kernel, cache, dev, far)
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
        "KA workspace was allocated for $(ws.n) nodes but problem now has $N nodes. " *
        "Build a workspace sized to the current node count."))
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
    # Launches are not synchronized individually; drain the device queue
    # before copying its results into host memory.
    KernelAbstractions.synchronize(KernelAbstractions.get_backend(ws.dev_vel_x))
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

# Reused workspace for the device-resident velocity path. Reusing one workspace
# across RK stages avoids reallocating the 7 segment buffers + 2 velocity
# buffers every evaluation (4×/RK4 step), and — by passing the workspace to
# `_ka_apply_velocity!` — keeps the periodic far-field buffers on-device across
# stages instead of reallocating them each call.
#
# The caller's ExecutionWorkspace owns these buffers. A task-local owner is
# used only by standalone internal calls that omit the workspace keyword.
# The concrete workspace type is resolved at the launch function barrier.
const _STATE_WS_KEY = :contourdynamics_state_velocity_workspace

function _get_state_workspace(dev::AbstractDevice, ::Type{T}, N::Int; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    store = workspace.buffers
    key = (_STATE_WS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    # Rebuild when absent or when surgery changed the node count. The velocity
    # kernels derive their segment count from the buffer length, so the
    # workspace must be sized exactly N (not merely ≥ N).
    if ws === nothing || (ws::_GPUWorkspace).n != N
        ws = _create_gpu_workspace(dev, T, N)
        store[key] = ws
    end
    return ws
end

# Multi-layer buffers share the explicit execution workspace lifetime,
# rebuilt when surgery changes the concatenated node count.
const _MULTILAYER_WS_KEY = :contourdynamics_multilayer_velocity_workspace

function _get_multilayer_workspace(dev::AbstractDevice, ::Type{T}, total::Int; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    store = workspace.buffers
    key = (_MULTILAYER_WS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    if ws === nothing || (ws::_MultilayerWorkspace).n != total
        ws = _create_multilayer_workspace(dev, T, total)
        store[key] = ws
    end
    return ws
end

# Concrete-typed barrier: `ws` is `Any` from the cache, so resolve it here once
# (per evaluation) and let the kernel launches specialize on the concrete types.
function _state_velocity_with_ws!(vel::AbstractVector{SVector{2,T}},
                                  ws::_GPUWorkspace{T},
                                  state::DeviceContourState{T}, kernel,
                                  domain::AbstractDomain, dev::AbstractDevice,
                                  N::Int) where {T}
    seg = _state_segment_data!(ws, state, dev)
    _ka_apply_velocity!(ws.dev_vel_x, ws.dev_vel_y, state.x, state.y, seg,
                        kernel, domain, dev, ws)
    return _copy_velocity_output!(vel, ws.dev_vel_x, ws.dev_vel_y, dev, N)
end

function _ka_velocity_from_state!(vel::AbstractVector{SVector{2,T}},
                                  state::DeviceContourState{T},
                                  kernel::Union{EulerKernel,QGKernel{T},SQGKernel{T}},
                                  domain::AbstractDomain,
                                  dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    N = _device_state_nnodes(state)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    ws = _get_state_workspace(dev, T, N; workspace=workspace)
    return _state_velocity_with_ws!(vel, ws, state, kernel, domain, dev, N)
end

# ── Batched point probes ─────────────────────────────────────────────────
#
# `velocity(prob, points)` on a device problem packs the segments once and
# evaluates every target in a single launch (ndrange = number of points). The
# single-point probe is the one-element case of the same path, so there is one
# set of launches to keep correct.

# Split host or device points into flat device target buffers plus zeroed
# device velocity buffers of the same length.
function _ka_point_targets(points::AbstractVector{SVector{2,T}},
                           dev::AbstractDevice) where {T}
    host = to_cpu(points)
    M = length(host)
    tx = Vector{T}(undef, M)
    ty = Vector{T}(undef, M)
    @inbounds for i in 1:M
        p = host[i]
        tx[i] = p[1]
        ty[i] = p[2]
    end
    return (to_device(dev, tx), to_device(dev, ty),
            device_zeros(dev, T, M), device_zeros(dev, T, M))
end

function _ka_points_result(vel_x, vel_y, ::Type{T}, M::Int) where {T}
    host_x = to_cpu(vel_x)
    host_y = to_cpu(vel_y)
    out = Vector{SVector{2,T}}(undef, M)
    @inbounds for i in 1:M
        out[i] = SVector{2,T}(host_x[i], host_y[i])
    end
    return out
end

function _ka_velocity_at_state(state::DeviceContourState{T},
                               kernel::Union{EulerKernel,QGKernel{T},SQGKernel{T}},
                               domain::AbstractDomain,
                               points::AbstractVector{SVector{2,T}},
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    M = length(points)
    M == 0 && return SVector{2,T}[]
    N = _device_state_nnodes(state)
    ws = _get_state_workspace(dev, T, N; workspace=workspace)
    seg = _state_segment_data!(ws, state, dev)
    target_x, target_y, vel_x, vel_y = _ka_point_targets(points, dev)
    _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg,
                        kernel, domain, dev, ws)
    return _ka_points_result(vel_x, vel_y, T, M)
end

function _ka_velocity_at_state(state::DeviceContourState{T},
                               kernel::Union{EulerKernel,QGKernel{T},SQGKernel{T}},
                               domain::AbstractDomain, x::SVector{2,T},
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    return only(_ka_velocity_at_state(state, kernel, domain, SVector{2,T}[x], dev;
                                      workspace=workspace))
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

    # CPU velocity! uses the direct scalar evaluator; this KA path on CPU exists
    # only to validate the KA kernels against the scalar reference in tests, so a
    # fresh per-call workspace is fine (no caching needed).
    ws = _create_gpu_workspace(dev, T, N)
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
                       dev::GPU) where {K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel}, D<:AbstractDomain, T}
    return _ka_velocity_from_state!(vel, _device_state(prob), prob.kernel,
                                    prob.domain, dev; workspace=execution_workspace(prob))
end

"""
    _ka_multilayer_velocity_from_states!(vel, states, kernel, domain, dev) -> vel

Device-resident modal velocity for multi-layer problems. For each vertical mode,
packs every layer's segments with PV scaled by `physical_to_modal[mode, layer]`,
evaluates the single-layer KA velocity kernels over the concatenated segments and
targets, and accumulates `modal_to_physical[layer, mode]` times the modal result into
the flat per-layer output. The flat layout follows [`_layer_state_ranges`](@ref).
"""
function _ka_multilayer_velocity_from_states!(vel::AbstractVector{SVector{2,T}},
                                              states::NTuple{N, <:DeviceContourState},
                                              kernel::MultiLayerQGKernel{N},
                                              domain::AbstractDomain,
                                              dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N, T}
    ranges = _layer_state_ranges(states)
    total = sum(length, ranges)
    length(vel) >= total || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($total)"))
    total == 0 && return vel

    # Reuse the execution workspace across RK stages instead of allocating 13
    # device arrays per evaluation. `ws` is `Any` from the cache; the concrete-
    # typed barrier `_multilayer_velocity_with_ws!` restores typing for the hot launches.
    ws = _get_multilayer_workspace(dev, T, total; workspace=workspace)
    return _multilayer_velocity_with_ws!(vel, ws, states, kernel, domain, dev, ranges, total)
end

function _pack_multilayer_mode_segments!(
        ws::_MultilayerWorkspace{T}, states::NTuple{N,<:DeviceContourState},
        ranges, to_modal, mode::Int, dev::AbstractDevice) where {N,T}
    for layer in 1:N
        range = ranges[layer]
        isempty(range) && continue
        state = states[layer]
        n_layer = length(range)
        @_ka_launch dev n_layer _state_segment_data_kernel!(
            view(ws.ax, range), view(ws.ay, range),
            view(ws.bx, range), view(ws.by, range), view(ws.pv, range),
            view(ws.ka, range), view(ws.kb, range),
            state.x, state.y, state.pv, state.wrapx, state.wrapy,
            state.offsets, state.lengths, state.corners,
            state.contour_of_node, state.local_index,
            T(to_modal[mode, layer]), n_layer)
    end
    return SegmentData(ws.ax, ws.ay, ws.bx, ws.by, ws.pv, ws.ka, ws.kb)
end

@inline function _ka_apply_modal_velocity!(
        mode_kernel, out_vx, out_vy, target_x, target_y, segments,
        domain, dev, ws)
    return _ka_apply_velocity!(out_vx, out_vy, target_x, target_y, segments,
                               mode_kernel, domain, dev, ws)
end

"""
    _ka_multilayer_velocity_to_host!(vel, states, kernel, domain, dev) -> vel

Evaluate device-resident multilayer velocity and scatter it into one host
vector per layer. Both the flat device result and its host transfer target live
in the execution workspace, so repeated calls allocate no buffers
proportional to the node count.
"""
function _ka_multilayer_velocity_to_host!(vel::NTuple{N,Vector{SVector{2,T}}},
                                          states::NTuple{N,<:DeviceContourState},
                                          kernel::MultiLayerQGKernel{N},
                                          domain::AbstractDomain,
                                          dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N,T}
    ranges = _layer_state_ranges(states)
    for layer in 1:N
        required = length(ranges[layer])
        length(vel[layer]) >= required || throw(DimensionMismatch(
            "vel[$layer] length ($(length(vel[layer]))) must be >= layer $layer nodes ($required)"))
    end

    total = sum(length, ranges)
    total == 0 && return vel
    ws = _get_multilayer_workspace(dev, T, total; workspace=workspace)
    return _multilayer_velocity_to_host_with_ws!(vel, ws, states, kernel,
                                                 domain, dev, ranges, total)
end

# The workspace buffer dictionary is `Any`-typed. Keep workspace field access and the host
# scatter behind this concrete barrier; otherwise dynamic `copyto!` dispatch
# boxes every `SVector` element on the CPU backend.
function _multilayer_velocity_to_host_with_ws!(
        vel::NTuple{N,Vector{SVector{2,T}}}, ws::_MultilayerWorkspace{T},
        states::NTuple{N,<:DeviceContourState}, kernel::MultiLayerQGKernel{N},
        domain::AbstractDomain, dev::AbstractDevice, ranges, total::Int) where {N,T}
    _multilayer_velocity_with_ws!(ws.flat_vel, ws, states, kernel, domain,
                                  dev, ranges, total)
    _device_synchronize(dev)
    copyto!(ws.host_flat, ws.flat_vel)

    for layer in 1:N
        r = ranges[layer]
        @inbounds for (local_index, global_index) in enumerate(r)
            vel[layer][local_index] = ws.host_flat[global_index]
        end
    end
    return vel
end

function _multilayer_velocity_with_ws!(vel::AbstractVector{SVector{2,T}},
                                       ws::_MultilayerWorkspace{T},
                                       states::NTuple{N, <:DeviceContourState},
                                       kernel::MultiLayerQGKernel{N},
                                       domain::AbstractDomain, dev::AbstractDevice,
                                       ranges, total::Int) where {N, T}
    evals = kernel.eigenvalues
    to_physical = kernel.modal_to_physical
    to_modal = kernel.physical_to_modal

    tx, ty = ws.tx, ws.ty
    mode_vx, mode_vy = ws.mode_vx, ws.mode_vy
    vel_x, vel_y = ws.vel_x, ws.vel_y

    # `vel_x`/`vel_y` are accumulated into across modes via `_modal_accumulate_ka!`,
    # so the reused buffers must start at zero (a fresh `device_zeros` did this
    # implicitly before). The other buffers are fully overwritten before use.
    fill!(vel_x, zero(T)); fill!(vel_y, zero(T))

    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        copyto!(view(tx, r), states[ℓ].x)
        copyto!(view(ty, r), states[ℓ].y)
    end

    for m in 1:N
        lam = evals[m]
        segments = _pack_multilayer_mode_segments!(
            ws, states, ranges, to_modal, m, dev)
        _dispatch_qg_mode(
            _ka_apply_modal_velocity!, kernel, lam,
            mode_vx, mode_vy, tx, ty, segments, domain, dev, ws)

        for ℓ in 1:N
            w = to_physical[ℓ, m]
            abs(w) < eps(T) && continue
            r = ranges[ℓ]
            isempty(r) && continue
            n_l = length(r)
            @_ka_launch dev n_l _modal_accumulate_ka!(
                view(vel_x, r), view(vel_y, r), view(mode_vx, r), view(mode_vy, r),
                T(w), n_l)
        end
    end

    return _copy_velocity_output!(vel, vel_x, vel_y, dev, total)
end

# Modal results and layer outputs are stored point-major per mode/layer:
# entry `(k - 1) * npoints + i` holds point `i` of mode/layer `k`.
@kernel function _project_point_modes_ka!(out_x, out_y, mode_x, mode_y,
                                          to_physical, nlayers, npoints)
    idx = @index(Global)
    if idx <= nlayers * npoints
        T = eltype(out_x)
        layer = (idx - 1) ÷ npoints + 1
        i = (idx - 1) % npoints + 1
        vx = zero(T)
        vy = zero(T)
        @inbounds for mode in 1:nlayers
            weight = to_physical[layer, mode]
            vx += weight * mode_x[(mode - 1) * npoints + i]
            vy += weight * mode_y[(mode - 1) * npoints + i]
        end
        @inbounds out_x[idx] = vx
        @inbounds out_y[idx] = vy
    end
end

function _ka_multilayer_velocity_at_states(
        states::NTuple{N,<:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, domain::AbstractDomain,
        points::AbstractVector{SVector{2,T}}, dev::AbstractDevice;
        workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N,T}
    M = length(points)
    M == 0 && return NTuple{N,SVector{2,T}}[]
    ranges = _layer_state_ranges(states)
    total = sum(length, ranges)
    ws = _get_multilayer_workspace(dev, T, total; workspace=workspace)
    target_x, target_y, point_x, point_y = _ka_point_targets(points, dev)
    mode_x = device_zeros(dev, T, N * M)
    mode_y = device_zeros(dev, T, N * M)
    to_modal = kernel.physical_to_modal

    for mode in 1:N
        segments = _pack_multilayer_mode_segments!(
            ws, states, ranges, to_modal, mode, dev)
        lam = kernel.eigenvalues[mode]
        _dispatch_qg_mode(
            _ka_apply_modal_velocity!, kernel, lam,
            point_x, point_y, target_x, target_y, segments, domain, dev, ws)
        slot = ((mode - 1) * M + 1):(mode * M)
        copyto!(view(mode_x, slot), point_x)
        copyto!(view(mode_y, slot), point_y)
    end

    out_x = device_zeros(dev, T, N * M)
    out_y = device_zeros(dev, T, N * M)
    @_ka_launch dev N * M _project_point_modes_ka!(
        out_x, out_y, mode_x, mode_y, kernel.modal_to_physical, N, M)
    host_x = to_cpu(out_x)
    host_y = to_cpu(out_y)
    return [ntuple(layer -> SVector{2,T}(host_x[(layer - 1) * M + i],
                                         host_y[(layer - 1) * M + i]), Val(N))
            for i in 1:M]
end

function _ka_multilayer_velocity_at_states(
        states::NTuple{N,<:DeviceContourState{T}},
        kernel::MultiLayerQGKernel{N}, domain::AbstractDomain,
        x::SVector{2,T}, dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {N,T}
    return only(_ka_multilayer_velocity_at_states(states, kernel, domain, SVector{2,T}[x],
                                                  dev; workspace=workspace))
end

# ── Beta-plane device velocity ───────────────────────────────────────────

# Combined live+reference segment buffers for the beta-plane path. The frozen
# reference staircase is packed once on the host (with signed node curvatures)
# with NEGATED pv into the tail of the buffers, so one periodic-QG launch over
# the concatenated segments yields `current - reference` directly; the analytic
# sawtooth zonal term is then added per target. Task-local like the other
# velocity workspaces.
mutable struct _BetaPlaneWorkspace{T, DA<:AbstractVector{T}}
    ax::DA; ay::DA; bx::DA; by::DA; pv::DA; ka::DA; kb::DA
    live_n::Int
    ref_n::Int
    last_reference::Union{Nothing, Vector{PVContour{T}}}
end

const _BETA_WS_KEY = :contourdynamics_beta_plane_velocity_workspace

function _pack_reference_segments(contours::Vector{PVContour{T}}) where {T}
    n = sum(c -> nnodes(c) >= 2 ? nnodes(c) : 0, contours; init=0)
    ax = Vector{T}(undef, n); ay = Vector{T}(undef, n)
    bx = Vector{T}(undef, n); by = Vector{T}(undef, n)
    pv = Vector{T}(undef, n)
    ka = Vector{T}(undef, n); kb = Vector{T}(undef, n)
    curvatures = _prepare_curvature_buffers!(Vector{Vector{T}}(), contours)
    idx = 1
    @inbounds for (ci, c) in pairs(contours)
        nc = nnodes(c)
        nc < 2 && continue
        κ = curvatures[ci]
        for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            ax[idx] = a[1]; ay[idx] = a[2]
            bx[idx] = b[1]; by[idx] = b[2]
            pv[idx] = -c.pv                    # negated: subtracts the reference field
            ka[idx] = κ[j]
            kb[idx] = κ[mod1(j + 1, nc)]
            idx += 1
        end
    end
    return ax, ay, bx, by, pv, ka, kb
end

function _create_beta_plane_workspace(dev::AbstractDevice, ::Type{T}, live_n::Int,
                                      reference::Vector{PVContour{T}}) where {T}
    rax, ray, rbx, rby, rpv, rka, rkb = _pack_reference_segments(reference)
    ref_n = length(rax)
    total = live_n + ref_n
    function mk(tail::Vector{T})
        host = zeros(T, total)
        copyto!(view(host, (live_n + 1):total), tail)
        return to_device(dev, host)
    end
    da = mk(rax)
    _BetaPlaneWorkspace{T, typeof(da)}(da, mk(ray), mk(rbx), mk(rby),
                                       mk(rpv), mk(rka), mk(rkb),
                                       live_n, ref_n, reference)
end

function _get_beta_plane_workspace(dev::AbstractDevice, ::Type{T}, live_n::Int,
                                   reference::Vector{PVContour{T}}; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    store = workspace.buffers
    key = (_BETA_WS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    if ws === nothing || (ws::_BetaPlaneWorkspace).live_n != live_n ||
       (ws::_BetaPlaneWorkspace).last_reference !== reference
        ws = _create_beta_plane_workspace(dev, T, live_n, reference)
        store[key] = ws
    end
    return ws
end

function _beta_plane_velocity_with_ws!(vel::AbstractVector{SVector{2,T}},
                                       gws::_GPUWorkspace{T},
                                       bws::_BetaPlaneWorkspace{T},
                                       state::DeviceContourState{T},
                                       kernel::BetaPlaneQGKernel{T},
                                       domain::PeriodicDomain{T},
                                       dev::AbstractDevice, N::Int) where {T}
    live = 1:N
    @_ka_launch dev N _state_segment_data_kernel!(
        view(bws.ax, live), view(bws.ay, live), view(bws.bx, live), view(bws.by, live),
        view(bws.pv, live), view(bws.ka, live), view(bws.kb, live),
        state.x, state.y, state.pv, state.wrapx, state.wrapy,
        state.offsets, state.lengths, state.corners,
        state.contour_of_node, state.local_index, one(T), N)
    seg = SegmentData(bws.ax, bws.ay, bws.bx, bws.by, bws.pv, bws.ka, bws.kb)
    _ka_apply_velocity!(gws.dev_vel_x, gws.dev_vel_y, state.x, state.y, seg,
                        QGKernel(kernel.Ld), domain, dev, gws)
    dy = 2 * domain.Ly / T(length(kernel.reference_contours))
    @_ka_launch dev N _beta_sawtooth_add_ka!(gws.dev_vel_x, state.y,
                                             kernel.beta, inv(kernel.Ld), dy,
                                             domain.Ly, N)
    return _copy_velocity_output!(vel, gws.dev_vel_x, gws.dev_vel_y, dev, N)
end

function _ka_velocity_from_state!(vel::AbstractVector{SVector{2,T}},
                                  state::DeviceContourState{T},
                                  kernel::BetaPlaneQGKernel{T},
                                  domain::PeriodicDomain{T},
                                  dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    N = _device_state_nnodes(state)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel
    gws = _get_state_workspace(dev, T, N; workspace=workspace)
    bws = _get_beta_plane_workspace(dev, T, N, kernel.reference_contours; workspace=workspace)
    return _beta_plane_velocity_with_ws!(vel, gws, bws, state, kernel, domain, dev, N)
end

function _ka_velocity_at_state(state::DeviceContourState{T},
                               kernel::BetaPlaneQGKernel{T},
                               domain::PeriodicDomain{T},
                               points::AbstractVector{SVector{2,T}},
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    M = length(points)
    M == 0 && return SVector{2,T}[]
    N = _device_state_nnodes(state)
    gws = _get_state_workspace(dev, T, N; workspace=workspace)
    bws = _get_beta_plane_workspace(dev, T, N, kernel.reference_contours; workspace=workspace)
    if N > 0
        live = 1:N
        @_ka_launch dev N _state_segment_data_kernel!(
            view(bws.ax, live), view(bws.ay, live), view(bws.bx, live), view(bws.by, live),
            view(bws.pv, live), view(bws.ka, live), view(bws.kb, live),
            state.x, state.y, state.pv, state.wrapx, state.wrapy,
            state.offsets, state.lengths, state.corners,
            state.contour_of_node, state.local_index, one(T), N)
    end
    seg = SegmentData(bws.ax, bws.ay, bws.bx, bws.by, bws.pv, bws.ka, bws.kb)
    target_x, target_y, vel_x, vel_y = _ka_point_targets(points, dev)
    _ka_apply_velocity!(vel_x, vel_y, target_x, target_y, seg,
                        QGKernel(kernel.Ld), domain, dev, gws)
    dy = 2 * domain.Ly / T(length(kernel.reference_contours))
    @_ka_launch dev M _beta_sawtooth_add_ka!(vel_x, target_y,
                                             kernel.beta, inv(kernel.Ld), dy,
                                             domain.Ly, M)
    return _ka_points_result(vel_x, vel_y, T, M)
end

function _ka_velocity_at_state(state::DeviceContourState{T},
                               kernel::BetaPlaneQGKernel{T},
                               domain::PeriodicDomain{T}, x::SVector{2,T},
                               dev::AbstractDevice; workspace::ExecutionWorkspace{T}=_default_execution_workspace(T)) where {T}
    return only(_ka_velocity_at_state(state, kernel, domain, SVector{2,T}[x], dev;
                                      workspace=workspace))
end
