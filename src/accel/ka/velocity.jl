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
# `_ka_apply_velocity!` — keeps the periodic Ewald tables on-device across
# stages (via `_ensure_device_ewald!`) instead of re-uploading them each call.
#
# Held in TASK-LOCAL storage (not a process-global) so concurrent velocity
# evaluation of independent problems on separate tasks/threads each get their
# own workspace and cannot race on the shared device buffers; within one task
# the RK stages reuse it. Returned as `Any` because the device array type is
# only known once an array backend (e.g. CUDA) is loaded; the function barrier
# `_state_velocity_with_ws!` restores concrete typing before the hot launches.
const _STATE_WS_TLS_KEY = :contourdynamics_state_velocity_workspace

function _get_state_workspace(dev::AbstractDevice, ::Type{T}, N::Int) where {T}
    store = task_local_storage()
    key = (_STATE_WS_TLS_KEY, T, typeof(dev))
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

# Task-local cache for the multi-layer modal velocity workspace. Same rationale
# and lifetime as `_get_state_workspace`: one workspace per (task, T, device),
# rebuilt when surgery changes the concatenated node count.
const _MULTILAYER_WS_TLS_KEY = :contourdynamics_multilayer_velocity_workspace

function _get_multilayer_workspace(dev::AbstractDevice, ::Type{T}, total::Int) where {T}
    store = task_local_storage()
    key = (_MULTILAYER_WS_TLS_KEY, T, typeof(dev))
    ws = get(store, key, nothing)
    if ws === nothing || (ws::_MultilayerWorkspace).n != total
        ws = _create_multilayer_workspace(dev, T, total)
        store[key] = ws
    end
    return ws
end

"""
    clear_state_workspace_cache!()

Drop the calling task's cached device velocity and energy workspaces, freeing
their buffers. Each path caches scratch in task-local storage and resizes it to
the current topology; the buffers otherwise persist for the task's lifetime.

This is the workspace counterpart to [`clear_ewald_cache!`](@ref). Caches are
per-task, so this only releases workspaces allocated by the calling task.
"""
function clear_state_workspace_cache!()
    store = task_local_storage()
    for key in collect(keys(store))
        key isa Tuple && length(key) == 3 &&
            (key[1] === _STATE_WS_TLS_KEY || key[1] === _MULTILAYER_WS_TLS_KEY ||
             key[1] === _BETA_WS_TLS_KEY || key[1] === _ENERGY_WS_TLS_KEY) &&
            delete!(store, key)
    end
    return nothing
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
                                  dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    ws = _get_state_workspace(dev, T, N)
    return _state_velocity_with_ws!(vel, ws, state, kernel, domain, dev, N)
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
    return _ka_velocity_from_state!(vel, prob.device_state, prob.kernel,
                                    prob.domain, dev)
end

"""
    _ka_multilayer_velocity_from_states!(vel, states, kernel, domain, dev) -> vel

Device-resident modal velocity for multi-layer problems. For each vertical mode,
packs every layer's segments with PV scaled by `eigenvectors_inv[mode, layer]`,
evaluates the single-layer KA velocity kernels over the concatenated segments and
targets, and accumulates `eigenvectors[layer, mode]` times the modal result into
the flat per-layer output. The flat layout follows [`_layer_state_ranges`](@ref).
"""
function _ka_multilayer_velocity_from_states!(vel::AbstractVector{SVector{2,T}},
                                              states::NTuple{N, <:DeviceContourState},
                                              kernel::MultiLayerQGKernel{N},
                                              domain::AbstractDomain,
                                              dev::AbstractDevice) where {N, T}
    ranges = _layer_state_ranges(states)
    total = sum(length, ranges)
    length(vel) >= total || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($total)"))
    total == 0 && return vel

    # Reuse a task-local workspace across RK stages instead of allocating 13
    # device arrays per evaluation. `ws` is `Any` from the cache; the concrete-
    # typed barrier `_multilayer_velocity_with_ws!` restores typing for the hot launches.
    ws = _get_multilayer_workspace(dev, T, total)
    return _multilayer_velocity_with_ws!(vel, ws, states, kernel, domain, dev, ranges, total)
end

"""
    _ka_multilayer_velocity_to_host!(vel, states, kernel, domain, dev) -> vel

Evaluate device-resident multilayer velocity and scatter it into one host
vector per layer. Both the flat device result and its host transfer target live
in the task-local multilayer workspace, so repeated calls allocate no buffers
proportional to the node count.
"""
function _ka_multilayer_velocity_to_host!(vel::NTuple{N,Vector{SVector{2,T}}},
                                          states::NTuple{N,<:DeviceContourState},
                                          kernel::MultiLayerQGKernel{N},
                                          domain::AbstractDomain,
                                          dev::AbstractDevice) where {N,T}
    ranges = _layer_state_ranges(states)
    for layer in 1:N
        required = length(ranges[layer])
        length(vel[layer]) >= required || throw(DimensionMismatch(
            "vel[$layer] length ($(length(vel[layer]))) must be >= layer $layer nodes ($required)"))
    end

    total = sum(length, ranges)
    total == 0 && return vel
    ws = _get_multilayer_workspace(dev, T, total)
    return _multilayer_velocity_to_host_with_ws!(vel, ws, states, kernel,
                                                 domain, dev, ranges, total)
end

# The task-local cache is `Any`-typed. Keep workspace field access and the host
# scatter behind this concrete barrier; otherwise dynamic `copyto!` dispatch
# boxes every `SVector` element on the CPU backend.
function _multilayer_velocity_to_host_with_ws!(
        vel::NTuple{N,Vector{SVector{2,T}}}, ws::_MultilayerWorkspace{T},
        states::NTuple{N,<:DeviceContourState}, kernel::MultiLayerQGKernel{N},
        domain::AbstractDomain, dev::AbstractDevice, ranges, total::Int) where {N,T}
    _multilayer_velocity_with_ws!(ws.flat_vel, ws, states, kernel, domain,
                                  dev, ranges, total)
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
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    ax, ay, bx, by = ws.ax, ws.ay, ws.bx, ws.by
    pv, ka, kb = ws.pv, ws.ka, ws.kb
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
        # Repack segments with this mode's per-layer PV weights. Geometry
        # (a, b, curvatures) is identical across modes; only seg_pv changes.
        for ℓ in 1:N
            r = ranges[ℓ]
            isempty(r) && continue
            n_l = length(r)
            s = states[ℓ]
            @_ka_launch dev n_l _state_segment_data_kernel!(
                view(ax, r), view(ay, r), view(bx, r), view(by, r), view(pv, r),
                view(ka, r), view(kb, r),
                s.x, s.y, s.pv, s.wrapx, s.wrapy,
                s.offsets, s.lengths, s.corners,
                s.contour_of_node, s.local_index, T(P_inv[m, ℓ]), n_l)
        end
        seg = SegmentData(ax, ay, bx, by, pv, ka, kb)
        # Velocity kernels overwrite mode_vx/mode_vy, so they are reused across modes.
        # Branch to a concrete kernel type so the launch is statically dispatched
        # (no Union-typed `mode_kernel` per mode).
        if abs(lam) < eps(T) * 100
            _ka_apply_velocity!(mode_vx, mode_vy, tx, ty, seg, EulerKernel(), domain, dev, ws)
        else
            _ka_apply_velocity!(mode_vx, mode_vy, tx, ty, seg,
                                QGKernel(one(T) / sqrt(abs(lam))), domain, dev, ws)
        end

        for ℓ in 1:N
            w = P[ℓ, m]
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

const _BETA_WS_TLS_KEY = :contourdynamics_beta_plane_velocity_workspace

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
                                   reference::Vector{PVContour{T}}) where {T}
    store = task_local_storage()
    key = (_BETA_WS_TLS_KEY, T, typeof(dev))
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
                                  dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel
    gws = _get_state_workspace(dev, T, N)
    bws = _get_beta_plane_workspace(dev, T, N, kernel.reference_contours)
    return _beta_plane_velocity_with_ws!(vel, gws, bws, state, kernel, domain, dev, N)
end
