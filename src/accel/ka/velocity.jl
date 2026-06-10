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
                       dev::GPU) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T}
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

    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    ax = device_zeros(dev, T, total); ay = device_zeros(dev, T, total)
    bx = device_zeros(dev, T, total); by = device_zeros(dev, T, total)
    pv = device_zeros(dev, T, total); ka = device_zeros(dev, T, total); kb = device_zeros(dev, T, total)
    tx = device_zeros(dev, T, total); ty = device_zeros(dev, T, total)
    mode_vx = device_zeros(dev, T, total); mode_vy = device_zeros(dev, T, total)
    vel_x = device_zeros(dev, T, total); vel_y = device_zeros(dev, T, total)

    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        copyto!(view(tx, r), states[ℓ].x)
        copyto!(view(ty, r), states[ℓ].y)
    end

    for m in 1:N
        lam = evals[m]
        mode_kernel = abs(lam) < eps(T) * 100 ? EulerKernel() :
                                                 QGKernel(one(T) / sqrt(abs(lam)))
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
        _ka_apply_velocity!(mode_vx, mode_vy, tx, ty, seg, mode_kernel, domain, dev)

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
