# Time integration orchestration.
#
# The heavy lifting for flattening/scattering nodes and device-compatible
# updates lives in `evolution_buffers.jl`; this file wires those utilities into
# RK4/leapfrog timestepping, buffer resizing after surgery, and the public
# `evolve!` loop.
include("evolution_buffers.jl")

"""
    timestep!(prob::ContourProblem, stepper::RK4Stepper)

Advance all contour nodes by one RK4 step.
"""
function timestep!(prob::ContourProblem, stepper::RK4Stepper{T}) where {T}
    dt = stepper.dt
    N = total_nodes(prob)
    k1, k2, k3, k4 = stepper.k1, stepper.k2, stepper.k3, stepper.k4
    nodes_orig = stepper.nodes_buf
    length(k1) >= N || throw(DimensionMismatch("Stepper buffer size ($(length(k1))) < total nodes ($N). Call resize_buffers! first."))
    ranges = _ensure_node_ranges!(stepper, prob)[1]

    _collect_all_nodes!(nodes_orig, prob, ranges)
    velocity!(k1, prob)
    _rk4_stage!(k2, prob, nodes_orig, k1, dt / 2, ranges)
    _rk4_stage!(k3, prob, nodes_orig, k2, dt / 2, ranges)
    _rk4_stage!(k4, prob, nodes_orig, k3, dt, ranges)

    return _finish_rk4_step!(prob, nodes_orig, k1, k2, k3, k4, dt, ranges)
end

function timestep!(prob::ContourProblem{K, D, T, GPU},
                   stepper::RK4Stepper{T}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel{T}},
                                                  D<:Union{UnboundedDomain,PeriodicDomain{T}}}
    _rk4_state_step!(prob.device_state, prob.kernel, prob.domain, stepper, prob.dev)
    return prob
end

"""
    timestep!(prob::ContourProblem, stepper::LeapfrogStepper)

Advance all contour nodes by one leapfrog step.
First step uses RK2 midpoint method to bootstrap.
"""
function timestep!(prob::ContourProblem, stepper::LeapfrogStepper{T}) where {T}
    dt = stepper.dt
    N = total_nodes(prob)
    nodes_current = stepper.nodes_buf
    length(nodes_current) >= N || throw(DimensionMismatch("Stepper buffer size ($(length(nodes_current))) < total nodes ($N). Call resize_buffers! first."))
    ranges = _ensure_node_ranges!(stepper, prob)[1]
    _collect_all_nodes!(nodes_current, prob, ranges)

    vel = stepper.vel_buf
    velocity!(vel, prob)

    if !stepper.initialized
        _bootstrap_leapfrog!(prob, stepper, nodes_current, vel, dt, ranges)
    else
        _cpu_leapfrog_step!(stepper.nodes_prev, nodes_current, vel, dt, stepper.ra_coeff)
        _scatter_nodes!(prob, nodes_current, ranges)
    end

    return prob
end

function timestep!(prob::ContourProblem{K, D, T, GPU},
                   stepper::LeapfrogStepper{T}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel{T}},
                                                       D<:Union{UnboundedDomain,PeriodicDomain{T}}}
    _leapfrog_state_step!(prob.device_state, prob.kernel, prob.domain, stepper, prob.dev)
    return prob
end

function timestep!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, D, T, GPU},
                   stepper::RK4Stepper{T}) where {N, T, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    _rk4_multilayer_state_step!(prob.device_state, prob.kernel, prob.domain, stepper, prob.dev)
    return prob
end

function timestep!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, D, T, GPU},
                   stepper::LeapfrogStepper{T}) where {N, T, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    _leapfrog_multilayer_state_step!(prob.device_state, prob.kernel, prob.domain, stepper, prob.dev)
    return prob
end

function timestep!(prob::MultiLayerContourProblem{N}, stepper::RK4Stepper{T}) where {N, T}
    # Multi-layer velocity is naturally returned per layer; RK4 buffers are flat
    # for compact storage, so each stage gathers/scatters through layer ranges.
    dt = stepper.dt
    Ntot = total_nodes(prob)
    k1, k2, k3, k4 = stepper.k1, stepper.k2, stepper.k3, stepper.k4
    nodes_orig = stepper.nodes_buf
    length(k1) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(k1))) < total nodes ($Ntot). Call resize_buffers! first."))
    all_ranges = _ensure_node_ranges!(stepper, prob)
    _collect_all_nodes!(nodes_orig, prob, all_ranges)

    vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)
    velocity!(vel_tuple, prob)
    _collect_velocities!(k1, vel_tuple)
    _rk4_stage!(k2, vel_tuple, prob, nodes_orig, k1, dt / 2, all_ranges)
    _rk4_stage!(k3, vel_tuple, prob, nodes_orig, k2, dt / 2, all_ranges)
    _rk4_stage!(k4, vel_tuple, prob, nodes_orig, k3, dt, all_ranges)

    return _finish_rk4_step!(prob, nodes_orig, k1, k2, k3, k4, dt, all_ranges)
end

function timestep!(prob::MultiLayerContourProblem{N}, stepper::LeapfrogStepper{T}) where {N, T}
    # Leapfrog keeps one previous flat node vector. For multi-layer problems the
    # per-layer velocity tuple is gathered into the same flat ordering before
    # the device update kernel runs.
    dt = stepper.dt
    Ntot = total_nodes(prob)
    nodes_current = stepper.nodes_buf
    length(nodes_current) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(nodes_current))) < total nodes ($Ntot). Call resize_buffers! first."))
    all_ranges = _ensure_node_ranges!(stepper, prob)
    _collect_all_nodes!(nodes_current, prob, all_ranges)

    vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)
    velocity!(vel_tuple, prob)
    flat_vel = stepper.vel_buf
    _collect_velocities!(flat_vel, vel_tuple)

    if !stepper.initialized
        _bootstrap_leapfrog!(prob, stepper, nodes_current, flat_vel, vel_tuple, dt, all_ranges)
    else
        _cpu_leapfrog_step!(stepper.nodes_prev, nodes_current, flat_vel, dt, stepper.ra_coeff)
        _scatter_nodes!(prob, nodes_current, all_ranges)
    end

    return prob
end

"""
    resize_buffers!(stepper::RK4Stepper, prob::ContourProblem)

Resize RK4 work arrays after surgery changes node count.
The buffers live on the stepper's device (a `Vector` for CPU problems, a device
array for GPU problems — see [`RK4Stepper`](@ref)); `resize!` preserves that type.
"""
function resize_buffers!(stepper::RK4Stepper{T}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.k1, N); fill!(stepper.k1, z)
    resize!(stepper.k2, N); fill!(stepper.k2, z)
    resize!(stepper.k3, N); fill!(stepper.k3, z)
    resize!(stepper.k4, N); fill!(stepper.k4, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    empty!(stepper.node_ranges)
    return stepper
end

function resize_buffers!(stepper::RK4Stepper{T}, prob::MultiLayerContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.k1, N); fill!(stepper.k1, z)
    resize!(stepper.k2, N); fill!(stepper.k2, z)
    resize!(stepper.k3, N); fill!(stepper.k3, z)
    resize!(stepper.k4, N); fill!(stepper.k4, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    empty!(stepper.vel_bufs)
    empty!(stepper.node_ranges)
    return stepper
end

"""
    resize_buffers!(stepper::LeapfrogStepper, prob::ContourProblem)

Resize leapfrog work arrays after surgery. Resets initialization flag
since node correspondence is lost. Buffers live on the stepper's device (CPU
`Vector` or device array) and `resize!` preserves that type.
"""
function resize_buffers!(stepper::LeapfrogStepper{T}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.nodes_prev, N); fill!(stepper.nodes_prev, z)
    resize!(stepper.vel_buf, N); fill!(stepper.vel_buf, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    resize!(stepper.vel_mid, N); fill!(stepper.vel_mid, z)
    stepper.initialized = false
    empty!(stepper.node_ranges)
    return stepper
end

function resize_buffers!(stepper::LeapfrogStepper{T}, prob::MultiLayerContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.nodes_prev, N); fill!(stepper.nodes_prev, z)
    resize!(stepper.vel_buf, N); fill!(stepper.vel_buf, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    resize!(stepper.vel_mid, N); fill!(stepper.vel_mid, z)
    stepper.initialized = false
    empty!(stepper.vel_bufs)
    empty!(stepper.node_ranges)
    return stepper
end

# `evolve!` owns the contract that topology changes (normally from surgery) are
# reflected in the reusable timestep buffers. Check every work array rather
# than only the node count before/after surgery: callbacks and device-resident
# topology rewrites can otherwise leave a stale stepper even when the count
# comparison used by the surgery caller misses the transition.
@inline function _stepper_buffers_match(stepper::RK4Stepper, N::Int)
    return length(stepper.k1) == N && length(stepper.k2) == N &&
           length(stepper.k3) == N && length(stepper.k4) == N &&
           length(stepper.nodes_buf) == N
end

@inline function _stepper_buffers_match(stepper::LeapfrogStepper, N::Int)
    return length(stepper.nodes_prev) == N && length(stepper.vel_buf) == N &&
           length(stepper.nodes_buf) == N && length(stepper.vel_mid) == N
end

@inline function _ensure_stepper_buffers!(prob, stepper)
    N = total_nodes(prob)
    _stepper_buffers_match(stepper, N) && return false
    resize_buffers!(stepper, prob)
    _stepper_buffers_match(stepper, N) || throw(DimensionMismatch(
        "Failed to resize stepper buffers to the authoritative node count ($N)."))
    return true
end

"""
    surgery!(prob::MultiLayerContourProblem, params::SurgeryParams)

Apply the same remesh/reconnect/cleanup surgery pass used for single-layer
problems independently to each layer of a multi-layer QG problem. Cross-layer
reconnections are not attempted; coupling enters through velocity and energy
diagnostics, not through topology changes.
"""
function surgery!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, <:AbstractDomain, T}, params::SurgeryParams) where {N, T}
    # Each layer gets the same Dritschel surgery pass used for single-layer
    # problems; cross-layer reconnections are not attempted. Workspace buffers
    # are shared across layers. The shared `_surgery_pass!` driver lives in
    # surgery.jl so single- and multi-layer surgery cannot silently diverge.
    domain = prob.domain
    remesh_buf = SVector{2, T}[]
    arc_buf = T[]
    vnodes_buf = SVector{2, T}[]
    for i in 1:N
        _surgery_pass!(prob.layers[i], domain, params, remesh_buf, arc_buf, vnodes_buf;
                       layer_label=" layer $i")
    end
    return prob
end


_maybe_wrap_nodes!(::ContourProblem{<:AbstractKernel, UnboundedDomain}) = nothing
_maybe_wrap_nodes!(prob::ContourProblem{<:AbstractKernel, <:PeriodicDomain}) = wrap_nodes!(prob)
_maybe_wrap_nodes!(prob::ContourProblem{<:AbstractKernel, PeriodicDomain{T}, T, GPU}) where {T} =
    _wrap_state_nodes!(prob.device_state, prob.domain, prob.dev)
# Device-agnostic: unbounded domains never wrap, on any device. A separate
# GPU method here would be equally specific, not more specific, and the
# resulting ambiguity would make wrapping unresolvable for GPU multi-layer
# unbounded problems.
_maybe_wrap_nodes!(::MultiLayerContourProblem{<:Any, <:Any, UnboundedDomain}) = nothing
_maybe_wrap_nodes!(prob::MultiLayerContourProblem{N, K, <:PeriodicDomain, T, GPU}) where {N, K, T} =
    wrap_nodes!(prob)
_maybe_wrap_nodes!(prob::MultiLayerContourProblem{N, K, D}) where {N, K<:MultiLayerQGKernel{N}, D<:PeriodicDomain} = wrap_nodes!(prob)

# Stepper-aware wrapping. Steppers without position history delegate to the
# node-only methods above. LeapfrogStepper keeps the Robert–Asselin-filtered
# previous node level in `nodes_prev`; the lattice shift applied to a contour's
# nodes must also be applied to its slice of that history, or the next step
# mixes coordinates from different periodic images and physically displaces
# the contour by a non-lattice offset.
#
# The CPU methods below MUST stay explicitly `CPU`-constrained. They read the
# host `prob.contours` / `prob.layers` shadow, which a GPU problem populates
# once at construction and never updates. If they were left device-agnostic,
# Julia would find neither them nor the GPU methods more specific (they
# constrain different type parameters) and would silently pick the CPU one for
# GPU problems, wrapping stale host geometry and corrupting the device-side
# leapfrog history. `test_device.jl` asserts the resolution.
_maybe_wrap_nodes!(prob, ::AbstractTimeStepper) = _maybe_wrap_nodes!(prob)

_maybe_wrap_nodes!(prob::ContourProblem{<:AbstractKernel, PeriodicDomain{T}, T, CPU},
                   stepper::LeapfrogStepper{T}) where {T} =
    _wrap_nodes_with_history!(prob, stepper)

function _maybe_wrap_nodes!(prob::ContourProblem{<:AbstractKernel, PeriodicDomain{T}, T, GPU},
                            stepper::LeapfrogStepper{T}) where {T}
    _wrap_state_nodes_with_history!(prob.device_state, prob.domain, stepper, prob.dev)
    return nothing
end

_maybe_wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, CPU},
                   stepper::LeapfrogStepper{T}) where {N, K<:MultiLayerQGKernel{N}, T} =
    _wrap_nodes_with_history!(prob, stepper)

function _maybe_wrap_nodes!(prob::MultiLayerContourProblem{N, K, <:PeriodicDomain, T, GPU},
                            stepper::LeapfrogStepper{T}) where {N, K, T}
    states = prob.device_state
    ranges = _layer_state_ranges(states)
    for ℓ in 1:N
        r = ranges[ℓ]
        prev = isempty(r) ? stepper.nodes_prev : view(stepper.nodes_prev, r)
        _wrap_state_nodes_with_history!(states[ℓ], prob.domain, stepper, prob.dev, prev)
    end
    return nothing
end

function _wrap_nodes_with_history!(prob::ContourProblem{<:AbstractKernel, PeriodicDomain{T}, T, CPU},
                                   stepper::LeapfrogStepper{T}) where {T}
    stepper.initialized || return wrap_nodes!(prob)
    domain = prob.domain
    ranges = _ensure_node_ranges!(stepper, prob)[1]
    nodes_prev = stepper.nodes_prev
    for (ci, c) in enumerate(prob.contours)
        is_spanning(c) && continue
        shift = contour_periodic_shift(c, domain)
        iszero(shift) && continue
        @inbounds for k in eachindex(c.nodes)
            c.nodes[k] += shift
        end
        @inbounds for k in ranges[ci]
            nodes_prev[k] += shift
        end
    end
    return prob
end

function _wrap_nodes_with_history!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, CPU},
                                   stepper::LeapfrogStepper{T}) where {N, K, T}
    stepper.initialized || return wrap_nodes!(prob)
    domain = prob.domain
    all_ranges = _ensure_node_ranges!(stepper, prob)
    nodes_prev = stepper.nodes_prev
    for ℓ in 1:N
        layer_ranges = all_ranges[ℓ]
        for (ci, c) in enumerate(prob.layers[ℓ])
            is_spanning(c) && continue
            shift = contour_periodic_shift(c, domain)
            iszero(shift) && continue
            @inbounds for k in eachindex(c.nodes)
                c.nodes[k] += shift
            end
            @inbounds for k in layer_ranges[ci]
                nodes_prev[k] += shift
            end
        end
    end
    return prob
end

function wrap_nodes!(prob::ContourProblem{<:AbstractKernel, PeriodicDomain{T}, T, GPU}) where {T}
    _wrap_state_nodes!(prob.device_state, prob.domain, prob.dev)
    return prob
end

# The unbounded multi-layer no-op lives in domains.jl and is device-agnostic;
# duplicating it for GPU here would be ambiguous rather than more specific.
function wrap_nodes!(prob::MultiLayerContourProblem{N, K, PeriodicDomain{T}, T, GPU}) where {N, K, T}
    for ℓ in 1:N
        _wrap_state_nodes!(prob.device_state[ℓ], prob.domain, prob.dev)
    end
    return prob
end

@inline function _run_callbacks(callbacks, prob, step::Int)
    callbacks === nothing && return nothing
    for cb in callbacks
        cb(prob, step)
    end
    return nothing
end

# Surgery remeshes every pass, moving nodes along their contours even when the
# total node count is preserved. Any stepper history recorded at the old node
# positions is therefore stale after every pass — not just after resizes — so
# steppers with history must re-bootstrap (as the SurgeryParams docstring
# promises). RK4 keeps no history and needs nothing.
@inline _invalidate_history!(::AbstractTimeStepper) = nothing
@inline function _invalidate_history!(stepper::LeapfrogStepper)
    stepper.initialized = false
    return nothing
end

@inline function _handle_post_surgery!(prob::ContourProblem, stepper, ::Int)
    if !_ensure_stepper_buffers!(prob, stepper)
        _invalidate_history!(stepper)
    end
    return nothing
end

@inline function _handle_post_surgery!(prob::MultiLayerContourProblem, stepper, ::Int)
    if !_ensure_stepper_buffers!(prob, stepper)
        _invalidate_history!(stepper)
    end
    return nothing
end

@inline _maybe_apply_surgery!(prob, stepper, ::Nothing, step::Int) = nothing

@inline function _maybe_apply_surgery!(prob, stepper, params::SurgeryParams, step::Int)
    step % params.n_surgery == 0 || return nothing
    old_N = total_nodes(prob)
    surgery!(prob, params)
    _handle_post_surgery!(prob, stepper, old_N)
    return nothing
end

"""
    evolve!(prob, stepper, params; nsteps, callbacks=nothing, step_offset=0,
            run_initial_callbacks=true)

Main simulation loop: timestep → surgery (at interval) → callbacks.
`step_offset` makes callback and surgery step numbers global across batched
calls. Set `run_initial_callbacks=false` after the first batch so the boundary
step is not emitted twice.
"""
function evolve!(prob::Union{ContourProblem, MultiLayerContourProblem},
                 stepper::AbstractTimeStepper,
                 params::Union{SurgeryParams, Nothing};
                 nsteps::Int,
                 callbacks=nothing,
                 step_offset::Int=0,
                 run_initial_callbacks::Bool=true)
    step_offset >= 0 || throw(ArgumentError("step_offset must be non-negative, got $step_offset"))
    run_initial_callbacks && _run_callbacks(callbacks, prob, step_offset)
    for local_step in 1:nsteps
        step = step_offset + local_step
        if total_nodes(prob) > 0
            _ensure_stepper_buffers!(prob, stepper)
            timestep!(prob, stepper)
            _maybe_wrap_nodes!(prob, stepper)
        end
        _maybe_apply_surgery!(prob, stepper, params, step)
        _run_callbacks(callbacks, prob, step)
    end
    return prob
end
