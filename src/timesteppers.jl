"""
    _collect_all_nodes!(buf, prob::ContourProblem)

Collect all nodes into pre-allocated buffer `buf` (in-place, non-allocating).
"""
# The timestepper path works on flat node/velocity buffers. These helpers keep
# the gather/scatter code explicit without repeating contour traversal details
# in every RK4/leapfrog stage.
@kernel function _copy_nodes_to_flat_ka!(dest, src, offset)
    i = @index(Global)
    dest[offset + i] = src[i]
end

@kernel function _copy_flat_to_nodes_ka!(dest, src, offset)
    i = @index(Global)
    dest[i] = src[offset + i]
end

@kernel function _scatter_shifted_slice_ka!(dest, base, delta, offset, scale)
    i = @index(Global)
    dest[i] = base[offset + i] + scale * delta[offset + i]
end

@inline function _flat_contour_ranges(contours, offset::Int=0)
    ranges = Vector{UnitRange{Int}}(undef, length(contours))
    idx = offset + 1
    for i in eachindex(contours)
        n = nnodes(contours[i])
        ranges[i] = idx:(idx + n - 1)
        idx += n
    end
    return ranges
end

function _build_prob_ranges(prob::ContourProblem)
    [_flat_contour_ranges(prob.contours)]
end

function _build_prob_ranges(prob::MultiLayerContourProblem{N}) where {N}
    all_ranges = Vector{Vector{UnitRange{Int}}}(undef, N)
    offset = 0
    for i in 1:N
        all_ranges[i] = _flat_contour_ranges(prob.layers[i], offset)
        offset = isempty(all_ranges[i]) ? offset : last(all_ranges[i][end])
    end
    return all_ranges
end

"""
    _ensure_node_ranges!(stepper, prob)

Refresh the stepper's cached flat node ranges if surgery or remeshing changed
the contour layout. The returned ranges map each contour's node array onto the
flat buffers used by the timesteppers.
"""
function _ensure_node_ranges!(stepper::AbstractTimeStepper, prob)
    expected = _build_prob_ranges(prob)
    if stepper.node_ranges != expected
        empty!(stepper.node_ranges)
        append!(stepper.node_ranges, expected)
    end
    return stepper.node_ranges
end

@inline function _check_flat_buffer_length(name::AbstractString, len::Int, required::Int)
    len >= required || throw(DimensionMismatch("$name length ($len) must be >= required length ($required)"))
    return nothing
end

function _for_each_contour_range!(f, prob::ContourProblem, ranges::Vector{UnitRange{Int}})
    for (c, r) in zip(prob.contours, ranges)
        f(c, r)
    end
    return nothing
end

function _for_each_contour_range!(f, contours::AbstractVector{<:PVContour}, ranges::Vector{UnitRange{Int}})
    for (c, r) in zip(contours, ranges)
        f(c, r)
    end
    return nothing
end

function _for_each_contour_range!(f, prob::MultiLayerContourProblem{N},
                                  all_ranges::Vector{Vector{UnitRange{Int}}}) where {N}
    for i in 1:N
        for (c, r) in zip(prob.layers[i], all_ranges[i])
            f(c, r)
        end
    end
    return nothing
end

function _ka_copy_nodes_to_flat!(dest, src, offset::Int)
    isempty(src) && return dest
    _ka_stepper_update!(_copy_nodes_to_flat_ka!, length(src), dest, src, offset)
    return dest
end

function _ka_copy_flat_to_nodes!(dest, src, offset::Int)
    isempty(dest) && return dest
    _ka_stepper_update!(_copy_flat_to_nodes_ka!, length(dest), dest, src, offset)
    return dest
end

function _ka_scatter_shifted_slice!(dest, base, delta, offset::Int, scale)
    isempty(dest) && return dest
    _ka_stepper_update!(_scatter_shifted_slice_ka!, length(dest), dest, base, delta, offset, scale)
    return dest
end

function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("buffer", length(buf), N)
    ranges = _flat_contour_ranges(prob.contours)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_copy_nodes_to_flat!(buf, c.nodes, first(r) - 1)
    end
end

function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::ContourProblem,
                             ranges::Vector{UnitRange{Int}}) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("buffer", length(buf), N)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_copy_nodes_to_flat!(buf, c.nodes, first(r) - 1)
    end
end

"""
    _scatter_nodes!(prob::ContourProblem, all_nodes)

Write flat node vector back into contour node arrays.
"""
function _scatter_nodes!(prob::ContourProblem, all_nodes::Vector{SVector{2,T}}) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("all_nodes", length(all_nodes), N)
    ranges = _flat_contour_ranges(prob.contours)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_copy_flat_to_nodes!(c.nodes, all_nodes, first(r) - 1)
    end
end

function _scatter_nodes!(prob::ContourProblem, all_nodes::Vector{SVector{2,T}},
                         ranges::Vector{UnitRange{Int}}) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("all_nodes", length(all_nodes), N)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_copy_flat_to_nodes!(c.nodes, all_nodes, first(r) - 1)
    end
end

"""
    _scatter_shifted!(prob, base, delta, scale)

Write `base[i] + scale * delta[i]` into contour nodes without allocating.
"""
function _scatter_shifted!(prob::ContourProblem, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("base", length(base), N)
    _check_flat_buffer_length("delta", length(delta), N)
    ranges = _flat_contour_ranges(prob.contours)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_scatter_shifted_slice!(c.nodes, base, delta, first(r) - 1, scale)
    end
end

function _scatter_shifted!(prob::ContourProblem, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T,
                           ranges::Vector{UnitRange{Int}}) where {T}
    N = total_nodes(prob)
    _check_flat_buffer_length("base", length(base), N)
    _check_flat_buffer_length("delta", length(delta), N)
    _for_each_contour_range!(prob, ranges) do c, r
        _ka_scatter_shifted_slice!(c.nodes, base, delta, first(r) - 1, scale)
    end
end

@kernel function _rk4_update_ka!(nodes, k1, k2, k3, k4, dt)
    i = @index(Global)
    nodes[i] = nodes[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
end

@kernel function _leapfrog_bootstrap_ka!(nodes_prev, nodes_current, vel_mid, dt)
    i = @index(Global)
    nodes_prev[i] = nodes_current[i]
    nodes_current[i] = nodes_current[i] + dt * vel_mid[i]
end

@kernel function _leapfrog_step_ka!(nodes_prev, nodes_current, vel, dt, nu)
    i = @index(Global)
    y_next = nodes_prev[i] + 2 * dt * vel[i]
    y_filtered = nodes_current[i] + (nu / 2) * (y_next - 2 * nodes_current[i] + nodes_prev[i])
    nodes_prev[i] = y_filtered
    nodes_current[i] = y_next
end

@kernel function _copy_with_offset_ka!(dest, src, offset)
    i = @index(Global)
    dest[offset + i] = src[i]
end

function _ka_stepper_update!(kernel!, ndrange::Int, args...)
    backend = _ka_backend(CPU())
    kernel = kernel!(backend)
    kernel(args...; ndrange=ndrange)
    KernelAbstractions.synchronize(backend)
    return nothing
end

function _ka_copy_with_offset!(dest, src, offset::Int)
    isempty(src) && return dest
    _ka_stepper_update!(_copy_with_offset_ka!, length(src), dest, src, offset)
    return dest
end

@inline function _rk4_stage!(k, prob::ContourProblem, nodes_orig, increment, scale, ranges)
    _scatter_shifted!(prob, nodes_orig, increment, scale, ranges)
    velocity!(k, prob)
    return k
end

@inline function _finish_rk4_step!(prob::ContourProblem, nodes_orig, k1, k2, k3, k4, dt, ranges)
    _ka_stepper_update!(_rk4_update_ka!, length(nodes_orig), nodes_orig, k1, k2, k3, k4, dt)
    _scatter_nodes!(prob, nodes_orig, ranges)
    return prob
end

@inline function _bootstrap_leapfrog!(prob::ContourProblem, stepper::LeapfrogStepper{T},
                                      nodes_current, vel, dt, ranges) where {T}
    _scatter_shifted!(prob, nodes_current, vel, dt / 2, ranges)
    velocity!(stepper.vel_mid, prob)
    _ka_stepper_update!(_leapfrog_bootstrap_ka!, length(nodes_current),
                        stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
    _scatter_nodes!(prob, nodes_current, ranges)
    stepper.initialized = true
    return prob
end

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

    # Save original positions into pre-allocated buffer
    _collect_all_nodes!(nodes_orig, prob, ranges)

    # k1 = v(t, y)
    velocity!(k1, prob)

    # k2 = v(t + dt/2, y + dt/2 * k1)
    _rk4_stage!(k2, prob, nodes_orig, k1, dt / 2, ranges)

    # k3 = v(t + dt/2, y + dt/2 * k2)
    _rk4_stage!(k3, prob, nodes_orig, k2, dt / 2, ranges)

    # k4 = v(t + dt, y + dt * k3)
    _rk4_stage!(k4, prob, nodes_orig, k3, dt, ranges)

    # Update: y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    return _finish_rk4_step!(prob, nodes_orig, k1, k2, k3, k4, dt, ranges)
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
        # Bootstrap with RK2 (midpoint method) so the first step matches the
        # leapfrog scheme's second-order accuracy.
        _bootstrap_leapfrog!(prob, stepper, nodes_current, vel, dt, ranges)
    else
        # Leapfrog: y_{n+1} = y_{n-1} + 2*dt * v(y_n)
        nu = stepper.ra_coeff
        _ka_stepper_update!(_leapfrog_step_ka!, N,
                            stepper.nodes_prev, nodes_current, vel, dt, nu)
        _scatter_nodes!(prob, nodes_current, ranges)
    end

    return prob
end

"""
    resize_buffers!(stepper::RK4Stepper, prob::ContourProblem)

Resize RK4 work arrays after surgery changes node count.
Stepper buffers are always CPU Vector (even for GPU problems), since the GPU
velocity path handles its own device allocation internally.
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

"""
    resize_buffers!(stepper::LeapfrogStepper, prob::ContourProblem)

Resize leapfrog work arrays after surgery. Resets initialization flag
since node correspondence is lost.
Stepper buffers are always CPU Vector (even for GPU problems).
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

"""
    evolve!(prob, stepper, params; nsteps, callbacks=nothing)

Main simulation loop: timestep → surgery (at interval) → callbacks.
"""
# No-op for non-periodic domains
_maybe_wrap_nodes!(::ContourProblem{<:AbstractKernel, UnboundedDomain}) = nothing
# Wrap nodes for periodic domains to maintain Ewald convergence
_maybe_wrap_nodes!(prob::ContourProblem{<:AbstractKernel, <:PeriodicDomain}) = wrap_nodes!(prob)

function evolve!(prob::ContourProblem, stepper::AbstractTimeStepper,
                 params::SurgeryParams; nsteps::Int, callbacks=nothing)
    # Call callbacks at step 0 to capture the initial condition
    if callbacks !== nothing
        for cb in callbacks
            cb(prob, 0)
        end
    end
    for step in 1:nsteps
        if total_nodes(prob) > 0
            timestep!(prob, stepper)
            _maybe_wrap_nodes!(prob)
        end

        if step % params.n_surgery == 0
            old_N = total_nodes(prob)
            surgery!(prob, params)
            if total_nodes(prob) != old_N
                _evict_gpu_workspace!(_gpu_ws_key(prob, old_N))
                resize_buffers!(stepper, prob)
            end
        end

        if callbacks !== nothing
            for cb in callbacks
                cb(prob, step)
            end
        end
    end
    return prob
end

# evolve! without surgery — just timestep + wrap + callbacks.
function evolve!(prob::ContourProblem, stepper::AbstractTimeStepper,
                 ::Nothing; nsteps::Int, callbacks=nothing)
    if callbacks !== nothing
        for cb in callbacks
            cb(prob, 0)
        end
    end
    for step in 1:nsteps
        if total_nodes(prob) > 0
            timestep!(prob, stepper)
            _maybe_wrap_nodes!(prob)
        end
        if callbacks !== nothing
            for cb in callbacks
                cb(prob, step)
            end
        end
    end
    return prob
end

function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::MultiLayerContourProblem{N}) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("buffer", length(buf), Ntot)
    offset = 0
    for i in 1:N
        ranges = _flat_contour_ranges(prob.layers[i])
        _for_each_contour_range!(prob.layers[i], ranges) do c, r
            _ka_copy_nodes_to_flat!(buf, c.nodes, offset + first(r) - 1)
        end
        offset += sum(nnodes(c) for c in prob.layers[i]; init=0)
    end
end

function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::MultiLayerContourProblem{N},
                             all_ranges::Vector{Vector{UnitRange{Int}}}) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("buffer", length(buf), Ntot)
    _for_each_contour_range!(prob, all_ranges) do c, r
        _ka_copy_nodes_to_flat!(buf, c.nodes, first(r) - 1)
    end
end

function _scatter_nodes!(prob::MultiLayerContourProblem{N}, all_nodes::Vector{SVector{2,T}}) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("all_nodes", length(all_nodes), Ntot)
    offset = 0
    for i in 1:N
        ranges = _flat_contour_ranges(prob.layers[i])
        _for_each_contour_range!(prob.layers[i], ranges) do c, r
            _ka_copy_flat_to_nodes!(c.nodes, all_nodes, offset + first(r) - 1)
        end
        offset += sum(nnodes(c) for c in prob.layers[i]; init=0)
    end
end

function _scatter_nodes!(prob::MultiLayerContourProblem{N}, all_nodes::Vector{SVector{2,T}},
                         all_ranges::Vector{Vector{UnitRange{Int}}}) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("all_nodes", length(all_nodes), Ntot)
    _for_each_contour_range!(prob, all_ranges) do c, r
        _ka_copy_flat_to_nodes!(c.nodes, all_nodes, first(r) - 1)
    end
end

"""
    _collect_velocities!(flat, vel)

Flatten a per-layer velocity tuple into a single contiguous buffer in layer
order. The multi-layer steppers use the flat buffer for the same KA update
kernels as the single-layer path.
"""
function _collect_velocities!(flat::Vector{SVector{2,T}}, vel::NTuple{N, Vector{SVector{2,T}}}) where {N, T}
    total = sum(length(vel[i]) for i in 1:N)
    _check_flat_buffer_length("flat", length(flat), total)
    idx = 1
    for i in 1:N
        _ka_copy_with_offset!(flat, vel[i], idx - 1)
        idx += length(vel[i])
    end
    return flat
end

function _make_vel_tuple(prob::MultiLayerContourProblem{N, <:Any, <:Any, T}) where {N, T}
    ntuple(i -> zeros(SVector{2,T}, sum(nnodes(c) for c in prob.layers[i]; init=0)), Val(N))
end

"""Ensure stepper.vel_bufs has the right number and size of buffers for multi-layer problems."""
function _ensure_vel_bufs!(vel_bufs::Vector{Vector{SVector{2, T}}},
                           prob::MultiLayerContourProblem{N}) where {N, T}
    z = zero(SVector{2, T})
    # Grow the outer vector if needed (first call or layer count changed)
    while length(vel_bufs) < N
        push!(vel_bufs, SVector{2, T}[])
    end
    # Resize each layer's buffer to match current node count
    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        resize!(vel_bufs[i], n_layer)
        fill!(vel_bufs[i], z)
    end
    return ntuple(i -> vel_bufs[i], Val(N))
end

function _scatter_shifted!(prob::MultiLayerContourProblem{N}, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("base", length(base), Ntot)
    _check_flat_buffer_length("delta", length(delta), Ntot)
    offset = 0
    for i in 1:N
        ranges = _flat_contour_ranges(prob.layers[i])
        _for_each_contour_range!(prob.layers[i], ranges) do c, r
            _ka_scatter_shifted_slice!(c.nodes, base, delta, offset + first(r) - 1, scale)
        end
        offset += sum(nnodes(c) for c in prob.layers[i]; init=0)
    end
end

function _scatter_shifted!(prob::MultiLayerContourProblem{N}, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T,
                           all_ranges::Vector{Vector{UnitRange{Int}}}) where {N, T}
    Ntot = total_nodes(prob)
    _check_flat_buffer_length("base", length(base), Ntot)
    _check_flat_buffer_length("delta", length(delta), Ntot)
    _for_each_contour_range!(prob, all_ranges) do c, r
        _ka_scatter_shifted_slice!(c.nodes, base, delta, first(r) - 1, scale)
    end
end

@inline function _rk4_stage!(flat_k, vel_tuple, prob::MultiLayerContourProblem,
                             nodes_orig, increment, scale, all_ranges)
    _scatter_shifted!(prob, nodes_orig, increment, scale, all_ranges)
    velocity!(vel_tuple, prob)
    _collect_velocities!(flat_k, vel_tuple)
    return flat_k
end

@inline function _finish_rk4_step!(prob::MultiLayerContourProblem, nodes_orig, k1, k2, k3, k4, dt, all_ranges)
    _ka_stepper_update!(_rk4_update_ka!, length(nodes_orig), nodes_orig, k1, k2, k3, k4, dt)
    _scatter_nodes!(prob, nodes_orig, all_ranges)
    return prob
end

@inline function _bootstrap_leapfrog!(prob::MultiLayerContourProblem, stepper::LeapfrogStepper{T},
                                      nodes_current, flat_vel, vel_tuple, dt, all_ranges) where {T}
    _scatter_shifted!(prob, nodes_current, flat_vel, dt / 2, all_ranges)
    velocity!(vel_tuple, prob)
    _collect_velocities!(stepper.vel_mid, vel_tuple)
    _ka_stepper_update!(_leapfrog_bootstrap_ka!, length(nodes_current),
                        stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
    _scatter_nodes!(prob, nodes_current, all_ranges)
    stepper.initialized = true
    return prob
end

function timestep!(prob::MultiLayerContourProblem{N}, stepper::RK4Stepper{T}) where {N, T}
    dt = stepper.dt
    Ntot = total_nodes(prob)
    k1, k2, k3, k4 = stepper.k1, stepper.k2, stepper.k3, stepper.k4
    nodes_orig = stepper.nodes_buf
    length(k1) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(k1))) < total nodes ($Ntot). Call resize_buffers! first."))
    all_ranges = _ensure_node_ranges!(stepper, prob)
    _collect_all_nodes!(nodes_orig, prob, all_ranges)

    vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)

    # k1
    velocity!(vel_tuple, prob)
    _collect_velocities!(k1, vel_tuple)

    # k2
    _rk4_stage!(k2, vel_tuple, prob, nodes_orig, k1, dt / 2, all_ranges)

    # k3
    _rk4_stage!(k3, vel_tuple, prob, nodes_orig, k2, dt / 2, all_ranges)

    # k4
    _rk4_stage!(k4, vel_tuple, prob, nodes_orig, k3, dt, all_ranges)

    # Update: y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    return _finish_rk4_step!(prob, nodes_orig, k1, k2, k3, k4, dt, all_ranges)
end

function surgery!(prob::MultiLayerContourProblem{N, <:MultiLayerQGKernel{N}, <:AbstractDomain, T}, params::SurgeryParams) where {N, T}
    domain = prob.domain
    # Allocate scratch buffers locally — thread-safe and negligible cost
    # relative to the O(N²) reconnection work that follows.
    _remesh_buf = SVector{2, T}[]
    _arc_buf = T[]
    _vnodes_buf = SVector{2, T}[]
    for i in 1:N
        contours = prob.layers[i]
        for j in eachindex(contours)
            contours[j] = remesh(contours[j], params; _buf=_remesh_buf, _arc_buf=_arc_buf, _vnodes_buf=_vnodes_buf)
        end
        reconnected = false
        max_reconnect_iter = 100
        prev_n_pairs = typemax(Int)
        min_n_pairs = typemax(Int)
        stall_count = 0
        no_improve_count = 0
        for iter in 1:max_reconnect_iter
            idx = build_spatial_index(contours, params.delta, domain)
            close_pairs = find_close_segments(contours, idx, params.delta, domain)
            isempty(close_pairs) && break
            n_pairs = length(close_pairs)
            if n_pairs > prev_n_pairs
                stall_count += 1
            else
                stall_count = 0
            end
            if n_pairs < min_n_pairs
                min_n_pairs = n_pairs
                no_improve_count = 0
            else
                no_improve_count += 1
            end
            if stall_count >= 3 || no_improve_count >= 6
                @warn "surgery!: layer $i reconnection stalled ($n_pairs close pairs, min seen: $min_n_pairs) — stopping early"
                break
            end
            prev_n_pairs = n_pairs
            reconnect!(contours, close_pairs, domain)
            reconnected = true
            if iter == max_reconnect_iter
                @warn "surgery!: layer $i reconnection limit ($max_reconnect_iter) reached with $n_pairs close pairs remaining"
            end
        end
        if reconnected
            for j in eachindex(contours)
                contours[j] = remesh(contours[j], params; _buf=_remesh_buf, _arc_buf=_arc_buf, _vnodes_buf=_vnodes_buf)
            end
        end
        remove_filaments!(contours, params.area_min)
        _check_spanning_proximity(contours, params.delta, domain)
    end
    return prob
end

_maybe_wrap_nodes!(::MultiLayerContourProblem{<:Any, <:Any, UnboundedDomain}) = nothing
_maybe_wrap_nodes!(prob::MultiLayerContourProblem{N, K, D}) where {N, K<:MultiLayerQGKernel{N}, D<:PeriodicDomain} = wrap_nodes!(prob)

function evolve!(prob::MultiLayerContourProblem, stepper::AbstractTimeStepper,
                 params::SurgeryParams; nsteps::Int, callbacks=nothing)
    # Call callbacks at step 0 to capture the initial condition
    if callbacks !== nothing
        for cb in callbacks
            cb(prob, 0)
        end
    end
    for step in 1:nsteps
        if total_nodes(prob) > 0
            timestep!(prob, stepper)
            _maybe_wrap_nodes!(prob)
        end

        if step % params.n_surgery == 0
            old_N = total_nodes(prob)
            surgery!(prob, params)
            if total_nodes(prob) != old_N
                resize_buffers!(stepper, prob)
            end
        end

        if callbacks !== nothing
            for cb in callbacks
                cb(prob, step)
            end
        end
    end
    return prob
end

function evolve!(prob::MultiLayerContourProblem, stepper::AbstractTimeStepper,
                 ::Nothing; nsteps::Int, callbacks=nothing)
    if callbacks !== nothing
        for cb in callbacks
            cb(prob, 0)
        end
    end
    for step in 1:nsteps
        if total_nodes(prob) > 0
            timestep!(prob, stepper)
            _maybe_wrap_nodes!(prob)
        end
        if callbacks !== nothing
            for cb in callbacks
                cb(prob, step)
            end
        end
    end
    return prob
end

function resize_buffers!(stepper::RK4Stepper{T}, prob::MultiLayerContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.k1, N); fill!(stepper.k1, z)
    resize!(stepper.k2, N); fill!(stepper.k2, z)
    resize!(stepper.k3, N); fill!(stepper.k3, z)
    resize!(stepper.k4, N); fill!(stepper.k4, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    empty!(stepper.vel_bufs)  # will be re-populated on next timestep
    empty!(stepper.node_ranges)
    return stepper
end

function timestep!(prob::MultiLayerContourProblem{NL}, stepper::LeapfrogStepper{T}) where {NL, T}
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
        # The layer tuple remains valid here because the half-step only moves
        # nodes; it does not change the layer/node counts.
        _bootstrap_leapfrog!(prob, stepper, nodes_current, flat_vel, vel_tuple, dt, all_ranges)
    else
        nu = stepper.ra_coeff
        _ka_stepper_update!(_leapfrog_step_ka!, Ntot,
                            stepper.nodes_prev, nodes_current, flat_vel, dt, nu)
        _scatter_nodes!(prob, nodes_current, all_ranges)
    end
    return prob
end

function resize_buffers!(stepper::LeapfrogStepper{T}, prob::MultiLayerContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.nodes_prev, N); fill!(stepper.nodes_prev, z)
    resize!(stepper.vel_buf, N); fill!(stepper.vel_buf, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    resize!(stepper.vel_mid, N); fill!(stepper.vel_mid, z)
    stepper.initialized = false
    empty!(stepper.vel_bufs)  # will be re-populated on next timestep
    empty!(stepper.node_ranges)
    return stepper
end
