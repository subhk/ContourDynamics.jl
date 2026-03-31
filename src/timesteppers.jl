"""
    _collect_all_nodes(prob::ContourProblem)

Collect all nodes from all contours into a flat vector.
"""
function _collect_all_nodes(prob::ContourProblem{K,D,T}) where {K,D,T}
    nodes = SVector{2,T}[]
    for c in prob.contours
        append!(nodes, c.nodes)
    end
    return nodes
end

"""
    _collect_all_nodes!(buf, prob::ContourProblem)

Collect all nodes into pre-allocated buffer `buf` (in-place, non-allocating).
"""
function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    length(buf) >= N || throw(DimensionMismatch("buffer length ($(length(buf))) must be >= total nodes ($N)"))
    idx = 1
    for c in prob.contours
        @inbounds for i in 1:nnodes(c)
            buf[idx] = c.nodes[i]
            idx += 1
        end
    end
end

"""
    _scatter_nodes!(prob::ContourProblem, all_nodes)

Write flat node vector back into contour node arrays.
"""
function _scatter_nodes!(prob::ContourProblem, all_nodes::Vector{SVector{2,T}}) where {T}
    N = total_nodes(prob)
    length(all_nodes) >= N || throw(DimensionMismatch("all_nodes length ($(length(all_nodes))) must be >= total nodes ($N)"))
    idx = 1
    for c in prob.contours
        @inbounds for i in 1:nnodes(c)
            c.nodes[i] = all_nodes[idx]
            idx += 1
        end
    end
end

"""
    _scatter_shifted!(prob, base, delta, scale)

Write `base[i] + scale * delta[i]` into contour nodes without allocating.
"""
function _scatter_shifted!(prob::ContourProblem, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T) where {T}
    N = total_nodes(prob)
    length(base) >= N || throw(DimensionMismatch("base length ($(length(base))) must be >= total nodes ($N)"))
    length(delta) >= N || throw(DimensionMismatch("delta length ($(length(delta))) must be >= total nodes ($N)"))
    idx = 1
    for c in prob.contours
        @inbounds for i in 1:nnodes(c)
            c.nodes[i] = base[idx] + scale * delta[idx]
            idx += 1
        end
    end
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

    # Save original positions into pre-allocated buffer
    _collect_all_nodes!(nodes_orig, prob)

    # k1 = v(t, y)
    velocity!(k1, prob)

    # k2 = v(t + dt/2, y + dt/2 * k1)
    _scatter_shifted!(prob, nodes_orig, k1, dt / 2)
    velocity!(k2, prob)

    # k3 = v(t + dt/2, y + dt/2 * k2)
    _scatter_shifted!(prob, nodes_orig, k2, dt / 2)
    velocity!(k3, prob)

    # k4 = v(t + dt, y + dt * k3)
    _scatter_shifted!(prob, nodes_orig, k3, dt)
    velocity!(k4, prob)

    # Update: y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    @inbounds for i in 1:N
        nodes_orig[i] = nodes_orig[i] + (dt / 6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    end
    _scatter_nodes!(prob, nodes_orig)

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
    _collect_all_nodes!(nodes_current, prob)

    vel = stepper.vel_buf
    velocity!(vel, prob)

    if !stepper.initialized
        # Bootstrap with RK2 (midpoint method) for 2nd-order accuracy,
        # matching the leapfrog's order instead of dropping to 1st-order Euler.
        # Half-step: y_mid = y_n + dt/2 * v(y_n)
        _scatter_shifted!(prob, nodes_current, vel, dt / 2)
        velocity!(stepper.vel_mid, prob)
        # Full step: y_{n+1} = y_n + dt * v(y_mid)
        @inbounds for i in 1:N
            stepper.nodes_prev[i] = nodes_current[i]  # save y_n into nodes_prev
            nodes_current[i] = nodes_current[i] + dt * stepper.vel_mid[i]
        end
        _scatter_nodes!(prob, nodes_current)
        stepper.initialized = true
    else
        # Leapfrog: y_{n+1} = y_{n-1} + 2*dt * v(y_n)
        nu = stepper.ra_coeff
        @inbounds for i in 1:N
            y_next = stepper.nodes_prev[i] + 2 * dt * vel[i]
            # Robert-Asselin filter: damp the computational mode by nudging
            # y_n toward the mean of y_{n-1} and y_{n+1}.
            y_filtered = nodes_current[i] + (nu / 2) * (y_next - 2 * nodes_current[i] + stepper.nodes_prev[i])
            stepper.nodes_prev[i] = y_filtered
            nodes_current[i] = y_next
        end
        _scatter_nodes!(prob, nodes_current)
    end

    return prob
end

"""
    resize_buffers!(stepper::RK4Stepper, prob::ContourProblem)

Resize RK4 work arrays after surgery changes node count.
"""
function resize_buffers!(stepper::RK4Stepper{T}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.k1, N); fill!(stepper.k1, z)
    resize!(stepper.k2, N); fill!(stepper.k2, z)
    resize!(stepper.k3, N); fill!(stepper.k3, z)
    resize!(stepper.k4, N); fill!(stepper.k4, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    return stepper
end

"""
    resize_buffers!(stepper::LeapfrogStepper, prob::ContourProblem)

Resize leapfrog work arrays after surgery. Resets initialization flag
since node correspondence is lost.
"""
function resize_buffers!(stepper::LeapfrogStepper{T}, prob::ContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.nodes_prev, N); fill!(stepper.nodes_prev, z)
    resize!(stepper.vel_buf, N); fill!(stepper.vel_buf, z)
    resize!(stepper.nodes_buf, N); fill!(stepper.nodes_buf, z)
    resize!(stepper.vel_mid, N); fill!(stepper.vel_mid, z)
    stepper.initialized = false
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
        if total_nodes(prob) == 0
            # All contours removed by surgery — skip dynamics but still fire
            # callbacks so that recording callbacks observe every step.
            if callbacks !== nothing
                for cb in callbacks
                    cb(prob, step)
                end
            end
            continue
        end
        timestep!(prob, stepper)
        _maybe_wrap_nodes!(prob)

        if step % params.n_surgery == 0
            surgery!(prob, params)
            resize_buffers!(stepper, prob)
        end

        if callbacks !== nothing
            for cb in callbacks
                cb(prob, step)
            end
        end
    end
    return prob
end

function _collect_all_nodes(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    nodes = SVector{2,T}[]
    for i in 1:N
        for c in prob.layers[i]
            append!(nodes, c.nodes)
        end
    end
    return nodes
end

function _collect_all_nodes!(buf::Vector{SVector{2,T}}, prob::MultiLayerContourProblem{N}) where {N, T}
    Ntot = total_nodes(prob)
    length(buf) >= Ntot || throw(DimensionMismatch("buffer length ($(length(buf))) must be >= total nodes ($Ntot)"))
    idx = 1
    for i in 1:N
        for c in prob.layers[i]
            @inbounds for j in 1:nnodes(c)
                buf[idx] = c.nodes[j]
                idx += 1
            end
        end
    end
end

function _scatter_nodes!(prob::MultiLayerContourProblem{N}, all_nodes::Vector{SVector{2,T}}) where {N, T}
    Ntot = total_nodes(prob)
    length(all_nodes) >= Ntot || throw(DimensionMismatch("all_nodes length ($(length(all_nodes))) must be >= total nodes ($Ntot)"))
    idx = 1
    for i in 1:N
        for c in prob.layers[i]
            @inbounds for j in 1:nnodes(c)
                c.nodes[j] = all_nodes[idx]
                idx += 1
            end
        end
    end
end

function _collect_velocities!(flat::Vector{SVector{2,T}}, vel::NTuple{N, Vector{SVector{2,T}}}) where {N, T}
    total = sum(length(vel[i]) for i in 1:N)
    length(flat) >= total || throw(DimensionMismatch("flat length ($(length(flat))) must be >= total velocities ($total)"))
    idx = 1
    for i in 1:N
        @inbounds for j in eachindex(vel[i])
            flat[idx] = vel[i][j]
            idx += 1
        end
    end
    return flat
end

function _make_vel_tuple(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
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
    length(base) >= Ntot || throw(DimensionMismatch("base length ($(length(base))) must be >= total nodes ($Ntot)"))
    length(delta) >= Ntot || throw(DimensionMismatch("delta length ($(length(delta))) must be >= total nodes ($Ntot)"))
    idx = 1
    for i in 1:N
        for c in prob.layers[i]
            @inbounds for j in 1:nnodes(c)
                c.nodes[j] = base[idx] + scale * delta[idx]
                idx += 1
            end
        end
    end
end

function timestep!(prob::MultiLayerContourProblem{N}, stepper::RK4Stepper{T}) where {N, T}
    dt = stepper.dt
    Ntot = total_nodes(prob)
    k1, k2, k3, k4 = stepper.k1, stepper.k2, stepper.k3, stepper.k4
    nodes_orig = stepper.nodes_buf
    length(k1) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(k1))) < total nodes ($Ntot). Call resize_buffers! first."))
    _collect_all_nodes!(nodes_orig, prob)

    vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)

    # k1
    velocity!(vel_tuple, prob)
    _collect_velocities!(k1, vel_tuple)

    # k2
    _scatter_shifted!(prob, nodes_orig, k1, dt / 2)
    velocity!(vel_tuple, prob)
    _collect_velocities!(k2, vel_tuple)

    # k3
    _scatter_shifted!(prob, nodes_orig, k2, dt / 2)
    velocity!(vel_tuple, prob)
    _collect_velocities!(k3, vel_tuple)

    # k4
    _scatter_shifted!(prob, nodes_orig, k3, dt)
    velocity!(vel_tuple, prob)
    _collect_velocities!(k4, vel_tuple)

    # Update: y_{n+1} = y_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
    @inbounds for i in 1:Ntot
        nodes_orig[i] = nodes_orig[i] + (dt / 6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
    end
    _scatter_nodes!(prob, nodes_orig)
    return prob
end

function surgery!(prob::MultiLayerContourProblem{N}, params::SurgeryParams) where {N}
    domain = prob.domain
    for i in 1:N
        contours = prob.layers[i]
        for j in eachindex(contours)
            contours[j] = remesh(contours[j], params)
        end
        reconnected = false
        max_reconnect_iter = 100
        prev_n_pairs = typemax(Int)
        stall_count = 0
        for iter in 1:max_reconnect_iter
            idx = build_spatial_index(contours, params.delta, domain)
            close_pairs = find_close_segments(contours, idx, params.delta, domain)
            isempty(close_pairs) && break
            n_pairs = length(close_pairs)
            if n_pairs > prev_n_pairs
                stall_count += 1
                if stall_count >= 3
                    @warn "surgery!: layer $i reconnection stalled (close pairs increasing: $n_pairs) — stopping early"
                    break
                end
            else
                stall_count = 0
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
                contours[j] = remesh(contours[j], params)
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
        if total_nodes(prob) == 0
            if callbacks !== nothing
                for cb in callbacks
                    cb(prob, step)
                end
            end
            continue
        end
        timestep!(prob, stepper)
        _maybe_wrap_nodes!(prob)
        if step % params.n_surgery == 0
            surgery!(prob, params)
            resize_buffers!(stepper, prob)
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
    return stepper
end

function timestep!(prob::MultiLayerContourProblem{NL}, stepper::LeapfrogStepper{T}) where {NL, T}
    dt = stepper.dt
    Ntot = total_nodes(prob)
    nodes_current = stepper.nodes_buf
    length(nodes_current) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(nodes_current))) < total nodes ($Ntot). Call resize_buffers! first."))
    _collect_all_nodes!(nodes_current, prob)

    vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)
    velocity!(vel_tuple, prob)
    flat_vel = stepper.vel_buf
    _collect_velocities!(flat_vel, vel_tuple)

    if !stepper.initialized
        _scatter_shifted!(prob, nodes_current, flat_vel, dt / 2)
        vel_tuple = _ensure_vel_bufs!(stepper.vel_bufs, prob)
        velocity!(vel_tuple, prob)
        _collect_velocities!(stepper.vel_mid, vel_tuple)
        @inbounds for i in 1:Ntot
            stepper.nodes_prev[i] = nodes_current[i]
            nodes_current[i] = nodes_current[i] + dt * stepper.vel_mid[i]
        end
        _scatter_nodes!(prob, nodes_current)
        stepper.initialized = true
    else
        nu = stepper.ra_coeff
        @inbounds for i in 1:Ntot
            y_next = stepper.nodes_prev[i] + 2 * dt * flat_vel[i]
            y_filtered = nodes_current[i] + (nu / 2) * (y_next - 2 * nodes_current[i] + stepper.nodes_prev[i])
            stepper.nodes_prev[i] = y_filtered
            nodes_current[i] = y_next
        end
        _scatter_nodes!(prob, nodes_current)
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
    return stepper
end
