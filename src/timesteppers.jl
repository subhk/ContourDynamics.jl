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
    _scatter_nodes!(prob::ContourProblem, all_nodes)

Write flat node vector back into contour node arrays.
"""
function _scatter_nodes!(prob::ContourProblem, all_nodes::Vector{SVector{2,T}}) where {T}
    idx = 1
    for c in prob.contours
        n = nnodes(c)
        for i in 1:n
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

    # Save original positions (one unavoidable allocation per step)
    nodes_orig = _collect_all_nodes(prob)

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
First step uses forward Euler to bootstrap.
"""
function timestep!(prob::ContourProblem, stepper::LeapfrogStepper{T}) where {T}
    dt = stepper.dt
    N = total_nodes(prob)
    nodes_current = _collect_all_nodes(prob)

    vel = Vector{SVector{2,T}}(undef, N)
    velocity!(vel, prob)

    if !stepper.initialized
        # Bootstrap with RK2 (midpoint method) for 2nd-order accuracy,
        # matching the leapfrog's order instead of dropping to 1st-order Euler.
        # Half-step: y_mid = y_n + dt/2 * v(y_n)
        _scatter_shifted!(prob, nodes_current, vel, dt / 2)
        vel_mid = Vector{SVector{2,T}}(undef, N)
        velocity!(vel_mid, prob)
        # Full step: y_{n+1} = y_n + dt * v(y_mid)
        stepper.nodes_prev .= nodes_current
        @inbounds for i in 1:N
            nodes_current[i] = nodes_current[i] + dt * vel_mid[i]
        end
        _scatter_nodes!(prob, nodes_current)
        stepper.initialized = true
    else
        # Leapfrog: y_{n+1} = y_{n-1} + 2*dt * v(y_n)
        @inbounds for i in 1:N
            nodes_current[i], stepper.nodes_prev[i] =
                stepper.nodes_prev[i] + 2 * dt * vel[i], nodes_current[i]
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
    stepper.initialized = false
    return stepper
end

"""
    evolve!(prob, stepper, params; nsteps, callbacks=nothing)

Main simulation loop: timestep → surgery (at interval) → callbacks.
"""
function evolve!(prob::ContourProblem, stepper::AbstractTimeStepper,
                 params::SurgeryParams; nsteps::Int, callbacks=nothing)
    for step in 1:nsteps
        timestep!(prob, stepper)

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

function _scatter_nodes!(prob::MultiLayerContourProblem{N}, all_nodes::Vector{SVector{2,T}}) where {N, T}
    idx = 1
    for i in 1:N
        for c in prob.layers[i]
            for j in 1:nnodes(c)
                c.nodes[j] = all_nodes[idx]
                idx += 1
            end
        end
    end
end

function _collect_velocities!(flat::Vector{SVector{2,T}}, vel::NTuple{N, Vector{SVector{2,T}}}) where {N, T}
    idx = 1
    for i in 1:N
        for j in eachindex(vel[i])
            flat[idx] = vel[i][j]
            idx += 1
        end
    end
    return flat
end

function _make_vel_tuple(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    ntuple(i -> zeros(SVector{2,T}, sum(nnodes(c) for c in prob.layers[i]; init=0)), Val(N))
end

function _scatter_shifted!(prob::MultiLayerContourProblem{N}, base::Vector{SVector{2,T}},
                           delta::Vector{SVector{2,T}}, scale::T) where {N, T}
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
    nodes_orig = _collect_all_nodes(prob)

    vel_tuple = _make_vel_tuple(prob)

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
    for i in 1:N
        contours = prob.layers[i]
        for j in eachindex(contours)
            contours[j] = remesh(contours[j], params)
        end
        for _ in 1:100
            idx = build_spatial_index(contours, params.delta)
            close_pairs = find_close_segments(contours, idx, params.delta)
            isempty(close_pairs) && break
            reconnect!(contours, close_pairs)
        end
        remove_filaments!(contours, params.area_min)
    end
    return prob
end

function evolve!(prob::MultiLayerContourProblem, stepper::AbstractTimeStepper,
                 params::SurgeryParams; nsteps::Int, callbacks=nothing)
    for step in 1:nsteps
        timestep!(prob, stepper)
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
    return stepper
end

function timestep!(prob::MultiLayerContourProblem{NL}, stepper::LeapfrogStepper{T}) where {NL, T}
    dt = stepper.dt
    Ntot = total_nodes(prob)
    nodes_current = _collect_all_nodes(prob)

    vel_tuple = _make_vel_tuple(prob)
    velocity!(vel_tuple, prob)
    flat_vel = Vector{SVector{2,T}}(undef, Ntot)
    _collect_velocities!(flat_vel, vel_tuple)

    if !stepper.initialized
        _scatter_shifted!(prob, nodes_current, flat_vel, dt / 2)
        vel_tuple2 = _make_vel_tuple(prob)
        velocity!(vel_tuple2, prob)
        flat_vel_mid = Vector{SVector{2,T}}(undef, Ntot)
        _collect_velocities!(flat_vel_mid, vel_tuple2)
        stepper.nodes_prev .= nodes_current
        @inbounds for i in 1:Ntot
            nodes_current[i] = nodes_current[i] + dt * flat_vel_mid[i]
        end
        _scatter_nodes!(prob, nodes_current)
        stepper.initialized = true
    else
        @inbounds for i in 1:Ntot
            nodes_current[i], stepper.nodes_prev[i] =
                stepper.nodes_prev[i] + 2 * dt * flat_vel[i], nodes_current[i]
        end
        _scatter_nodes!(prob, nodes_current)
    end
    return prob
end

function resize_buffers!(stepper::LeapfrogStepper{T}, prob::MultiLayerContourProblem) where {T}
    N = total_nodes(prob)
    z = zero(SVector{2, T})
    resize!(stepper.nodes_prev, N); fill!(stepper.nodes_prev, z)
    stepper.initialized = false
    return stepper
end
