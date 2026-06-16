# Buffer utilities shared by RK4 and leapfrog timesteppers.
#
# Contour geometry is stored as nested vectors (`contours -> nodes` or
# `layers -> contours -> nodes`) because that is natural for surgery and user
# inspection. Timesteppers, by contrast, need flat work arrays so RK stages and
# KernelAbstractions update kernels can operate over a single contiguous index
# range. The functions in this file define the mapping between those two views:
#
#   gather:  contour nodes -> flat node buffer
#   scatter: flat node buffer -> contour nodes
#   ranges:  cached contour/layer slices inside the flat buffer
#
# Keeping this mapping isolated prevents RK4/leapfrog code from duplicating
# contour traversal logic and makes post-surgery resizing easier to reason about.

"""
    _collect_all_nodes!(buf, prob)

Collect the current contour nodes into pre-allocated flat buffer `buf`.
"""
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

@kernel function _state_nodes_to_flat_ka!(dest, x, y)
    i = @index(Global)
    T = eltype(x)
    dest[i] = SVector{2,T}(x[i], y[i])
end

@kernel function _flat_to_state_nodes_ka!(x, y, src)
    i = @index(Global)
    node = src[i]
    x[i] = node[1]
    y[i] = node[2]
end

@kernel function _state_scatter_shifted_ka!(x, y, base, delta, scale)
    i = @index(Global)
    node = base[i] + scale * delta[i]
    x[i] = node[1]
    y[i] = node[2]
end

@kernel function _rk4_state_update_ka!(x, y, nodes_orig, k1, k2, k3, k4, dt)
    i = @index(Global)
    node = nodes_orig[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    x[i] = node[1]
    y[i] = node[2]
end

@kernel function _periodic_state_shifts_ka!(shiftx, shifty, x, y, wrapx, wrapy,
                                            offsets, lengths, Lx, Ly, ncontours)
    ci = @index(Global)
    if ci <= ncontours
        T = eltype(x)
        shiftx[ci] = zero(T)
        shifty[ci] = zero(T)

        n = lengths[ci]
        if iszero(wrapx[ci]) && iszero(wrapy[ci]) && n > 0
            off = offsets[ci]
            Lx2 = T(2) * Lx
            Ly2 = T(2) * Ly

            # Area-weighted centroid on minimum-image-unwrapped coordinates
            # (anchored at the first node), matching the CPU
            # `_periodic_reference_point`. Unwrapping makes this robust to
            # contours stored straddling a periodic seam; for a contiguous
            # contour the unwrap is the identity.
            p0x = x[off]
            p0y = y[off]
            area2 = zero(T)
            cx_num = zero(T)
            cy_num = zero(T)
            sx = zero(T)
            sy = zero(T)
            for li in 1:n
                g = off + li - 1
                ng = li < n ? g + 1 : off
                dxi = x[g] - p0x
                dyi = y[g] - p0y
                uix = p0x + dxi - Lx2 * round(dxi / Lx2)
                uiy = p0y + dyi - Ly2 * round(dyi / Ly2)
                dxj = x[ng] - p0x
                dyj = y[ng] - p0y
                ujx = p0x + dxj - Lx2 * round(dxj / Lx2)
                ujy = p0y + dyj - Ly2 * round(dyj / Ly2)
                cross = uix * ujy - ujx * uiy
                area2 += cross
                cx_num += (uix + ujx) * cross
                cy_num += (uiy + ujy) * cross
                sx += uix
                sy += uiy
            end

            refx = zero(T)
            refy = zero(T)
            if n < 3 || abs(area2) <= T(2) * eps(T)
                refx = sx / T(n)
                refy = sy / T(n)
            else
                inv3A2 = one(T) / (T(3) * area2)
                refx = cx_num * inv3A2
                refy = cy_num * inv3A2
            end

            wrapped_x = refx - Lx2 * floor((refx + Lx) / Lx2)
            wrapped_y = refy - Ly2 * floor((refy + Ly) / Ly2)
            shiftx[ci] = wrapped_x - refx
            shifty[ci] = wrapped_y - refy
        end
    end
end

@kernel function _apply_state_shifts_ka!(x, y, shiftx, shifty, contour_of_node, total_nodes)
    i = @index(Global)
    if i <= total_nodes
        ci = contour_of_node[i]
        x[i] += shiftx[ci]
        y[i] += shifty[ci]
    end
end

@inline function _flat_contour_ranges(contours, offset::Int=0)
    # Return one flat-buffer slice per contour. `offset` lets multi-layer
    # problems append each layer's ranges after the previous layer.
    ranges = Vector{UnitRange{Int}}(undef, length(contours))
    idx = offset + 1
    for i in eachindex(contours)
        n = nnodes(contours[i])
        ranges[i] = idx:(idx + n - 1)
        idx += n
    end
    return ranges
end

@inline _layer_node_count(contours) = sum(nnodes(c) for c in contours; init=0)

function _build_prob_ranges(prob::ContourProblem)
    [_flat_contour_ranges(prob.contours)]
end

function _build_prob_ranges(prob::MultiLayerContourProblem{N}) where {N}
    # Multi-layer ranges are nested as all_ranges[layer][contour].
    # The flat buffer still stores all layers in one contiguous node order.
    all_ranges = Vector{Vector{UnitRange{Int}}}(undef, N)
    offset = 0
    for i in 1:N
        all_ranges[i] = _flat_contour_ranges(prob.layers[i], offset)
        offset = isempty(all_ranges[i]) ? offset : last(all_ranges[i][end])
    end
    return all_ranges
end

function _contour_ranges_match(ranges::Vector{UnitRange{Int}}, contours, offset::Int=0)
    length(ranges) == length(contours) || return false
    idx = offset + 1
    for i in eachindex(contours)
        n = nnodes(contours[i])
        r = ranges[i]
        (first(r) == idx && last(r) == idx + n - 1) || return false
        idx += n
    end
    return true
end

function _node_ranges_match(cached::Vector{Vector{UnitRange{Int}}}, prob::ContourProblem)
    length(cached) == 1 || return false
    return _contour_ranges_match(cached[1], prob.contours)
end

function _node_ranges_match(cached::Vector{Vector{UnitRange{Int}}}, prob::MultiLayerContourProblem{N}) where {N}
    length(cached) == N || return false
    offset = 0
    for i in 1:N
        _contour_ranges_match(cached[i], prob.layers[i], offset) || return false
        offset += _layer_node_count(prob.layers[i])
    end
    return true
end

"""
    _ensure_node_ranges!(stepper, prob)

Refresh the stepper's cached flat node ranges if surgery or remeshing changed
the contour layout. The returned ranges map each contour's node array onto the
flat buffers used by the timesteppers.
"""
function _ensure_node_ranges!(stepper::AbstractTimeStepper, prob)
    _node_ranges_match(stepper.node_ranges, prob) && return stepper.node_ranges
    expected = _build_prob_ranges(prob)
    empty!(stepper.node_ranges)
    append!(stepper.node_ranges, expected)
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

function _ka_stepper_update!(kernel!, ndrange::Int, args...)
    # Stepper update kernels currently run on the CPU backend even for GPU
    # velocity problems. GPU velocity kernels manage their own device packing,
    # while stepper buffers remain CPU-owned for surgery compatibility.
    @_ka_launch CPU() ndrange kernel!(args...)
    return nothing
end

function _ka_stepper_update!(dev::AbstractDevice, kernel!, ndrange::Int, args...)
    @_ka_launch dev ndrange kernel!(args...)
    return nothing
end

# CPU buffer/stepper updates run as plain loops rather than through
# `@_ka_launch CPU()`: a KernelAbstractions launch costs ~288 B (closure +
# ndrange object) plus a synchronize on every call, which is pure overhead for
# these trivial elementwise vector operations and accumulates across the 4
# velocity stages of every RK4/leapfrog step. GPU problems keep the
# device-resident `@_ka_launch dev` path via the methods above.
@inline function _cpu_copy_to_flat!(dest, src, offset::Int)
    @inbounds for i in eachindex(src)
        dest[offset + i] = src[i]
    end
    return dest
end

@inline function _cpu_copy_from_flat!(dest, src, offset::Int)
    @inbounds for i in eachindex(dest)
        dest[i] = src[offset + i]
    end
    return dest
end

@inline function _cpu_scatter_shifted!(dest, base, delta, offset::Int, scale)
    @inbounds for i in eachindex(dest)
        dest[i] = base[offset + i] + scale * delta[offset + i]
    end
    return dest
end

@inline function _cpu_rk4_combine!(nodes, k1, k2, k3, k4, dt)
    @inbounds for i in eachindex(nodes)
        nodes[i] = nodes[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
    end
    return nodes
end

@inline function _cpu_leapfrog_bootstrap!(nodes_prev, nodes_current, vel_mid, dt)
    @inbounds for i in eachindex(nodes_current)
        nodes_prev[i] = nodes_current[i]
        nodes_current[i] = nodes_current[i] + dt * vel_mid[i]
    end
    return nothing
end

@inline function _cpu_leapfrog_step!(nodes_prev, nodes_current, vel, dt, nu)
    @inbounds for i in eachindex(nodes_current)
        y_next = nodes_prev[i] + 2 * dt * vel[i]
        y_filtered = nodes_current[i] + (nu / 2) * (y_next - 2 * nodes_current[i] + nodes_prev[i])
        nodes_prev[i] = y_filtered
        nodes_current[i] = y_next
    end
    return nothing
end

function _ka_copy_nodes_to_flat!(dest, src, offset::Int)
    isempty(src) && return dest
    _cpu_copy_to_flat!(dest, src, offset)
    return dest
end

function _ka_copy_flat_to_nodes!(dest, src, offset::Int)
    isempty(dest) && return dest
    _cpu_copy_from_flat!(dest, src, offset)
    return dest
end

function _ka_scatter_shifted_slice!(dest, base, delta, offset::Int, scale)
    isempty(dest) && return dest
    _cpu_scatter_shifted!(dest, base, delta, offset, scale)
    return dest
end

function _ka_copy_with_offset!(dest, src, offset::Int)
    isempty(src) && return dest
    _cpu_copy_to_flat!(dest, src, offset)
    return dest
end

function _collect_state_nodes!(buf::AbstractVector{SVector{2,T}},
                               state::DeviceContourState{T},
                               dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    _check_flat_buffer_length("buffer", length(buf), N)
    N == 0 && return buf
    _ka_stepper_update!(dev, _state_nodes_to_flat_ka!, N, buf, state.x, state.y)
    return buf
end

function _scatter_state_nodes!(state::DeviceContourState{T},
                               nodes::AbstractVector{SVector{2,T}},
                               dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    _check_flat_buffer_length("nodes", length(nodes), N)
    N == 0 && return state
    _ka_stepper_update!(dev, _flat_to_state_nodes_ka!, N, state.x, state.y, nodes)
    return state
end

function _scatter_state_shifted!(state::DeviceContourState{T},
                                 base::AbstractVector{SVector{2,T}},
                                 delta::AbstractVector{SVector{2,T}},
                                 scale::T,
                                 dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    _check_flat_buffer_length("base", length(base), N)
    _check_flat_buffer_length("delta", length(delta), N)
    N == 0 && return state
    _ka_stepper_update!(dev, _state_scatter_shifted_ka!, N,
                        state.x, state.y, base, delta, scale)
    return state
end

function _wrap_state_nodes!(state::DeviceContourState{T},
                            domain::PeriodicDomain{T},
                            dev::AbstractDevice) where {T}
    ncontours = length(state.lengths)
    ncontours == 0 && return state
    shiftx = device_zeros(dev, T, ncontours)
    shifty = device_zeros(dev, T, ncontours)
    _ka_stepper_update!(dev, _periodic_state_shifts_ka!, ncontours,
                        shiftx, shifty, state.x, state.y, state.wrapx, state.wrapy,
                        state.offsets, state.lengths, domain.Lx, domain.Ly, ncontours)
    N = _device_state_nnodes(state)
    N == 0 && return state
    _ka_stepper_update!(dev, _apply_state_shifts_ka!, N,
                        state.x, state.y, shiftx, shifty, state.contour_of_node, N)
    return state
end

_wrap_state_nodes!(state::DeviceContourState, ::UnboundedDomain, ::AbstractDevice) = state

@inline function _rk4_state_stage!(k, state::DeviceContourState{T}, kernel, domain,
                                   nodes_orig, increment, scale::T,
                                   dev::AbstractDevice) where {T}
    _scatter_state_shifted!(state, nodes_orig, increment, scale, dev)
    _ka_velocity_from_state!(k, state, kernel, domain, dev)
    return k
end

@inline function _finish_rk4_state_step!(state::DeviceContourState{T}, nodes_orig,
                                         k1, k2, k3, k4, dt::T,
                                         dev::AbstractDevice) where {T}
    N = _device_state_nnodes(state)
    N == 0 && return state
    _ka_stepper_update!(dev, _rk4_state_update_ka!, N,
                        state.x, state.y, nodes_orig, k1, k2, k3, k4, dt)
    return state
end

function _rk4_state_step!(state::DeviceContourState{T}, kernel, domain,
                          stepper::RK4Stepper{T},
                          dev::AbstractDevice) where {T}
    dt = stepper.dt
    N = _device_state_nnodes(state)
    length(stepper.k1) >= N || throw(DimensionMismatch("Stepper buffer size ($(length(stepper.k1))) < total nodes ($N). Call resize_buffers! first."))
    nodes_orig = stepper.nodes_buf

    _collect_state_nodes!(nodes_orig, state, dev)
    _ka_velocity_from_state!(stepper.k1, state, kernel, domain, dev)
    _rk4_state_stage!(stepper.k2, state, kernel, domain, nodes_orig, stepper.k1, dt / 2, dev)
    _rk4_state_stage!(stepper.k3, state, kernel, domain, nodes_orig, stepper.k2, dt / 2, dev)
    _rk4_state_stage!(stepper.k4, state, kernel, domain, nodes_orig, stepper.k3, dt, dev)

    return _finish_rk4_state_step!(state, nodes_orig, stepper.k1, stepper.k2,
                                   stepper.k3, stepper.k4, dt, dev)
end

function _leapfrog_state_step!(state::DeviceContourState{T}, kernel, domain,
                               stepper::LeapfrogStepper{T},
                               dev::AbstractDevice) where {T}
    dt = stepper.dt
    N = _device_state_nnodes(state)
    nodes_current = stepper.nodes_buf
    length(nodes_current) >= N || throw(DimensionMismatch("Stepper buffer size ($(length(nodes_current))) < total nodes ($N). Call resize_buffers! first."))

    _collect_state_nodes!(nodes_current, state, dev)
    _ka_velocity_from_state!(stepper.vel_buf, state, kernel, domain, dev)

    if !stepper.initialized
        _scatter_state_shifted!(state, nodes_current, stepper.vel_buf, dt / 2, dev)
        _ka_velocity_from_state!(stepper.vel_mid, state, kernel, domain, dev)
        _ka_stepper_update!(dev, _leapfrog_bootstrap_ka!, N,
                            stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
        _scatter_state_nodes!(state, nodes_current, dev)
        stepper.initialized = true
    else
        _ka_stepper_update!(dev, _leapfrog_step_ka!, N,
                            stepper.nodes_prev, nodes_current, stepper.vel_buf,
                            dt, stepper.ra_coeff)
        _scatter_state_nodes!(state, nodes_current, dev)
    end

    return state
end

# ---------------------------------------------------------------------------
# Multi-layer device stepping
# ---------------------------------------------------------------------------

# Multi-layer twin of _rk4_state_stage! — keep the two in lockstep.
@inline function _ml_rk4_stage!(k, states::NTuple{N, <:DeviceContourState}, kernel, domain,
                                nodes_orig, increment, scale::T,
                                ranges::NTuple{N, UnitRange{Int}},
                                dev::AbstractDevice) where {N, T}
    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        _scatter_state_shifted!(states[ℓ], view(nodes_orig, r), view(increment, r), scale, dev)
    end
    _ka_multilayer_velocity_from_states!(k, states, kernel, domain, dev)
    return k
end

# Multi-layer twin of _rk4_state_step! — keep the two in lockstep.
function _rk4_multilayer_state_step!(states::NTuple{N, <:DeviceContourState}, kernel, domain,
                                     stepper::RK4Stepper{T},
                                     dev::AbstractDevice) where {N, T}
    dt = stepper.dt
    ranges = _layer_state_ranges(states)
    Ntot = sum(length, ranges)
    length(stepper.k1) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(stepper.k1))) < total nodes ($Ntot). Call resize_buffers! first."))
    Ntot == 0 && return states
    nodes_orig = stepper.nodes_buf

    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        _collect_state_nodes!(view(nodes_orig, r), states[ℓ], dev)
    end
    _ka_multilayer_velocity_from_states!(stepper.k1, states, kernel, domain, dev)
    _ml_rk4_stage!(stepper.k2, states, kernel, domain, nodes_orig, stepper.k1, dt / 2, ranges, dev)
    _ml_rk4_stage!(stepper.k3, states, kernel, domain, nodes_orig, stepper.k2, dt / 2, ranges, dev)
    _ml_rk4_stage!(stepper.k4, states, kernel, domain, nodes_orig, stepper.k3, dt, ranges, dev)

    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        _finish_rk4_state_step!(states[ℓ], view(nodes_orig, r), view(stepper.k1, r),
                                view(stepper.k2, r), view(stepper.k3, r),
                                view(stepper.k4, r), dt, dev)
    end
    return states
end

# Multi-layer twin of _leapfrog_state_step! — keep the two in lockstep.
function _leapfrog_multilayer_state_step!(states::NTuple{N, <:DeviceContourState}, kernel, domain,
                                          stepper::LeapfrogStepper{T},
                                          dev::AbstractDevice) where {N, T}
    dt = stepper.dt
    ranges = _layer_state_ranges(states)
    Ntot = sum(length, ranges)
    nodes_current = stepper.nodes_buf
    length(nodes_current) >= Ntot || throw(DimensionMismatch("Stepper buffer size ($(length(nodes_current))) < total nodes ($Ntot). Call resize_buffers! first."))
    Ntot == 0 && return states

    for ℓ in 1:N
        r = ranges[ℓ]
        isempty(r) && continue
        _collect_state_nodes!(view(nodes_current, r), states[ℓ], dev)
    end
    _ka_multilayer_velocity_from_states!(stepper.vel_buf, states, kernel, domain, dev)

    if !stepper.initialized
        for ℓ in 1:N
            r = ranges[ℓ]
            isempty(r) && continue
            _scatter_state_shifted!(states[ℓ], view(nodes_current, r),
                                    view(stepper.vel_buf, r), dt / 2, dev)
        end
        _ka_multilayer_velocity_from_states!(stepper.vel_mid, states, kernel, domain, dev)
        _ka_stepper_update!(dev, _leapfrog_bootstrap_ka!, Ntot,
                            stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
        for ℓ in 1:N
            r = ranges[ℓ]
            isempty(r) && continue
            _scatter_state_nodes!(states[ℓ], view(nodes_current, r), dev)
        end
        stepper.initialized = true
    else
        _ka_stepper_update!(dev, _leapfrog_step_ka!, Ntot,
                            stepper.nodes_prev, nodes_current, stepper.vel_buf,
                            dt, stepper.ra_coeff)
        for ℓ in 1:N
            r = ranges[ℓ]
            isempty(r) && continue
            _scatter_state_nodes!(states[ℓ], view(nodes_current, r), dev)
        end
    end
    return states
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

@inline function _rk4_stage!(k, prob::ContourProblem, nodes_orig, increment, scale, ranges)
    _scatter_shifted!(prob, nodes_orig, increment, scale, ranges)
    velocity!(k, prob)
    return k
end

@inline function _finish_rk4_step!(prob::ContourProblem, nodes_orig, k1, k2, k3, k4, dt, ranges)
    _cpu_rk4_combine!(nodes_orig, k1, k2, k3, k4, dt)
    _scatter_nodes!(prob, nodes_orig, ranges)
    return prob
end

@inline function _bootstrap_leapfrog!(prob::ContourProblem, stepper::LeapfrogStepper{T},
                                      nodes_current, vel, dt, ranges) where {T}
    _scatter_shifted!(prob, nodes_current, vel, dt / 2, ranges)
    velocity!(stepper.vel_mid, prob)
    _cpu_leapfrog_bootstrap!(stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
    _scatter_nodes!(prob, nodes_current, ranges)
    stepper.initialized = true
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
        offset += _layer_node_count(prob.layers[i])
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
        offset += _layer_node_count(prob.layers[i])
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
    ntuple(i -> zeros(SVector{2,T}, _layer_node_count(prob.layers[i])), Val(N))
end

function _ensure_vel_bufs!(vel_bufs::Vector{Vector{SVector{2, T}}},
                           prob::MultiLayerContourProblem{N}) where {N, T}
    z = zero(SVector{2, T})
    while length(vel_bufs) < N
        push!(vel_bufs, SVector{2, T}[])
    end
    for i in 1:N
        n_layer = _layer_node_count(prob.layers[i])
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
        offset += _layer_node_count(prob.layers[i])
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
    _cpu_rk4_combine!(nodes_orig, k1, k2, k3, k4, dt)
    _scatter_nodes!(prob, nodes_orig, all_ranges)
    return prob
end

@inline function _bootstrap_leapfrog!(prob::MultiLayerContourProblem, stepper::LeapfrogStepper{T},
                                      nodes_current, flat_vel, vel_tuple, dt, all_ranges) where {T}
    _scatter_shifted!(prob, nodes_current, flat_vel, dt / 2, all_ranges)
    velocity!(vel_tuple, prob)
    _collect_velocities!(stepper.vel_mid, vel_tuple)
    _cpu_leapfrog_bootstrap!(stepper.nodes_prev, nodes_current, stepper.vel_mid, dt)
    _scatter_nodes!(prob, nodes_current, all_ranges)
    stepper.initialized = true
    return prob
end
