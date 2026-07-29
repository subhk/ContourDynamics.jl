# Shared velocity helpers and dispatch policies.
#
# Sign convention: positive PV induces counterclockwise circulation.
# For a vortex patch with uniform vorticity q bounded by contour C,
# the velocity is obtained by converting the area integral of the
# Green's function to a contour integral via Green's theorem:
#
#   u(x) = -(q/(4π)) ∮_C log|x-x'|² dx'
#   v(x) = -(q/(4π)) ∮_C log|x-x'|² dy'
#
# i.e.  (u, v) = -(q/(4π)) ∮_C log|x-x'|²  ds'
#
# Straight Euler segments are integrated analytically; curved Euler segments use
# cubic interpolation with fixed Gauss-Legendre quadrature.

"""Minimum target count before CPU velocity loops use `Threads.@threads`."""
const _VELOCITY_THREADING_THRESHOLD = 128

@inline _should_thread_velocity(n::Integer) =
    Threads.nthreads() > 1 && n >= _VELOCITY_THREADING_THRESHOLD

# 5-point Gauss-Legendre nodes and weights on [-1,1].
# Precomputed for Float64 and Float32 to avoid repeated sqrt/division in
# the innermost velocity loop. Generic fallback for BigFloat etc.
let
    _n2_64 = sqrt((5.0 - 2.0 * sqrt(10.0/7.0)) / 9.0)
    _n3_64 = sqrt((5.0 + 2.0 * sqrt(10.0/7.0)) / 9.0)
    _w1_64 = 128.0 / 225.0
    _w2_64 = (322.0 + 13.0 * sqrt(70.0)) / 900.0
    _w3_64 = (322.0 - 13.0 * sqrt(70.0)) / 900.0
    global const _GL5_NODES_F64   = SVector{5,Float64}(-_n3_64, -_n2_64, 0.0, _n2_64, _n3_64)
    global const _GL5_WEIGHTS_F64 = SVector{5,Float64}(_w3_64, _w2_64, _w1_64, _w2_64, _w3_64)
    global const _GL5_NODES_F32   = SVector{5,Float32}(Float32.(_GL5_NODES_F64)...)
    global const _GL5_WEIGHTS_F32 = SVector{5,Float32}(Float32.(_GL5_WEIGHTS_F64)...)

    _n1_64 = sqrt(3.0/5.0)
    global const _GL3_NODES_F64   = SVector{3,Float64}(-_n1_64, 0.0, _n1_64)
    global const _GL3_WEIGHTS_F64 = SVector{3,Float64}(5.0/9.0, 8.0/9.0, 5.0/9.0)
    global const _GL3_NODES_F32   = SVector{3,Float32}(Float32.(_GL3_NODES_F64)...)
    global const _GL3_WEIGHTS_F32 = SVector{3,Float32}(Float32.(_GL3_WEIGHTS_F64)...)
end

@inline _gl5_nodes_weights(::Type{Float64}) = (_GL5_NODES_F64, _GL5_WEIGHTS_F64)

@inline _gl5_nodes_weights(::Type{Float32}) = (_GL5_NODES_F32, _GL5_WEIGHTS_F32)

@inline function _gl5_nodes_weights(::Type{T}) where {T<:AbstractFloat}
    # Generic construction keeps high-precision scalar types working while the
    # Float32/Float64 methods above avoid recomputing constants in hot loops.
    n2 = sqrt((T(5) - T(2) * sqrt(T(10)/T(7))) / T(9))
    n3 = sqrt((T(5) + T(2) * sqrt(T(10)/T(7))) / T(9))

    w1 = T(128) / T(225)
    w2 = (T(322) + T(13) * sqrt(T(70))) / T(900)
    w3 = (T(322) - T(13) * sqrt(T(70))) / T(900)
    
    nodes = SVector{5,T}(-n3, -n2, zero(T), n2, n3)
    weights = SVector{5,T}(w3, w2, w1, w2, w3)
    
    return (nodes, weights)
end

@inline _gl3_nodes_weights(::Type{Float64}) = (_GL3_NODES_F64, _GL3_WEIGHTS_F64)

@inline _gl3_nodes_weights(::Type{Float32}) = (_GL3_NODES_F32, _GL3_WEIGHTS_F32)

@inline function _gl3_nodes_weights(::Type{T}) where {T<:AbstractFloat}
    # 3-point quadrature is used by energy diagnostics where the integrand is
    # smoother after singular subtraction.
    n1 = sqrt(T(3)/T(5))
    nodes = SVector{3,T}(-n1, zero(T), n1)
    weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    return (nodes, weights)
end

function _prepare_curvature_buffers!(buffers::Vector{Vector{T}},
                                     contours::Vector{PVContour{T}}) where {T}
    while length(buffers) < length(contours)
        push!(buffers, T[])
    end
    resize!(buffers, length(contours))
    @inbounds for i in eachindex(contours)
        resize!(buffers[i], nnodes(contours[i]))
        _signed_node_curvatures!(buffers[i], contours[i])
    end
    return buffers
end

function _prepare_layer_curvature_buffers!(buffers::Vector{Vector{Vector{T}}},
                                           layers::NTuple{N, Vector{PVContour{T}}}) where {N, T}
    while length(buffers) < N
        push!(buffers, Vector{T}[])
    end
    resize!(buffers, N)
    @inbounds for i in 1:N
        _prepare_curvature_buffers!(buffers[i], layers[i])
    end
    return buffers
end

function _prepare_contour_offsets!(offsets::Vector{Int},
                                   contours::Vector{PVContour{T}}) where {T}
    resize!(offsets, length(contours) + 1)
    offsets[1] = 0
    @inbounds for ci in eachindex(contours)
        offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
    end
    return offsets
end

function _max_layer_node_count(prob::MultiLayerContourProblem{N}) where {N}
    max_nodes = 0
    @inbounds for i in 1:N
        max_nodes = max(max_nodes, sum(nnodes(c) for c in prob.layers[i]; init=0))
    end
    return max_nodes
end

function _prepare_modal_matrices!(scratch::_VelocityScratch{T},
                                  kernel::MultiLayerQGKernel{N,M,T}) where {N, M, T}
    # The kernel's eigenvector matrices are fixed at construction, so the dense
    # scratch copies only need to be materialized once — when the buffer is first
    # sized for this problem. The previous version re-copied both N×N matrices on
    # every call, allocating ~128 B per multilayer velocity! (512 B per RK4 step)
    # for no benefit, since `kernel.eigenvectors` never changes.
    if size(scratch.modal_matrix) != (N, N)
        scratch.modal_matrix = Matrix{T}(kernel.eigenvectors)
        scratch.modal_matrix_inv = Matrix{T}(kernel.eigenvectors_inv)
    end
    return scratch.modal_matrix, scratch.modal_matrix_inv
end

@inline function _segment_velocity_with_geometry(kernel::AbstractKernel,
                                                 domain::AbstractDomain,
                                                 x::SVector{2,T},
                                                 a::SVector{2,T},
                                                 b::SVector{2,T},
                                                 κa::T, κb::T,
                                                 ewald) where {T}
    # Dispatch between straight analytic integration and curved cubic
    # integration using a dimensionless curvature measure. This keeps nearly
    # straight segments on the cheaper and more stable analytic path.
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    ds_len < eps(T) && return zero(SVector{2,T})
    max(abs(κa), abs(κb)) * ds_len <= sqrt(eps(T)) &&
        return segment_velocity(kernel, domain, x, a, b, ewald)
    return curved_segment_velocity(kernel, domain, x, a, b, κa, κb, ewald)
end

# Sum the velocity at one target point `xi` from every source segment of every
# contour. Shared by the threaded and serial branches of _direct_velocity! so the
# inner loop lives in one place. @inline + concrete kernel/domain keep it
# allocation-free and fully specialized.
@inline function _accumulate_node_velocity(kernel, domain,
                                           contours::Vector{PVContour{T}},
                                           source_curvatures::Vector{Vector{T}},
                                           ewald, xi::SVector{2,T}) where {T}
    v = zero(SVector{2,T})
    for (source_ci, c) in pairs(contours)
        nc = nnodes(c)
        nc < 2 && continue
        pv = c.pv
        κ = source_curvatures[source_ci]
        @inbounds for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            v = v + pv * _segment_velocity_with_geometry(
                kernel, domain, xi, a, b, κ[j], κ[mod1(j + 1, nc)], ewald)
        end
    end
    return v
end

"""
    _direct_velocity!(vel, prob::ContourProblem)

Direct O(N²) velocity computation at every contour node of `prob`, storing
results in `vel`. This is the brute-force reference implementation.
"""
function _direct_velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem) where {T}
    kernel = prob.kernel
    domain = prob.domain
    contours = prob.contours
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))

    # Pre-fetch Ewald cache once (returns `nothing` for unbounded domains)
    ewald = _prefetch_ewald(domain, kernel)
    scratch = prob.velocity_scratch
    source_curvatures = _prepare_curvature_buffers!(scratch.contour_curvatures, contours)

    # Thread over target nodes only once the workload is large enough to pay for it.
    if _should_thread_velocity(N)
        n_contours = length(contours)
        offsets = _prepare_contour_offsets!(scratch.offsets, contours)
        Threads.@threads for i in 1:N
            ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
            ci = clamp(ci, 1, n_contours)
            local_i = i - offsets[ci]
            (1 <= local_i <= nnodes(contours[ci])) || throw(BoundsError(contours[ci].nodes, local_i))
            xi = contours[ci].nodes[local_i]
            vel[i] = _accumulate_node_velocity(kernel, domain, contours, source_curvatures, ewald, xi)
        end
    else
        idx = 1
        for target_contour in contours
            @inbounds for local_i in 1:nnodes(target_contour)
                xi = target_contour.nodes[local_i]
                vel[idx] = _accumulate_node_velocity(kernel, domain, contours, source_curvatures, ewald, xi)
                idx += 1
            end
        end
    end

    return vel
end

@inline function _validate_velocity_buffer!(vel::Vector{SVector{2,T}},
                                            prob::ContourProblem) where {T}
    # Public velocity! accepts oversized reusable buffers; only the prefix
    # containing current nodes is written.
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    return N
end

@inline function _validate_velocity_buffer!(vel::NTuple{N, Vector{SVector{2,T}}},
                                            prob::MultiLayerContourProblem{N}) where {N, T}
    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        length(vel[i]) >= n_layer || throw(DimensionMismatch("vel[$i] length ($(length(vel[i]))) must be >= layer $i nodes ($n_layer)"))
    end
    return total_nodes(prob)
end

@inline function _validate_multilayer_state_velocity_buffer!(
        vel::NTuple{N, Vector{SVector{2,T}}},
        states::NTuple{N, <:DeviceContourState}) where {N, T}
    for layer in 1:N
        required = _device_state_nnodes(states[layer])
        length(vel[layer]) >= required || throw(DimensionMismatch(
            "vel[$layer] length ($(length(vel[layer]))) must be >= layer $layer nodes ($required)"))
    end
    return vel
end

"""
    _small_velocity!(vel, prob::ContourProblem)

The CPU/GPU seam for single-layer velocity. This is the **only** place the device
is branched on: dispatch picks the method by the `CPU`/`GPU` type parameter of
`prob`, so every layer above (`velocity!`, [`_velocity_policy!`](@ref)) stays
device-agnostic.

- **CPU** — a single method for all kernels (Euler/QG/SQG, unbounded or periodic):
  the direct evaluator [`_direct_velocity!`](@ref), which reuses scratch buffers
  and is allocation-free.
- **GPU** — the KernelAbstractions path in `src/accel/ka/`.
"""
@inline _small_velocity!(vel::Vector{SVector{2,T}},
                         prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) where {T} =
    _direct_velocity!(vel, prob)

@inline _small_velocity!(vel::Vector{SVector{2,T}},
                         prob::ContourProblem{K, D, T, GPU}) where {K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel}, D<:AbstractDomain, T} =
    _ka_velocity!(vel, prob, prob.dev)

"""
    _velocity_policy!(vel, prob::ContourProblem)

Validate-then-compute skeleton shared by the CPU and GPU `velocity!` methods.
Both public methods are one line because the common work — buffer validation
([`_validate_velocity_buffer!`](@ref)) followed by the device dispatch
([`_small_velocity!`](@ref)) — lives here once.
"""
function _velocity_policy!(vel::Vector{SVector{2,T}},
                           prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T}) where {T}
    _validate_velocity_buffer!(vel, prob)
    _small_velocity!(vel, prob)
    return vel
end

@inline _small_multilayer_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                    prob::MultiLayerContourProblem{N, <:Any, <:Any, T, CPU}) where {N, T} =
    _direct_velocity!(vel, prob)

@inline function _small_multilayer_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                             prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T}
    return _ka_multilayer_velocity_to_host!(vel, prob.device_state, prob.kernel,
                                            prob.domain, prob.dev)
end

function _multilayer_velocity_policy!(vel::NTuple{N, Vector{SVector{2,T}}},
                                      prob::MultiLayerContourProblem{N, <:Any, <:Any, T}) where {N, T}
    _validate_velocity_buffer!(vel, prob)
    _small_multilayer_velocity!(vel, prob)
    return vel
end

"""
    velocity!(vel, prob::ContourProblem)

Compute velocity at every contour node of `prob`, storing results in `vel`.

The dispatcher uses the direct reference evaluator on CPU and the
KernelAbstractions direct evaluator for supported GPU-tagged problems.
"""
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) where {T}
    return _velocity_policy!(vel, prob)
end

"""
    velocity(prob::ContourProblem, x::SVector{2,T})

Compute velocity at a single point `x` from all contours in `prob`.
"""
function velocity(prob::ContourProblem, x::SVector{2,T}) where {T}
    # Single-point evaluation mirrors the node-wise direct evaluator, but avoids
    # allocating a temporary one-node problem.
    v = zero(SVector{2,T})
    ewald = _prefetch_ewald(prob.domain, prob.kernel)
    for c in prob.contours
        nc = nnodes(c)
        nc < 2 && continue
        for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            v = v + c.pv * _segment_velocity_with_geometry(
                prob.kernel, prob.domain, x, a, b,
                _signed_node_curvature(c, j),
                _signed_node_curvature(c, mod1(j + 1, nc)),
                ewald)
        end
    end
    return v
end

function _velocity_at_state(state::DeviceContourState{T}, kernel::AbstractKernel,
                            domain::AbstractDomain, x::SVector{2,T}) where {T}
    host_prob = ContourProblem(kernel, domain, materialize_contours(state); dev=CPU())
    return velocity(host_prob, x)
end

function velocity(prob::ContourProblem{K,D,T,GPU},
                  x::SVector{2,T}) where {K,D,T}
    # GPU timestepping and surgery mutate only `device_state`; `prob.contours`
    # is the construction-time host shadow. Point queries are an explicit host
    # inspection boundary, so materialize the authoritative state before using
    # the scalar direct evaluator.
    return _velocity_at_state(prob.device_state, prob.kernel, prob.domain, x)
end

"""
    velocity(prob::MultiLayerContourProblem, x)

Compute the velocity induced at point `x` in each layer of a multi-layer
problem. Returns an `NTuple` with one velocity vector per target layer.
"""
function velocity(prob::MultiLayerContourProblem{N, <:Any, <:Any, T},
                  x::SVector{2,T}) where {N, T}
    # Single-point multilayer velocity is evaluated in vertical modes, then
    # projected back to physical layers. Each mode behaves like an Euler or QG
    # single-layer problem depending on its eigenvalue.
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    scratch = prob.velocity_scratch
    P, P_inv = _prepare_modal_matrices!(scratch, kernel)
    source_curvatures = _prepare_layer_curvature_buffers!(
        scratch.layer_curvatures, prob.layers)

    vel = resize!(scratch.mode_vel, N)
    fill!(vel, zero(SVector{2,T}))

    for mode in 1:N
        lam = evals[mode]
        # Resolve a concrete mode kernel via a function barrier so the inner
        # segment_velocity is fully specialised (no Union-typed dynamic
        # dispatch per segment). Each mode fetches its own cache: QG modes need
        # their Ld-specific correction coefficients, the Euler mode does not.
        v_mode = if abs(lam) < eps(T) * 100
            _accumulate_mode_node_velocity(EulerKernel(), domain, prob.layers,
                source_curvatures, _prefetch_ewald(domain, EulerKernel()), P_inv, mode, x)
        else
            mode_kernel = QGKernel(one(T) / sqrt(abs(lam)))
            _accumulate_mode_node_velocity(mode_kernel, domain, prob.layers,
                source_curvatures, _prefetch_ewald(domain, mode_kernel), P_inv, mode, x)
        end

        # Project the completed modal velocity back onto each physical target
        # layer with the forward eigenvector matrix.
        for target_layer in 1:N
            projection_weight = P[target_layer, mode]
            abs(projection_weight) < eps(T) && continue
            vel[target_layer] = vel[target_layer] + projection_weight * v_mode
        end
    end

    return ntuple(i -> vel[i], Val(N))
end

function _velocity_at_state(states::NTuple{N,<:DeviceContourState{T}},
                            kernel::MultiLayerQGKernel{N}, domain::AbstractDomain,
                            x::SVector{2,T}) where {N,T}
    host_layers = ntuple(i -> materialize_contours(states[i]), Val(N))
    host_prob = MultiLayerContourProblem(kernel, domain, host_layers; dev=CPU())
    return velocity(host_prob, x)
end

function velocity(prob::MultiLayerContourProblem{N,K,D,T,GPU},
                  x::SVector{2,T}) where {N,K,D,T}
    return _velocity_at_state(prob.device_state, prob.kernel, prob.domain, x)
end

# Sum the modal velocity at one target point `x` from every source layer/segment,
# weighted by the inverse modal projection. Shared by the threaded and serial
# branches of _multilayer_mode_velocity!. Concrete mode_kernel type K keeps
# segment_velocity fully specialized inside the loop.
@inline function _accumulate_mode_node_velocity(mode_kernel::K, domain,
                                                layers::NTuple{N, Vector{PVContour{T}}},
                                                source_curvatures, ewald,
                                                P_inv::Matrix{T}, mode::Int,
                                                x::SVector{2,T}) where {N, T, K}
    v_mode = zero(SVector{2,T})
    for source_layer in 1:N
        source_weight = P_inv[mode, source_layer]
        abs(source_weight) < eps(T) && continue
        for (sci, sc) in pairs(layers[source_layer])
            nsc = nnodes(sc)
            nsc < 2 && continue
            κ = source_curvatures[source_layer][sci]
            for sj in 1:nsc
                a = sc.nodes[sj]
                b = next_node(sc, sj)
                v_mode = v_mode + source_weight * sc.pv *
                    _segment_velocity_with_geometry(
                        mode_kernel, domain, x, a, b,
                        κ[sj], κ[mod1(sj + 1, nsc)], ewald)
            end
        end
    end
    return v_mode
end

# Function barrier: the concrete kernel type is resolved here so that
# segment_velocity is fully specialised inside the @threads loop.
function _multilayer_mode_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                    prob::MultiLayerContourProblem{N},
                                    mode::Int, mode_kernel::K,
                                    target_nodes::Vector{SVector{2,T}},
                                    mode_vel::Vector{SVector{2,T}},
                                    source_curvatures,
                                    ewald,
                                    P::Matrix{T},
                                    P_inv::Matrix{T}) where {N, T, K}
    domain = prob.domain

    for target_layer in 1:N
        target_contours = prob.layers[target_layer]
        projection_weight = P[target_layer, mode]
        abs(projection_weight) < eps(T) && continue

        n_target = sum(nnodes(tc) for tc in target_contours; init=0)
        n_target == 0 && continue

        # Flatten target nodes for this layer once so threaded and serial loops
        # share the same indexing and write into mode_vel.
        idx = 0
        for tc in target_contours
            for ti in 1:nnodes(tc)
                idx += 1
                target_nodes[idx] = tc.nodes[ti]
            end
        end

        if _should_thread_velocity(n_target)
            @inbounds Threads.@threads for ti in 1:n_target
                mode_vel[ti] = _accumulate_mode_node_velocity(
                    mode_kernel, domain, prob.layers, source_curvatures, ewald,
                    P_inv, mode, target_nodes[ti])
            end
        else
            @inbounds for ti in 1:n_target
                mode_vel[ti] = _accumulate_mode_node_velocity(
                    mode_kernel, domain, prob.layers, source_curvatures, ewald,
                    P_inv, mode, target_nodes[ti])
            end
        end

        for ti in 1:n_target
            # mode_vel is modal space; vel is physical-layer space.
            vel[target_layer][ti] = vel[target_layer][ti] + projection_weight * mode_vel[ti]
        end
    end
end

"""
    _direct_velocity!(vel, prob::MultiLayerContourProblem)

Direct O(N^2) velocity computation at every contour node across all layers of
`prob`, storing results in `vel`. Uses modal decomposition with direct summation.
"""
function _direct_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                           prob::MultiLayerContourProblem{N}) where {N, T}
    kernel = prob.kernel
    domain = prob.domain

    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        length(vel[i]) >= n_layer || throw(DimensionMismatch("vel[$i] length ($(length(vel[i]))) must be >= layer $i nodes ($n_layer)"))
        fill!(vel[i], zero(SVector{2,T}))
    end

    evals = kernel.eigenvalues

    scratch = prob.velocity_scratch
    P, P_inv = _prepare_modal_matrices!(scratch, kernel)
    max_nodes = _max_layer_node_count(prob)
    target_nodes = resize!(scratch.target_nodes, max_nodes)
    mode_vel = resize!(scratch.mode_vel, max_nodes)

    source_curvatures = _prepare_layer_curvature_buffers!(
        scratch.layer_curvatures, prob.layers)
    for mode in 1:N
        lam = evals[mode]

        # Zero eigenvalues represent barotropic Euler modes; nonzero
        # eigenvalues become QG modes with deformation radius 1/sqrt(abs(lambda)).
        # Each mode fetches its own cache so the QG modes read their Ld-specific
        # correction coefficients (the Euler cache has none).
        if abs(lam) < eps(T) * 100
            _multilayer_mode_velocity!(vel, prob, mode, EulerKernel(),
                                       target_nodes, mode_vel, source_curvatures,
                                       _prefetch_ewald(domain, EulerKernel()), P, P_inv)
        else
            mode_kernel = QGKernel(one(T) / sqrt(abs(lam)))
            _multilayer_mode_velocity!(vel, prob, mode, mode_kernel,
                                       target_nodes, mode_vel, source_curvatures,
                                       _prefetch_ewald(domain, mode_kernel), P, P_inv)
        end
    end

    return vel
end


"""
    velocity!(vel, prob::MultiLayerContourProblem)

Compute velocity at all nodes across all layers using modal decomposition.

The dispatcher uses direct modal decomposition on CPU and the direct
KernelAbstractions modal path for supported GPU-tagged problems.
"""
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N, <:Any, <:Any, T, CPU}) where {N, T}
    return _multilayer_velocity_policy!(vel, prob)
end

# GPU dispatch — velocity computed in SoA layout via KernelAbstractions from the
# device-resident contour state, then repacked into the CPU vel buffer.
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel{T}}, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    return _velocity_policy!(vel, prob)
end

function velocity!(vel::AbstractVector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel,BetaPlaneQGKernel{T}}, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    return _ka_velocity!(vel, prob, prob.dev)
end

# Fallback for unsupported GPU kernel/domain combinations
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    throw(ArgumentError(
        "GPU velocity is implemented for single-layer EulerKernel, QGKernel, and SQGKernel " *
        "on UnboundedDomain or PeriodicDomain, and BetaPlaneQGKernel on PeriodicDomain. " *
        "Got $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Use dev=CPU() for other kernel/domain combinations."))
end

function velocity!(vel::AbstractVector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    throw(ArgumentError(
        "GPU velocity is implemented for single-layer EulerKernel, QGKernel, and SQGKernel " *
        "on UnboundedDomain or PeriodicDomain, and BetaPlaneQGKernel on PeriodicDomain. " *
        "Got $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Use dev=CPU() for other kernel/domain combinations."))
end

# GPU fallback for multi-layer problems
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T}
    _validate_multilayer_state_velocity_buffer!(vel, prob.device_state)
    return _ka_multilayer_velocity_to_host!(vel, prob.device_state, prob.kernel,
                                            prob.domain, prob.dev)
end
