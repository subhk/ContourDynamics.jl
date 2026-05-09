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
    source_curvatures = [_signed_node_curvatures(c) for c in contours]

    # Thread over target nodes only once the workload is large enough to pay for it.
    if _should_thread_velocity(N)
        n_contours = length(contours)
        offsets = Vector{Int}(undef, n_contours + 1)
        offsets[1] = 0
        for ci in 1:n_contours
            offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
        end
        Threads.@threads for i in 1:N
            ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
            ci = clamp(ci, 1, n_contours)
            local_i = i - offsets[ci]
            (1 <= local_i <= nnodes(contours[ci])) || throw(BoundsError(contours[ci].nodes, local_i))
            xi = contours[ci].nodes[local_i]

            v = zero(SVector{2,T})
            for (source_ci, c) in pairs(contours)
                local nc = nnodes(c)
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
            vel[i] = v
        end
    else
        idx = 1
        for target_contour in contours
            @inbounds for local_i in 1:nnodes(target_contour)
                xi = target_contour.nodes[local_i]

                v = zero(SVector{2,T})
                for (source_ci, c) in pairs(contours)
                    local nc = nnodes(c)
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
                vel[idx] = v
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

@inline _small_velocity!(vel::Vector{SVector{2,T}},
                         prob::ContourProblem{EulerKernel, UnboundedDomain, T, CPU}) where {T} =
    _ka_velocity!(vel, prob, prob.dev)

@inline _small_velocity!(vel::Vector{SVector{2,T}},
                         prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) where {T} =
    _direct_velocity!(vel, prob)

@inline _small_velocity!(vel::Vector{SVector{2,T}},
                         prob::ContourProblem{K, D, T, GPU}) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T} =
    _ka_velocity!(vel, prob, prob.dev)

function _velocity_policy!(vel::Vector{SVector{2,T}},
                           prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T}) where {T}
    # Central dispatch point for single-layer velocity. Specializing the helper
    # methods by device/kernel keeps the public method surface small.
    _validate_velocity_buffer!(vel, prob)
    _small_velocity!(vel, prob)
    return vel
end

@inline _small_multilayer_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                    prob::MultiLayerContourProblem{N, <:Any, <:Any, T, CPU}) where {N, T} =
    _direct_velocity!(vel, prob)

@inline _small_multilayer_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                    prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T} =
    _ka_multilayer_velocity!(vel, prob, prob.dev)

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
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv
    ewald = _prefetch_ewald(domain, EulerKernel())
    source_curvatures = [
        [_signed_node_curvatures(c) for c in prob.layers[layer]]
        for layer in 1:N
    ]

    vel = MVector{N, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(N)))

    for mode in 1:N
        lam = evals[mode]
        mode_kernel = abs(lam) < eps(T) * 100 ? EulerKernel() :
                      QGKernel(one(T) / sqrt(abs(lam)))

        # Accumulate the modal velocity at x from every source layer, weighted
        # by the inverse eigenvector matrix.
        v_mode = zero(SVector{2,T})
        for source_layer in 1:N
            source_weight = P_inv[mode, source_layer]
            abs(source_weight) < eps(T) && continue
            for (sci, sc) in pairs(prob.layers[source_layer])
                nsc = nnodes(sc)
                nsc < 2 && continue
                κ = source_curvatures[source_layer][sci]
                for sj in 1:nsc
                    a = sc.nodes[sj]
                    b = next_node(sc, sj)
                    v_mode = v_mode + source_weight * sc.pv *
                        _segment_velocity_with_geometry(
                            mode_kernel, domain, x, a, b, κ[sj], κ[mod1(sj + 1, nsc)], ewald)
                end
            end
        end

        # Project the completed modal velocity back onto each physical target
        # layer with the forward eigenvector matrix.
        for target_layer in 1:N
            projection_weight = P[target_layer, mode]
            abs(projection_weight) < eps(T) && continue
            vel[target_layer] = vel[target_layer] + projection_weight * v_mode
        end
    end

    return Tuple(vel)
end

# Function barrier: the concrete kernel type is resolved here so that
# segment_velocity is fully specialised inside the @threads loop.
function _multilayer_mode_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                    prob::MultiLayerContourProblem{N},
                                    mode::Int, mode_kernel::K,
                                    target_nodes::Vector{SVector{2,T}},
                                    mode_vel::Vector{SVector{2,T}},
                                    source_curvatures,
                                    ewald) where {N, T, K}
    P = prob.kernel.eigenvectors
    P_inv = prob.kernel.eigenvectors_inv
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
                x = target_nodes[ti]
                v_mode = zero(SVector{2,T})
                for source_layer in 1:N
                    source_weight = P_inv[mode, source_layer]
                    abs(source_weight) < eps(T) && continue
                    for (sci, sc) in pairs(prob.layers[source_layer])
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
                mode_vel[ti] = v_mode
            end
        else
            @inbounds for ti in 1:n_target
                x = target_nodes[ti]
                v_mode = zero(SVector{2,T})
                for source_layer in 1:N
                    source_weight = P_inv[mode, source_layer]
                    abs(source_weight) < eps(T) && continue
                    for (sci, sc) in pairs(prob.layers[source_layer])
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
                mode_vel[ti] = v_mode
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
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    # Pre-fetch Ewald cache once (all modes use the Euler cache for periodic domains)
    ewald = _prefetch_ewald(domain, EulerKernel())

    max_nodes = maximum(sum(nnodes(c) for c in prob.layers[i]; init=0) for i in 1:N)
    target_nodes = Vector{SVector{2,T}}(undef, max_nodes)
    mode_vel = Vector{SVector{2,T}}(undef, max_nodes)

    source_curvatures = [
        [_signed_node_curvatures(c) for c in prob.layers[layer]]
        for layer in 1:N
    ]
    for mode in 1:N
        lam = evals[mode]

        # Zero eigenvalues represent barotropic Euler modes; nonzero
        # eigenvalues become QG modes with deformation radius 1/sqrt(abs(lambda)).
        if abs(lam) < eps(T) * 100
            _multilayer_mode_velocity!(vel, prob, mode, EulerKernel(),
                                       target_nodes, mode_vel, source_curvatures, ewald)
        else
            Ld_mode = one(T) / sqrt(abs(lam))
            _multilayer_mode_velocity!(vel, prob, mode, QGKernel(Ld_mode),
                                       target_nodes, mode_vel, source_curvatures, ewald)
        end
    end

    return vel
end

@inline function _multilayer_layer_ranges(prob::MultiLayerContourProblem{N}) where {N}
    # Ranges map each physical layer into the flat target vector used by the KA
    # modal evaluator.
    ranges = Vector{UnitRange{Int}}(undef, N)
    idx = 1
    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        ranges[i] = idx:(idx + n_layer - 1)
        idx += n_layer
    end
    return ranges
end

function _ka_multilayer_velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                                  prob::MultiLayerContourProblem{N, <:Any, <:Any, T},
                                  dev::AbstractDevice) where {N, T}
    # Reuse the single-layer KA path per vertical mode by building weighted
    # contours. Projection back to physical layers happens after each mode.
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    total = total_nodes(prob)
    n_contours = sum(length(prob.layers[i]) for i in 1:N)
    layer_ranges = _multilayer_layer_ranges(prob)
    mode_vel = zeros(SVector{2,T}, total)
    weighted = Vector{PVContour{T}}(undef, n_contours)

    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        length(vel[i]) >= n_layer || throw(DimensionMismatch("vel[$i] length ($(length(vel[i]))) must be >= layer $i nodes ($n_layer)"))
        fill!(vel[i], zero(SVector{2,T}))
    end

    for mode in 1:N
        lam = evals[mode]
        mode_kernel = abs(lam) < eps(T) * 100 ? EulerKernel() : QGKernel(one(T) / sqrt(abs(lam)))

        # Build a temporary single-layer modal problem by weighting each layer's
        # contour PV by the inverse modal projection.
        ci = 1
        for layer in 1:N
            weight = P_inv[mode, layer]
            for c in prob.layers[layer]
                weighted[ci] = PVContour(c.nodes, weight * c.pv, c.wrap, c.corners)
                ci += 1
            end
        end

        mode_prob = ContourProblem(mode_kernel, domain, weighted; dev=dev)
        _ka_velocity!(mode_vel, mode_prob, dev)

        # Accumulate the modal result into each physical layer using precomputed
        # flat ranges, preserving the user's per-layer velocity buffers.
        for target_layer in 1:N
            projection_weight = P[target_layer, mode]
            abs(projection_weight) < eps(T) && continue
            r = layer_ranges[target_layer]
            idx_local = 1
            @inbounds for gi in r
                vel[target_layer][idx_local] = vel[target_layer][idx_local] +
                    projection_weight * mode_vel[gi]
                idx_local += 1
            end
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

# GPU dispatch — velocity computed in SoA layout via KernelAbstractions,
# then repacked into the CPU vel buffer.
# Uses a cached workspace to avoid repeated GPU/CPU allocations across
# the 4 velocity evaluations per RK4 step.
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    return _velocity_policy!(vel, prob)
end

function velocity!(vel::AbstractVector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {T, K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:Union{UnboundedDomain, PeriodicDomain{T}}}
    return _ka_velocity!(vel, prob, prob.dev)
end

# Fallback for unsupported GPU kernel/domain combinations
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    throw(ArgumentError(
        "GPU velocity is implemented for single-layer EulerKernel, QGKernel, and SQGKernel " *
        "on UnboundedDomain or PeriodicDomain. " *
        "Got $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Use dev=CPU() for other kernel/domain combinations."))
end

function velocity!(vel::AbstractVector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    throw(ArgumentError(
        "GPU velocity is implemented for single-layer EulerKernel, QGKernel, and SQGKernel " *
        "on UnboundedDomain or PeriodicDomain. " *
        "Got $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Use dev=CPU() for other kernel/domain combinations."))
end

# GPU fallback for multi-layer problems
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T}
    return _multilayer_velocity_policy!(vel, prob)
end
