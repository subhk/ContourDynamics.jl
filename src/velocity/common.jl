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
# Each segment contribution is integrated analytically.

"""Minimum target count before CPU velocity loops use `Threads.@threads`."""
const _VELOCITY_THREADING_THRESHOLD = 128

"""Minimum leaf/check-point work size before FMM/treecode loops use threading."""
const _ACCELERATOR_THREADING_THRESHOLD = 16

@inline _should_thread_velocity(n::Integer) =
    Threads.nthreads() > 1 && n >= _VELOCITY_THREADING_THRESHOLD

@inline _should_thread_accelerator(n::Integer) =
    Threads.nthreads() > 1 && n >= _ACCELERATOR_THREADING_THRESHOLD

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
    n1 = sqrt(T(3)/T(5))
    nodes = SVector{3,T}(-n1, zero(T), n1)
    weights = SVector{3,T}(T(5)/T(9), T(8)/T(9), T(5)/T(9))
    return (nodes, weights)
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

    n_contours = length(contours)

    # Pre-fetch Ewald cache once (returns `nothing` for unbounded domains)
    ewald = _prefetch_ewald(domain, kernel)

    # Prefix-sum array for O(log n) flat-index → (contour, local) mapping.
    offsets = Vector{Int}(undef, n_contours + 1)
    offsets[1] = 0
    for ci in 1:n_contours
        offsets[ci + 1] = offsets[ci] + nnodes(contours[ci])
    end

    # Thread over target nodes only once the workload is large enough to pay for it.
    if _should_thread_velocity(N)
        Threads.@threads for i in 1:N
            ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
            ci = clamp(ci, 1, n_contours)
            local_i = i - offsets[ci]
            (1 <= local_i <= nnodes(contours[ci])) || throw(BoundsError(contours[ci].nodes, local_i))
            xi = contours[ci].nodes[local_i]

            v = zero(SVector{2,T})
            for c in contours
                local nc = nnodes(c)
                nc < 2 && continue
                pv = c.pv
                @inbounds for j in 1:nc
                    a = c.nodes[j]
                    b = next_node(c, j)
                    v = v + pv * segment_velocity(kernel, domain, xi, a, b, ewald)
                end
            end
            vel[i] = v
        end
    else
        for i in 1:N
            ci = searchsortedlast(offsets, i - 1, 1, n_contours + 1, Base.Order.Forward)
            ci = clamp(ci, 1, n_contours)
            local_i = i - offsets[ci]
            (1 <= local_i <= nnodes(contours[ci])) || throw(BoundsError(contours[ci].nodes, local_i))
            xi = contours[ci].nodes[local_i]

            v = zero(SVector{2,T})
            for c in contours
                local nc = nnodes(c)
                nc < 2 && continue
                pv = c.pv
                @inbounds for j in 1:nc
                    a = c.nodes[j]
                    b = next_node(c, j)
                    v = v + pv * segment_velocity(kernel, domain, xi, a, b, ewald)
                end
            end
            vel[i] = v
        end
    end

    return vel
end

"""
    velocity!(vel, prob::ContourProblem)

Compute velocity at every contour node of `prob`, storing results in `vel`.

Current large-problem policy for single-layer CPU problems:

- small problems: direct evaluator
- large problems: proxy FMM when `_FMM_ACCELERATION_ENABLED = true`, otherwise treecode

The same FMM-vs-treecode policy is used for periodic single-layer problems and
multi-layer problems.
"""
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{EulerKernel, UnboundedDomain, T, CPU}) where {T}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))

    if _FMM_ACCELERATION_ENABLED && N >= _FMM_THRESHOLD
        _fmm_velocity!(vel, prob)
    elseif N >= _FMM_THRESHOLD
        _treecode_velocity!(vel, prob)
    else
        _ka_velocity!(vel, prob, prob.dev)
    end

    return vel
end

function velocity!(vel::Vector{SVector{2,T}}, prob::ContourProblem{<:AbstractKernel, <:AbstractDomain, T, CPU}) where {T}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))

    if _FMM_ACCELERATION_ENABLED && N >= _FMM_THRESHOLD
        _fmm_velocity!(vel, prob)
    elseif N >= _FMM_THRESHOLD
        _treecode_velocity!(vel, prob)
    else
        _direct_velocity!(vel, prob)
    end

    return vel
end

"""
    velocity(prob::ContourProblem, x::SVector{2,T})

Compute velocity at a single point `x` from all contours in `prob`.
"""
function velocity(prob::ContourProblem, x::SVector{2,T}) where {T}
    v = zero(SVector{2,T})
    ewald = _prefetch_ewald(prob.domain, prob.kernel)
    for c in prob.contours
        nc = nnodes(c)
        nc < 2 && continue
        for j in 1:nc
            a = c.nodes[j]
            b = next_node(c, j)
            v = v + c.pv * segment_velocity(prob.kernel, prob.domain, x, a, b, ewald)
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
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv
    ewald = _prefetch_ewald(domain, EulerKernel())

    vel = MVector{N, SVector{2,T}}(ntuple(_ -> zero(SVector{2,T}), Val(N)))

    for mode in 1:N
        lam = evals[mode]
        mode_kernel = abs(lam) < eps(T) * 100 ? EulerKernel() :
                      QGKernel(one(T) / sqrt(abs(lam)))

        v_mode = zero(SVector{2,T})
        for source_layer in 1:N
            source_weight = P_inv[mode, source_layer]
            abs(source_weight) < eps(T) && continue
            for sc in prob.layers[source_layer]
                nsc = nnodes(sc)
                nsc < 2 && continue
                for sj in 1:nsc
                    a = sc.nodes[sj]
                    b = next_node(sc, sj)
                    v_mode = v_mode + source_weight * sc.pv *
                        segment_velocity(mode_kernel, domain, x, a, b, ewald)
                end
            end
        end

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
                    for sc in prob.layers[source_layer]
                        nsc = nnodes(sc)
                        nsc < 2 && continue
                        for sj in 1:nsc
                            a = sc.nodes[sj]
                            b = next_node(sc, sj)
                            v_mode = v_mode + source_weight * sc.pv *
                                segment_velocity(mode_kernel, domain, x, a, b, ewald)
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
                    for sc in prob.layers[source_layer]
                        nsc = nnodes(sc)
                        nsc < 2 && continue
                        for sj in 1:nsc
                            a = sc.nodes[sj]
                            b = next_node(sc, sj)
                            v_mode = v_mode + source_weight * sc.pv *
                                segment_velocity(mode_kernel, domain, x, a, b, ewald)
                        end
                    end
                end
                mode_vel[ti] = v_mode
            end
        end

        for ti in 1:n_target
            vel[target_layer][ti] = vel[target_layer][ti] + projection_weight * mode_vel[ti]
        end
    end
end

"""
    _direct_velocity!(vel, prob::MultiLayerContourProblem)

Direct O(N^2) velocity computation at every contour node across all layers of
`prob`, storing results in `vel`. Uses modal decomposition with direct summation.
"""
# Scratch buffers for multi-layer velocity, allocated per-call to avoid
# thread-safety issues when multiple problems are evaluated concurrently.
# The allocation cost is negligible relative to the O(N²) velocity computation.
function _alloc_ml_scratch(::Type{T}, max_nodes::Int) where {T}
    tn = Vector{SVector{2,T}}(undef, max_nodes)
    mv = Vector{SVector{2,T}}(undef, max_nodes)
    return (tn, mv)
end

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

    # Reuse module-level scratch buffers sized to the largest layer.
    # SAFETY INVARIANT: these buffers are shared across the sequential mode and
    # target_layer loops below.  Only the innermost @threads loop (over target
    # nodes) runs in parallel, and each thread writes to a distinct index.
    # Parallelizing the outer `for mode` or `for target_layer` loops would
    # require per-thread copies of target_nodes and mode_vel.
    max_nodes = maximum(sum(nnodes(c) for c in prob.layers[i]; init=0) for i in 1:N)
    target_nodes, mode_vel = _alloc_ml_scratch(T, max_nodes)

    for mode in 1:N
        lam = evals[mode]

        if abs(lam) < eps(T) * 100
            _multilayer_mode_velocity!(vel, prob, mode, EulerKernel(),
                                       target_nodes, mode_vel, ewald)
        else
            Ld_mode = one(T) / sqrt(abs(lam))
            _multilayer_mode_velocity!(vel, prob, mode, QGKernel(Ld_mode),
                                       target_nodes, mode_vel, ewald)
        end
    end

    return vel
end

@inline function _multilayer_layer_ranges(prob::MultiLayerContourProblem{N}) where {N}
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
    kernel = prob.kernel
    domain = prob.domain
    evals = kernel.eigenvalues
    P = kernel.eigenvectors
    P_inv = kernel.eigenvectors_inv

    total = total_nodes(prob)
    layer_ranges = _multilayer_layer_ranges(prob)
    n_contours = sum(length(prob.layers[i]) for i in 1:N)
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

        ci = 1
        for layer in 1:N
            weight = P_inv[mode, layer]
            for c in prob.layers[layer]
                weighted[ci] = PVContour(c.nodes, weight * c.pv, c.wrap)
                ci += 1
            end
        end

        mode_prob = ContourProblem(mode_kernel, domain, weighted; dev=dev)
        _ka_velocity!(mode_vel, mode_prob, dev)

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

function _gpu_velocity_policy!(vel::Vector{SVector{2,T}},
                               prob::ContourProblem{K, D, T, GPU}) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, D<:AbstractDomain, T}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))

    if _FMM_ACCELERATION_ENABLED && N >= _FMM_THRESHOLD
        _fmm_velocity!(vel, prob)
    elseif N >= _FMM_THRESHOLD
        _treecode_velocity!(vel, prob)
    else
        _ka_velocity!(vel, prob, prob.dev)
    end

    return vel
end

function _gpu_multilayer_velocity_policy!(vel::NTuple{N, Vector{SVector{2,T}}},
                                          prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T}
    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        length(vel[i]) >= n_layer || throw(DimensionMismatch("vel[$i] length ($(length(vel[i]))) must be >= layer $i nodes ($n_layer)"))
    end

    Ntot = total_nodes(prob)
    if _FMM_ACCELERATION_ENABLED && Ntot >= _FMM_THRESHOLD
        _fmm_velocity!(vel, prob)
    elseif Ntot >= _FMM_THRESHOLD
        _treecode_velocity!(vel, prob)
    else
        _ka_multilayer_velocity!(vel, prob, prob.dev)
    end

    return vel
end

"""
    velocity!(vel, prob::MultiLayerContourProblem)

Compute velocity at all nodes across all layers using modal decomposition.

Current large-problem policy for multi-layer CPU problems:

- small problems: direct evaluator
- large problems: production treecode
- `_FMM_ACCELERATION_ENABLED = true`: still falls back to the current direct
  multi-layer `_fmm_velocity!` stub, so there is no production multi-layer FMM

In other words, multi-layer acceleration currently means treecode, not FMM.
"""
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N, <:Any, <:Any, T, CPU}) where {N, T}
    for i in 1:N
        n_layer = sum(nnodes(c) for c in prob.layers[i]; init=0)
        length(vel[i]) >= n_layer || throw(DimensionMismatch("vel[$i] length ($(length(vel[i]))) must be >= layer $i nodes ($n_layer)"))
    end
    Ntot = total_nodes(prob)
    if _FMM_ACCELERATION_ENABLED && Ntot >= _FMM_THRESHOLD
        _fmm_velocity!(vel, prob)
    elseif Ntot >= _FMM_THRESHOLD
        _treecode_velocity!(vel, prob)
    else
        _direct_velocity!(vel, prob)
    end
    return vel
end

# GPU dispatch — velocity computed in SoA layout via KernelAbstractions,
# then repacked into the CPU vel buffer.
# Uses a cached workspace to avoid repeated GPU/CPU allocations across
# the 4 velocity evaluations per RK4 step.
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, UnboundedDomain, T, GPU}) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, T}
    return _gpu_velocity_policy!(vel, prob)
end

function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, PeriodicDomain{T}, T, GPU}) where {K<:Union{EulerKernel,QGKernel,SQGKernel}, T}
    return _gpu_velocity_policy!(vel, prob)
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

# GPU fallback for multi-layer problems
function velocity!(vel::NTuple{N, Vector{SVector{2,T}}},
                   prob::MultiLayerContourProblem{N, <:Any, <:Any, T, GPU}) where {N, T}
    return _gpu_multilayer_velocity_policy!(vel, prob)
end
