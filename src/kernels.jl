# Contour dynamics velocity implementations for unbounded domains.
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

# 5-point Gauss-Legendre nodes and weights on [-1,1].
# Precomputed for Float64 and Float32 to avoid repeated sqrt/division in
# the innermost velocity loop. Generic fallback for BigFloat etc.
let
    _n2_64 = sqrt((5.0 - 2.0 * sqrt(10.0/7.0)) / 9.0)
    _n3_64 = sqrt((5.0 + 2.0 * sqrt(10.0/7.0)) / 9.0)
    _w1_64 = 128.0 / 225.0
    _w2_64 = (322.0 + 13.0 * sqrt(70.0)) / 900.0
    _w3_64 = (322.0 - 13.0 * sqrt(70.0)) / 900.0
    global const _GL5_NODES_F64 = SVector{5,Float64}(-_n3_64, -_n2_64, 0.0, _n2_64, _n3_64)
    global const _GL5_WEIGHTS_F64 = SVector{5,Float64}(_w3_64, _w2_64, _w1_64, _w2_64, _w3_64)
    global const _GL5_NODES_F32 = SVector{5,Float32}(Float32.(_GL5_NODES_F64)...)
    global const _GL5_WEIGHTS_F32 = SVector{5,Float32}(Float32.(_GL5_WEIGHTS_F64)...)

    _n1_64 = sqrt(3.0/5.0)
    global const _GL3_NODES_F64 = SVector{3,Float64}(-_n1_64, 0.0, _n1_64)
    global const _GL3_WEIGHTS_F64 = SVector{3,Float64}(5.0/9.0, 8.0/9.0, 5.0/9.0)
    global const _GL3_NODES_F32 = SVector{3,Float32}(Float32.(_GL3_NODES_F64)...)
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
    segment_velocity(::EulerKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from node `a`
to node `b` with unit PV jump, using the 2D Euler Green's function in an
unbounded domain.

Computes the contour dynamics integral analytically:
  v_seg = -(1/(4π)) * (bx-ax, by-ay) * ∫₀¹ log|x - a - t(b-a)|² dt

The velocity direction is along `ds = b - a`, not rotated.
"""
# Antiderivative for the Euler segment velocity integral.
# F(u; h, h_sq) = u*log(u² + h²) - 2u + 2h*arctan(u/h)
@inline function _euler_antideriv(u::T, h::T, h_sq::T) where {T}
    r2 = u * u + h_sq
    if r2 < eps(T)^2
        return zero(T)
    end
    val = u * log(r2) - 2 * u
    if abs(h) > eps(T)
        val += 2 * h * atan(u / h)
    end
    # When h ≈ 0 the atan term vanishes: lim_{h→0} 2h·atan(u/h) = 0.
    return val
end

function segment_velocity(::EulerKernel, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ds = b - a
    ds_len_sq = ds[1]^2 + ds[2]^2
    ds_len = sqrt(ds_len_sq)

    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    t_hat = ds / ds_len
    n_hat = SVector{2,T}(-t_hat[2], t_hat[1])

    r0 = x - a  # vector from a to x
    # Project onto segment coordinates
    u_a = r0[1] * t_hat[1] + r0[2] * t_hat[2]   # tangential component
    h   = r0[1] * n_hat[1] + r0[2] * n_hat[2]    # normal component
    u_b = u_a - ds_len

    h_sq = h * h

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * t_hat * (_euler_antideriv(u_a, h, h_sq) - _euler_antideriv(u_b, h, h_sq))
end

# Unbounded domains don't use Ewald caches; ignore the argument.
@inline segment_velocity(k::AbstractKernel, d::UnboundedDomain,
                          x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T},
                          ::Nothing) where {T} =
    segment_velocity(k, d, x, a, b)

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

    # Thread over target nodes — each node accumulates its velocity independently.
    Threads.@threads for i in 1:N
        # Binary search to map flat index i → (contour index, local index)
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
            for j in 1:nc
                a = c.nodes[j]
                b = next_node(c, j)
                v = v + pv * segment_velocity(kernel, domain, xi, a, b, ewald)
            end
        end
        vel[i] = v
    end

    return vel
end

"""
    velocity!(vel, prob::ContourProblem)

Compute velocity at every contour node of `prob`, storing results in `vel`.
Uses the proxy-FMM path only when that experimental accelerator is explicitly
enabled. Otherwise, large single-layer problems use the production treecode
path and small problems fall back to the validated direct evaluator.
"""
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
    segment_velocity(::QGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a vortex patch contour segment from `a` to `b`
using the QG Green's function G(r) = -1/(2π) K₀(r/Ld).

The contour dynamics velocity is:
  v_seg = (1/(2π)) ∫₀¹ K₀(|x-P(t)|/Ld) ds dt

Uses singular subtraction: the log singularity in K₀ is handled analytically
(matching the Euler kernel), and the smooth remainder [K₀(r/Ld) + log(r)]
is integrated with 5-point Gauss-Legendre quadrature.
"""
function segment_velocity(kernel::QGKernel{T}, domain::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    Ld = kernel.Ld
    ds = b - a
    ds_len = sqrt(ds[1]^2 + ds[2]^2)
    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    # Analytic Euler contribution (handles the log singularity)
    v_euler = segment_velocity(EulerKernel(), domain, x, a, b)

    # 5-point Gauss-Legendre on [-1, 1] for the smooth correction
    # Correction integrand: K₀(r/Ld) + log(r) which is smooth at r=0
    # (since K₀(r/Ld) ~ -log(r) + log(2Ld) - γ as r→0)
    # The full QG integral is:
    #   v_seg = (1/(2π)) ds ∫₀¹ K₀(r/Ld) dt
    #         = (1/(2π)) ds ∫₀¹ [-log(r) + (K₀(r/Ld) + log(r))] dt
    # The -log(r) part gives the Euler contribution (with appropriate factors).
    # The correction is: (1/(2π)) ds ∫₀¹ [K₀(r/Ld) + log(r)] dt
    # This uses log(r) (not log(r/Ld)) to match the Euler singularity exactly.
    # Using log(r/Ld) would introduce a per-segment error of -log(Ld)*ds that
    # cancels for closed contours (Σ ds = 0) but not for spanning contours.

    g_nodes, g_weights = _gl5_nodes_weights(T)

    mid = (a + b) / 2
    half_ds = ds / 2

    corr_integral = zero(T)
    inv2pi = one(T) / (2 * T(π))

    for q in 1:5
        s = mid + g_nodes[q] * half_ds
        r_vec = s - x
        r2 = r_vec[1]^2 + r_vec[2]^2

        if r2 < eps(T)^2
            # K₀(r/Ld) + log(r) → log(2Ld) - γ as r→0, finite
            corr_integral += g_weights[q] * (log(2 * Ld) - T(Base.MathConstants.eulergamma))
            continue
        end

        r = sqrt(r2)
        rr = r / Ld
        # K₀(r/Ld) + log(r) is smooth and bounded near r=0
        corr_integral += g_weights[q] * (besselk(0, rr) + log(r))
    end

    # Scale: the Gauss quadrature approximates ∫₋₁¹ f(t) dt, and our
    # parameterization maps [-1,1] to [0,1] via t → (1+t)/2, giving factor 1/2.
    # But the segment parameterization is already handled by the mid/half_ds.
    # So: ∫₀¹ correction dt ≈ (1/2) * sum(w_i * f(t_i))
    corr_integral *= T(0.5)  # [-1,1] to [0,1] Jacobian

    # v_seg_QG = v_Euler + v_corr where v_Euler already has the correct sign
    # The Euler part handles the -log(r) singularity analytically.
    # The correction adds: (1/(2π)) * ds * ∫₀¹ [K₀(r/Ld) + log(r)] dt
    # (per unit PV; pv is multiplied in velocity!).
    v_corr = inv2pi * ds * corr_integral

    return v_euler + v_corr
end

"""
    segment_velocity(::SQGKernel, ::UnboundedDomain, x, a, b)

Velocity at point `x` due to a surface buoyancy patch contour segment from
node `a` to node `b` with unit buoyancy jump, using the regularized SQG
Green's function `G(r) = -1/(2π√(r²+δ²))`.

The contour integral is:
  v_seg = -(1/(2π)) t̂ [F(u_a) - F(u_b)]

where `F(u) = log(u + √(u² + h_eff²))` and `h_eff² = h² + δ²`.

The `-(1/(2π))` prefactor (vs `-(1/(4π))` for Euler) reflects the different
Green's function normalisation: SQG uses `1/(2πr)` while Euler uses
`log(r²)/(4π)`.
"""
function segment_velocity(kernel::SQGKernel{T}, ::UnboundedDomain,
                           x::SVector{2,T}, a::SVector{2,T}, b::SVector{2,T}) where {T}
    ds = b - a
    ds_len_sq = ds[1]^2 + ds[2]^2
    ds_len = sqrt(ds_len_sq)

    if ds_len < eps(T)
        return zero(SVector{2,T})
    end

    t_hat = ds / ds_len
    n_hat = SVector{2,T}(-t_hat[2], t_hat[1])

    r0 = x - a  # vector from a to x
    # Project onto segment coordinates
    u_a = r0[1] * t_hat[1] + r0[2] * t_hat[2]   # tangential component
    h   = r0[1] * n_hat[1] + r0[2] * n_hat[2]    # normal component
    u_b = u_a - ds_len

    h_eff_sq = h * h + kernel.delta^2
    h_eff = sqrt(h_eff_sq)

    # Antiderivative F(u) = arcsinh(u / h_eff)
    # Numerically stable form of log(u + √(u² + h_eff²)) — avoids
    # catastrophic cancellation when u is large negative.
    F_a = asinh(u_a / h_eff)
    F_b = asinh(u_b / h_eff)

    inv2pi = one(T) / (2 * T(π))
    return -inv2pi * t_hat * (F_a - F_b)
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

"""
    velocity!(vel, prob::MultiLayerContourProblem)

Compute velocity at all nodes across all layers using modal decomposition.
Uses the FMM path only when acceleration is explicitly enabled and the problem
is large enough; otherwise falls back to the validated direct evaluator.
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
                   prob::ContourProblem{EulerKernel, UnboundedDomain, T, GPU}) where {T}
    N = total_nodes(prob)
    length(vel) >= N || throw(DimensionMismatch("vel length ($(length(vel))) must be >= total nodes ($N)"))
    N == 0 && return vel

    dev = prob.dev
    ws = _get_gpu_workspace!(dev, T, N)
    _gpu_velocity_ws!(ws, prob, dev)

    # Pack CPU results into SVector format
    vx = ws.cpu_vx
    vy = ws.cpu_vy
    @inbounds for i in 1:N
        vel[i] = SVector{2,T}(vx[i], vy[i])
    end

    return vel
end

# Fallback for unsupported GPU kernel/domain combinations
function velocity!(vel::Vector{SVector{2,T}},
                   prob::ContourProblem{K, D, T, GPU}) where {K, D, T}
    throw(ArgumentError(
        "GPU velocity is only implemented for EulerKernel on UnboundedDomain. " *
        "Got $(typeof(prob.kernel)) on $(typeof(prob.domain)). " *
        "Use dev=CPU() for other kernel/domain combinations."))
end
