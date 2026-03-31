# Proxy surface / equivalent charge methods for kernel-independent FMM.

# ── Constants ───────────────────────────────────────────────

"""Number of proxy (equivalent) points on the proxy surface."""
const _FMM_PROXY_ORDER = 36

"""Number of check points on the check surface."""
const _FMM_CHECK_ORDER = 72

"""Ratio of proxy surface radius to box half-width."""
const _FMM_PROXY_RADIUS = 1.5

"""Ratio of check surface radius to box half-width."""
const _FMM_CHECK_RADIUS = 2.5

# ── Types ───────────────────────────────────────────────────

"""
    ProxyData{T}

Equivalent-source and local-expansion strengths for a single box.

- `equiv_strengths`: proxy charges that reproduce the far field (from S2M / M2M).
- `local_strengths`: proxy charges that reproduce the local incoming field (from L2L).
"""
struct ProxyData{T<:AbstractFloat}
    equiv_strengths::Vector{SVector{2,T}}
    local_strengths::Vector{SVector{2,T}}
end

"""
    LevelOperators{T}

Precomputed operators for one level of the quadtree.

- `check_to_proxy_pinv`: pseudoinverse mapping check-surface values to proxy strengths.
- `child_to_parent`: M2M translation operators for the four child quadrants (SW, SE, NW, NE).
"""
struct LevelOperators{T<:AbstractFloat}
    check_to_proxy_pinv::Matrix{T}
    child_to_parent::NTuple{4, Matrix{T}}
end

# ── Point generation ────────────────────────────────────────

"""
    _proxy_points(center, half_width, p; radius_ratio=_FMM_PROXY_RADIUS)

Generate `p` equispaced points on a circle of radius `radius_ratio * half_width`
centered at `center`.  Uses `cospi`/`sinpi` for accuracy at cardinal angles.
"""
function _proxy_points(center::SVector{2,T}, half_width::T, p::Int;
                       radius_ratio::T = T(_FMM_PROXY_RADIUS)) where {T<:AbstractFloat}
    r = radius_ratio * half_width
    pts = Vector{SVector{2,T}}(undef, p)
    @inbounds for k in 1:p
        theta_over_pi = 2 * T(k - 1) / T(p)   # angle / pi
        pts[k] = SVector{2,T}(center[1] + r * cospi(theta_over_pi),
                               center[2] + r * sinpi(theta_over_pi))
    end
    return pts
end

"""
    _check_points(center, half_width, p_check; radius_ratio=_FMM_CHECK_RADIUS)

Generate `p_check` equispaced points on a circle of radius `radius_ratio * half_width`
centered at `center`.  Uses `cospi`/`sinpi` for accuracy.
"""
function _check_points(center::SVector{2,T}, half_width::T, p_check::Int;
                       radius_ratio::T = T(_FMM_CHECK_RADIUS)) where {T<:AbstractFloat}
    r = radius_ratio * half_width
    pts = Vector{SVector{2,T}}(undef, p_check)
    @inbounds for k in 1:p_check
        theta_over_pi = 2 * T(k - 1) / T(p_check)
        pts[k] = SVector{2,T}(center[1] + r * cospi(theta_over_pi),
                               center[2] + r * sinpi(theta_over_pi))
    end
    return pts
end

# ── Kernel evaluation (point-to-point) ─────────────────────

"""
    _kernel_value(kernel, domain, x, y)

Evaluate the scalar Green's function G(|x - y|) for the given kernel and domain.
Returns a scalar value.
"""
function _kernel_value(::EulerKernel, ::UnboundedDomain,
                       x::SVector{2,T}, y::SVector{2,T}) where {T}
    dx = x[1] - y[1]
    dy = x[2] - y[2]
    r2 = dx * dx + dy * dy
    r2 < eps(T) && return zero(T)
    return -log(r2) / (4 * T(pi))
end

function _kernel_value(kernel::QGKernel{T}, ::UnboundedDomain,
                       x::SVector{2,T}, y::SVector{2,T}) where {T}
    dx = x[1] - y[1]
    dy = x[2] - y[2]
    r2 = dx * dx + dy * dy
    r2 < eps(T) && return zero(T)
    r = sqrt(r2)
    return besselk(0, r / kernel.Ld) / (2 * T(pi))
end

function _kernel_value(kernel::SQGKernel{T}, ::UnboundedDomain,
                       x::SVector{2,T}, y::SVector{2,T}) where {T}
    dx = x[1] - y[1]
    dy = x[2] - y[2]
    r2 = dx * dx + dy * dy
    return -one(T) / (2 * T(pi) * sqrt(r2 + kernel.delta^2))
end

# ── Kernel matrix construction ─────────────────────────────

"""
    _build_kernel_matrix(kernel, domain, targets, sources)

Build the kernel matrix `K[i,j] = _kernel_value(kernel, domain, targets[i], sources[j])`.
"""
function _build_kernel_matrix(kernel::AbstractKernel, domain::AbstractDomain,
                              targets::Vector{SVector{2,T}},
                              sources::Vector{SVector{2,T}}) where {T}
    nt = length(targets)
    ns = length(sources)
    K = Matrix{T}(undef, nt, ns)
    @inbounds for j in 1:ns
        for i in 1:nt
            K[i, j] = _kernel_value(kernel, domain, targets[i], sources[j])
        end
    end
    return K
end

# ── Level operator precomputation ──────────────────────────

"""
    precompute_level_operators(tree, kernel[, domain]; p, p_check)

For each level of the quadtree, precompute:
- The pseudoinverse of the check-to-proxy kernel matrix (truncated SVD).
- M2M operators mapping child proxy strengths to parent proxy strengths for
  each of the four child quadrants.

Returns a `Vector{LevelOperators{T}}` of length `tree.max_level + 1`.
"""
function precompute_level_operators(
    tree::FMMTree{T},
    kernel::AbstractKernel,
    domain::AbstractDomain = UnboundedDomain();
    p::Int = _FMM_PROXY_ORDER,
    p_check::Int = _FMM_CHECK_ORDER,
) where {T}
    max_level = tree.max_level
    root_hw = tree.boxes[1].half_width
    ops = Vector{LevelOperators{T}}(undef, max_level + 1)

    origin = SVector{2,T}(zero(T), zero(T))

    for level in 0:max_level
        hw = root_hw / T(2)^level

        # Proxy and check points for a canonical box centred at the origin
        proxy_pts = _proxy_points(origin, hw, p)
        check_pts = _check_points(origin, hw, p_check)

        # Kernel matrix: K(check, proxy)
        K_cp = _build_kernel_matrix(kernel, domain, check_pts, proxy_pts)

        # Truncated SVD pseudoinverse
        F = svd(K_cp)
        cutoff = eps(T) * T(100) * F.S[1]
        S_inv = [s > cutoff ? one(T) / s : zero(T) for s in F.S]
        pinv_K = F.Vt' * Diagonal(S_inv) * F.U'  # p x p_check

        # M2M operators: for each child quadrant, build
        #   M2M_q = pinv(K_parent) * K(parent_check, child_proxy)
        # where parent_check is at the parent level (one level up)
        parent_hw = hw * T(2)
        parent_check_pts = _check_points(origin, parent_hw, p_check)

        child_to_parent = ntuple(Val(4)) do q
            child_center = _child_center(origin, parent_hw, q)
            child_proxy_pts = _proxy_points(child_center, hw, p)

            # Kernel matrix: parent_check evaluated at child proxy sources
            K_pc = _build_kernel_matrix(kernel, domain, parent_check_pts, child_proxy_pts)

            # For the parent level, we need the parent's pseudoinverse
            # But since the parent is canonical (centred at origin with parent_hw),
            # we build it here
            parent_proxy_pts = _proxy_points(origin, parent_hw, p)
            K_parent_cp = _build_kernel_matrix(kernel, domain, parent_check_pts, parent_proxy_pts)
            F_parent = svd(K_parent_cp)
            cutoff_parent = eps(T) * T(100) * F_parent.S[1]
            S_inv_parent = [s > cutoff_parent ? one(T) / s : zero(T) for s in F_parent.S]
            pinv_parent = F_parent.Vt' * Diagonal(S_inv_parent) * F_parent.U'

            Matrix{T}(pinv_parent * K_pc)
        end

        ops[level + 1] = LevelOperators{T}(Matrix{T}(pinv_K), child_to_parent)
    end

    return ops
end

# ── Source-to-Multipole (S2M) ──────────────────────────────

"""
    _s2m!(proxy_data, tree, contours, kernel, domain, ops, ewald_cache; p, p_check)

For each leaf box, compute equivalent proxy strengths that reproduce the
far field of segments inside the box.

The procedure evaluates the velocity from all segments in the leaf at check
surface points, then applies the pseudoinverse to obtain proxy strengths.
The x- and y-components are treated independently (scalar proxy expansion
applied component-wise).
"""
function _s2m!(
    proxy_data::Vector{ProxyData{T}},
    tree::FMMTree{T},
    contours::AbstractVector{PVContour{T}},
    kernel::AbstractKernel,
    domain::AbstractDomain,
    ops::Vector{LevelOperators{T}},
    ewald_cache;
    p::Int = _FMM_PROXY_ORDER,
    p_check::Int = _FMM_CHECK_ORDER,
) where {T}
    leaves = tree.leaf_indices

    Threads.@threads for li_idx in 1:length(leaves)
        leaf = leaves[li_idx]
        box = tree.boxes[leaf]
        level = box.level

        # Check surface points for this leaf
        check_pts = _check_points(box.center, box.half_width, p_check)

        # Evaluate velocity from all segments in this box at the check points
        vel_check_x = Vector{T}(undef, p_check)
        vel_check_y = Vector{T}(undef, p_check)

        seg_range = box.segment_range
        @inbounds for ic in 1:p_check
            xc = check_pts[ic]
            vx = zero(T)
            vy = zero(T)
            for si in seg_range
                ci, nj = tree.sorted_segments[si]
                c = contours[ci]
                a = c.nodes[nj]
                b = next_node(c, nj)
                v = c.pv * segment_velocity(kernel, domain, xc, a, b, ewald_cache)
                vx += v[1]
                vy += v[2]
            end
            vel_check_x[ic] = vx
            vel_check_y[ic] = vy
        end

        # Apply pseudoinverse to get equivalent strengths
        pinv = ops[level + 1].check_to_proxy_pinv
        strengths_x = pinv * vel_check_x   # length p
        strengths_y = pinv * vel_check_y   # length p

        # Store as vector of SVector{2,T}
        equiv = proxy_data[leaf].equiv_strengths
        @inbounds for k in 1:p
            equiv[k] = SVector{2,T}(strengths_x[k], strengths_y[k])
        end
    end

    return nothing
end
