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
    child_to_parent::NTuple{4, Matrix{T}}   # M2M: child proxy → parent proxy
    parent_to_child::NTuple{4, Matrix{T}}   # L2L: parent local → child local
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

@inline _kernel_value(kernel::EulerKernel, ::PeriodicDomain{T},
                      x::SVector{2,T}, y::SVector{2,T}) where {T} =
    _kernel_value(kernel, UnboundedDomain(), x, y)

function _kernel_value(kernel::QGKernel{T}, ::UnboundedDomain,
                       x::SVector{2,T}, y::SVector{2,T}) where {T}
    dx = x[1] - y[1]
    dy = x[2] - y[2]
    r2 = dx * dx + dy * dy
    r2 < eps(T) && return zero(T)
    r = sqrt(r2)
    return besselk(0, r / kernel.Ld) / (2 * T(pi))
end

@inline _kernel_value(kernel::QGKernel{T}, ::PeriodicDomain{T},
                      x::SVector{2,T}, y::SVector{2,T}) where {T} =
    _kernel_value(kernel, UnboundedDomain(), x, y)

function _kernel_value(kernel::SQGKernel{T}, ::UnboundedDomain,
                       x::SVector{2,T}, y::SVector{2,T}) where {T}
    dx = x[1] - y[1]
    dy = x[2] - y[2]
    r2 = dx * dx + dy * dy
    return -one(T) / (2 * T(pi) * sqrt(r2 + kernel.delta^2))
end

@inline _kernel_value(kernel::SQGKernel{T}, ::PeriodicDomain{T},
                      x::SVector{2,T}, y::SVector{2,T}) where {T} =
    _kernel_value(kernel, UnboundedDomain(), x, y)

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
        # L2L operators: for each child quadrant, build
        #   L2L_q = pinv(K_child) * K(child_check, parent_proxy)
        # where parent is at the parent level (one level up)
        parent_hw = hw * T(2)
        parent_check_pts = _check_points(origin, parent_hw, p_check)

        # Compute parent pseudoinverse once (shared by all 4 quadrants)
        parent_proxy_pts = _proxy_points(origin, parent_hw, p)
        K_parent_cp = _build_kernel_matrix(kernel, domain, parent_check_pts, parent_proxy_pts)
        F_parent = svd(K_parent_cp)
        cutoff_parent = eps(T) * T(100) * F_parent.S[1]
        S_inv_parent = [s > cutoff_parent ? one(T) / s : zero(T) for s in F_parent.S]
        pinv_parent = F_parent.Vt' * Diagonal(S_inv_parent) * F_parent.U'

        child_to_parent = ntuple(Val(4)) do q
            child_center = _child_center(origin, parent_hw, q)
            child_proxy_pts = _proxy_points(child_center, hw, p)
            K_pc = _build_kernel_matrix(kernel, domain, parent_check_pts, child_proxy_pts)
            Matrix{T}(pinv_parent * K_pc)
        end

        # L2L: for each child quadrant, map parent proxy strengths → child proxy strengths
        # L2L_q = pinv(K_child_check_child_proxy) * K(child_check, parent_proxy)
        parent_proxy_pts_for_l2l = _proxy_points(origin, parent_hw, p)
        parent_to_child = ntuple(Val(4)) do q
            child_center = _child_center(origin, parent_hw, q)
            child_check_pts = _check_points(child_center, hw, p_check)
            child_proxy_pts = _proxy_points(child_center, hw, p)
            K_child_cp = _build_kernel_matrix(kernel, domain, child_check_pts, child_proxy_pts)
            F_child = svd(K_child_cp)
            cutoff_child = eps(T) * T(100) * F_child.S[1]
            S_inv_child = [s > cutoff_child ? one(T) / s : zero(T) for s in F_child.S]
            pinv_child = F_child.Vt' * Diagonal(S_inv_child) * F_child.U'
            K_cp_parent = _build_kernel_matrix(kernel, domain, child_check_pts, parent_proxy_pts_for_l2l)
            Matrix{T}(pinv_child * K_cp_parent)
        end

        ops[level + 1] = LevelOperators{T}(Matrix{T}(pinv_K), child_to_parent, parent_to_child)
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

    # The least-squares solves below dispatch to LAPACK, which is itself
    # multithreaded.  Running LAPACK from multiple Julia threads causes
    # oversubscription (and can deadlock with OpenBLAS).  Pin BLAS to one
    # thread for the duration of this parallel section.
    prev_blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try

    Threads.@threads for li_idx in 1:length(leaves)
        leaf = leaves[li_idx]
        box = tree.boxes[leaf]
        level = box.level

        # Check surfaces for this leaf. Using two radii constrains the exterior
        # harmonic continuation much better than a single collocation circle.
        check_pts_inner = _check_points(box.center, box.half_width, p_check)
        check_pts_outer = _check_points(box.center, box.half_width, p_check; radius_ratio=T(4))
        check_pts = vcat(check_pts_inner, check_pts_outer)

        # Evaluate velocity from all segments in this box at the check points
        n_check = length(check_pts)
        vel_check_x = Vector{T}(undef, n_check)
        vel_check_y = Vector{T}(undef, n_check)

        seg_range = box.segment_range
        @inbounds for ic in 1:n_check
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

        # Fit the equivalent-source strengths directly for this leaf.
        # The precomputed truncated pseudoinverse proved too fragile for the
        # current proxy setup and significantly degraded even single-box S2M
        # accuracy.
        proxy_pts = _proxy_points(box.center, box.half_width, p)
        K_cp = _build_kernel_matrix(kernel, domain, check_pts, proxy_pts)

        # For Euler kernel (log singularity), the equivalent source strengths
        # must have zero net charge to avoid polluting the far field with an
        # unphysical logarithmic monopole. For QG/SQG kernels the Green's function
        # decays at infinity, so the constraint is unnecessary and would
        # overconstrain the fit.
        if kernel isa EulerKernel
            K_aug = vcat(K_cp, reshape(fill(one(T), p), 1, p))
            rhs_x = vcat(vel_check_x, zero(T))
            rhs_y = vcat(vel_check_y, zero(T))
        else
            K_aug = K_cp
            rhs_x = vel_check_x
            rhs_y = vel_check_y
        end
        strengths_x = K_aug \ rhs_x
        strengths_y = K_aug \ rhs_y

        # Store as vector of SVector{2,T}
        equiv = proxy_data[leaf].equiv_strengths
        @inbounds for k in 1:p
            equiv[k] = SVector{2,T}(strengths_x[k], strengths_y[k])
        end
    end

    finally
        BLAS.set_num_threads(prev_blas_threads)
    end
    return nothing
end
