"""
    vortex_area(c::PVContour)

Signed area enclosed by contour `c` using the shoelace formula.
"""
function vortex_area(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    is_spanning(c) && return zero(T)  # area undefined for spanning contours
    A = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        A += nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
    end
    return A / 2
end

"""
    centroid(c::PVContour)

Centroid of the region enclosed by contour `c`, via Green's theorem.
"""
function centroid(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(SVector{2, T})
    A = vortex_area(c)
    abs(A) < eps(T) && return zero(SVector{2, T})

    cx = zero(T)
    cy = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        cross = nodes[i][1] * nxt[2] - nxt[1] * nodes[i][2]
        cx += (nodes[i][1] + nxt[1]) * cross
        cy += (nodes[i][2] + nxt[2]) * cross
    end
    inv6A = one(T) / (6 * A)
    return SVector{2, T}(cx * inv6A, cy * inv6A)
end

"""
    ellipse_moments(c::PVContour)

Second moments → (aspect_ratio, orientation_angle).
"""
function ellipse_moments(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    A = vortex_area(c)
    ctr = centroid(c)

    Jxx = zero(T)
    Jyy = zero(T)
    Jxy = zero(T)

    @inbounds for i in 1:n
        nxt = next_node(c, i)
        xi, yi = nodes[i][1] - ctr[1], nodes[i][2] - ctr[2]
        xj, yj = nxt[1] - ctr[1], nxt[2] - ctr[2]
        cross = xi * yj - xj * yi
        Jxx += (xi^2 + xi * xj + xj^2) * cross
        Jyy += (yi^2 + yi * yj + yj^2) * cross
        Jxy += (xi * yj + 2 * xi * yi + 2 * xj * yj + xj * yi) * cross
    end

    Jxx /= 12
    Jyy /= 12
    Jxy /= 24

    Jxx /= abs(A)
    Jyy /= abs(A)
    Jxy /= abs(A)

    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy^2
    disc = sqrt(max(zero(T), trace^2 / 4 - det))
    lambda1 = trace / 2 + disc
    lambda2 = trace / 2 - disc

    aspect_ratio = sqrt(max(one(T), lambda1 / max(lambda2, eps(T))))
    angle = T(0.5) * atan(2 * Jxy, Jxx - Jyy)

    return (aspect_ratio, angle)
end

"""
    circulation(prob)

Total circulation `Γ = ∑ qᵢ Aᵢ` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
function circulation(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * vortex_area(c)
    end
    return s
end

"""
    enstrophy(prob)

Enstrophy `½ ∑ qᵢ² Aᵢ` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
function enstrophy(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv^2 * vortex_area(c)
    end
    return s / 2
end

"""
    angular_momentum(prob)

Angular momentum `∑ qᵢ ∫ r² dA` of a [`ContourProblem`](@ref) or
[`MultiLayerContourProblem`](@ref).
"""
function angular_momentum(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * _second_moment_r2(c)
    end
    return s
end

function _second_moment_r2(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    is_spanning(c) && return zero(T)  # moment undefined for spanning contours
    s = zero(T)
    @inbounds for i in 1:n
        nxt = next_node(c, i)
        xi, yi = nodes[i][1], nodes[i][2]
        xj, yj = nxt[1], nxt[2]
        cross = xi * yj - xj * yi
        s += (xi^2 + xi * xj + xj^2) * cross
        s += (yi^2 + yi * yj + yj^2) * cross
    end
    return s / 12
end

"""
    energy(prob)

Kinetic energy of a [`ContourProblem`](@ref) or [`MultiLayerContourProblem`](@ref),
computed via contour integrals of the appropriate Green's function.
"""
function energy(prob::ContourProblem{EulerKernel, UnboundedDomain, T}) where {T}
    contours = prob.contours
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    for ci in contours
        nci = nnodes(ci)
        nci < 2 && continue
        for cj in contours
            ncj = nnodes(cj)
            ncj < 2 && continue
            E += ci.pv * cj.pv * _energy_contour_pair_euler(ci, cj)
        end
    end
    # Factor 1/2: the double sum counts both (i,j) and (j,i) for the symmetric integrand.
    return -inv4pi * E / 2
end

function _energy_contour_pair_euler(ci::PVContour{T}, cj::PVContour{T}) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # 2-point Gauss-Legendre nodes/weights on [-1,1]
    g = one(T) / sqrt(T(3))
    g_nodes = SVector{2,T}(-g, g)
    g_weight = one(T)  # both weights are 1
    # Thread over outer segments, each thread accumulates a partial sum
    partial = zeros(T, nci)
    @inbounds Threads.@threads for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            # 2×2 Gauss-Legendre quadrature over both segments
            quad = zero(T)
            for qi in 1:2
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:2
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    r2 = dx^2 + dy^2
                    r2 < eps(T) && continue
                    quad += g_weight * g_weight * log(r2) / 2
                end
            end
            # Jacobian: each ∫₋₁¹ → ½ ∫₀¹, two of them → ¼
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(partial)
end

function energy(prob::ContourProblem{QGKernel{T}, UnboundedDomain, T}) where {T}
    contours = prob.contours
    Ld = prob.kernel.Ld
    E = zero(T)
    inv4pi = one(T) / (4 * T(π))
    for ci in contours
        nnodes(ci) < 2 && continue
        for cj in contours
            nnodes(cj) < 2 && continue
            E += ci.pv * cj.pv * _energy_contour_pair_qg(ci, cj, Ld)
        end
    end
    return -inv4pi * E / 2
end

function _energy_contour_pair_qg(ci::PVContour{T}, cj::PVContour{T}, Ld::T) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # 2-point Gauss-Legendre nodes/weights on [-1,1]
    g = one(T) / sqrt(T(3))
    g_nodes = SVector{2,T}(-g, g)
    g_weight = one(T)
    partial = zeros(T, nci)
    @inbounds Threads.@threads for i in 1:nci
        ai = ci.nodes[i]
        bi = next_node(ci, i)
        dsi = bi - ai
        midi = (ai + bi) / 2
        half_dsi = dsi / 2
        local_s = zero(T)
        for j in 1:ncj
            aj = cj.nodes[j]
            bj = next_node(cj, j)
            dsj = bj - aj
            midj = (aj + bj) / 2
            half_dsj = dsj / 2
            dot_ds = dsi[1] * dsj[1] + dsi[2] * dsj[2]
            # 2×2 Gauss-Legendre quadrature over both segments
            quad = zero(T)
            for qi in 1:2
                pi_pt = midi + g_nodes[qi] * half_dsi
                for qj in 1:2
                    pj_pt = midj + g_nodes[qj] * half_dsj
                    dx = pi_pt[1] - pj_pt[1]
                    dy = pi_pt[2] - pj_pt[2]
                    r = sqrt(dx^2 + dy^2)
                    r < eps(T) * Ld && continue
                    quad += g_weight * g_weight * besselk(0, r / Ld)
                end
            end
            local_s += quad / 4 * dot_ds
        end
        partial[i] = local_s
    end
    return sum(partial)
end

function circulation(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv * vortex_area(c)
        end
    end
    return s
end

function enstrophy(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv^2 * vortex_area(c)
        end
    end
    return s / 2
end

function angular_momentum(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    s = zero(T)
    for i in 1:N
        for c in prob.layers[i]
            s += c.pv * _second_moment_r2(c)
        end
    end
    return s
end

function energy(prob::MultiLayerContourProblem{N, K, D, T}) where {N, K, D, T}
    kernel = prob.kernel
    evals = kernel.eigenvalues
    P_inv = kernel.eigenvectors_inv
    E = zero(T)

    for mode in 1:N
        lam = evals[mode]
        for li in 1:N
            wi = P_inv[mode, li]
            abs(wi) < eps(T) && continue
            for lj in 1:N
                wj = P_inv[mode, lj]
                abs(wj) < eps(T) && continue
                for ci in prob.layers[li]
                    nci = nnodes(ci)
                    nci < 2 && continue
                    for cj in prob.layers[lj]
                        ncj = nnodes(cj)
                        ncj < 2 && continue
                        if abs(lam) < eps(T) * 100
                            pair_E = _energy_contour_pair_euler(ci, cj)
                        else
                            Ld_mode = one(T) / sqrt(abs(lam))
                            pair_E = _energy_contour_pair_qg(ci, cj, Ld_mode)
                        end
                        E += wi * wj * ci.pv * cj.pv * pair_E
                    end
                end
            end
        end
    end

    inv4pi = one(T) / (4 * T(π))
    return -inv4pi * E / 2
end
