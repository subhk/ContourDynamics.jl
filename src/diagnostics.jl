"""
    vortex_area(c::PVContour)

Signed area enclosed by contour `c` using the shoelace formula.
"""
function vortex_area(c::PVContour{T}) where {T}
    nodes = c.nodes
    n = length(nodes)
    n < 3 && return zero(T)
    A = zero(T)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        A += nodes[i][1] * nodes[j][2] - nodes[j][1] * nodes[i][2]
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
        j = mod1(i + 1, n)
        cross = nodes[i][1] * nodes[j][2] - nodes[j][1] * nodes[i][2]
        cx += (nodes[i][1] + nodes[j][1]) * cross
        cy += (nodes[i][2] + nodes[j][2]) * cross
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
        j = mod1(i + 1, n)
        xi, yi = nodes[i][1] - ctr[1], nodes[i][2] - ctr[2]
        xj, yj = nodes[j][1] - ctr[1], nodes[j][2] - ctr[2]
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

function circulation(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv * vortex_area(c)
    end
    return s
end

function enstrophy(prob::ContourProblem{K, D, T}) where {K, D, T}
    s = zero(T)
    for c in prob.contours
        s += c.pv^2 * vortex_area(c)
    end
    return s / 2
end

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
    s = zero(T)
    @inbounds for i in 1:n
        j = mod1(i + 1, n)
        xi, yi = nodes[i][1], nodes[i][2]
        xj, yj = nodes[j][1], nodes[j][2]
        cross = xi * yj - xj * yi
        s += (xi^2 + xi * xj + xj^2) * cross
        s += (yi^2 + yi * yj + yj^2) * cross
    end
    return s / 12
end

function energy(prob::ContourProblem{EulerKernel, D, T}) where {D, T}
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
    return -inv4pi * E
end

function _energy_contour_pair_euler(ci::PVContour{T}, cj::PVContour{T}) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    # Thread over outer segments, each thread accumulates a partial sum
    partial = zeros(T, nci)
    @inbounds Threads.@threads for i in 1:nci
        i_next = mod1(i + 1, nci)
        dxi = ci.nodes[i_next][1] - ci.nodes[i][1]
        dyi = ci.nodes[i_next][2] - ci.nodes[i][2]
        mx_i = (ci.nodes[i][1] + ci.nodes[i_next][1]) / 2
        my_i = (ci.nodes[i][2] + ci.nodes[i_next][2]) / 2
        local_s = zero(T)
        for j in 1:ncj
            j_next = mod1(j + 1, ncj)
            dxj = cj.nodes[j_next][1] - cj.nodes[j][1]
            dyj = cj.nodes[j_next][2] - cj.nodes[j][2]
            mx_j = (cj.nodes[j][1] + cj.nodes[j_next][1]) / 2
            my_j = (cj.nodes[j][2] + cj.nodes[j_next][2]) / 2
            dx = mx_i - mx_j
            dy = my_i - my_j
            r2 = dx^2 + dy^2
            r2 < eps(T) && continue
            dot_ds = dxi * dxj + dyi * dyj
            local_s += log(sqrt(r2)) * dot_ds
        end
        partial[i] = local_s
    end
    return sum(partial)
end

function energy(prob::ContourProblem{QGKernel{T}, D, T}) where {D, T}
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
    return -inv4pi * E
end

function _energy_contour_pair_qg(ci::PVContour{T}, cj::PVContour{T}, Ld::T) where {T}
    nci = nnodes(ci)
    ncj = nnodes(cj)
    partial = zeros(T, nci)
    @inbounds Threads.@threads for i in 1:nci
        i_next = mod1(i + 1, nci)
        dxi = ci.nodes[i_next][1] - ci.nodes[i][1]
        dyi = ci.nodes[i_next][2] - ci.nodes[i][2]
        mx_i = (ci.nodes[i][1] + ci.nodes[i_next][1]) / 2
        my_i = (ci.nodes[i][2] + ci.nodes[i_next][2]) / 2
        local_s = zero(T)
        for j in 1:ncj
            j_next = mod1(j + 1, ncj)
            dxj = cj.nodes[j_next][1] - cj.nodes[j][1]
            dyj = cj.nodes[j_next][2] - cj.nodes[j][2]
            mx_j = (cj.nodes[j][1] + cj.nodes[j_next][1]) / 2
            my_j = (cj.nodes[j][2] + cj.nodes[j_next][2]) / 2
            dx = mx_i - mx_j
            dy = my_i - my_j
            r = sqrt(dx^2 + dy^2)
            r < eps(T) * Ld && continue
            dot_ds = dxi * dxj + dyi * dyj
            local_s += besselk(0, r / Ld) * dot_ds
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
    return -inv4pi * E
end
