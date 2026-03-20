module ContourDynamicsGeophysicalFlowsExt

using ContourDynamics
using GeophysicalFlows
using StaticArrays

function ContourDynamics.contours_from_gridfield(grid_pv::AbstractMatrix{T},
                                                  levels::AbstractVector{T};
                                                  grid=nothing) where {T}
    nx, ny = size(grid_pv)
    contours = PVContour{T}[]

    if grid === nothing
        xs = range(zero(T), one(T), length=nx)
        ys = range(zero(T), one(T), length=ny)
    else
        xs = grid.x
        ys = grid.y
    end

    for (li, level) in enumerate(levels)
        pv_below = li == 1 ? zero(T) : levels[li-1]
        pv_jump = level - pv_below

        iso_nodes = _marching_squares(grid_pv, xs, ys, level)
        for node_set in iso_nodes
            if length(node_set) >= 3
                push!(contours, PVContour(node_set, pv_jump))
            end
        end
    end

    return contours
end

function _marching_squares(field::AbstractMatrix{T}, xs, ys, level::T) where {T}
    nx, ny = size(field)
    contour_sets = Vector{SVector{2,T}}[]
    visited = falses(nx-1, ny-1)

    for i in 1:(nx-1), j in 1:(ny-1)
        visited[i,j] && continue
        v = (field[i,j] >= level, field[i+1,j] >= level,
             field[i+1,j+1] >= level, field[i,j+1] >= level)
        case = v[1] + 2*v[2] + 4*v[3] + 8*v[4]
        (case == 0 || case == 15) && continue

        nodes = _trace_contour(field, xs, ys, level, i, j, visited)
        if length(nodes) >= 3
            push!(contour_sets, nodes)
        end
    end

    return contour_sets
end

function _trace_contour(field::AbstractMatrix{T}, xs, ys, level::T,
                        start_i, start_j, visited) where {T}
    nodes = SVector{2,T}[]
    nx, ny = size(field)
    i, j = start_i, start_j

    for _ in 1:10000
        (i < 1 || i >= nx || j < 1 || j >= ny) && break
        visited[i,j] && length(nodes) > 2 && break
        visited[i,j] = true

        f00, f10, f01 = field[i,j], field[i+1,j], field[i,j+1]
        x0, x1 = xs[i], xs[i+1]
        y0, y1 = ys[j], ys[j+1]

        t_x = (level - f00) / (f10 - f00 + eps(T))
        t_y = (level - f00) / (f01 - f00 + eps(T))
        t_x = clamp(t_x, zero(T), one(T))
        t_y = clamp(t_y, zero(T), one(T))

        push!(nodes, SVector{2,T}(x0 + t_x * (x1 - x0), y0 + t_y * (y1 - y0)))

        moved = false
        for (di, dj) in ((1,0), (0,1), (-1,0), (0,-1))
            ni, nj = i + di, j + dj
            (ni < 1 || ni >= nx || nj < 1 || nj >= ny) && continue
            visited[ni,nj] && continue
            f_check = field[ni,nj]
            f_check2 = field[ni+1,nj+1]
            if (f_check - level) * (f_check2 - level) <= 0
                i, j = ni, nj
                moved = true
                break
            end
        end
        moved || break
    end

    return nodes
end

function ContourDynamics.gridfield_from_contours(prob::ContourProblem{K,D,T},
                                                  nx::Int, ny::Int;
                                                  xlims=nothing, ylims=nothing) where {K,D,T}
    # Compute bounding box from contour geometry with 10% padding
    if xlims === nothing || ylims === nothing
        has_nodes = false
        xmin = zero(T); xmax = zero(T)
        ymin = zero(T); ymax = zero(T)
        for c in prob.contours
            for node in c.nodes
                if !has_nodes
                    xmin = xmax = node[1]
                    ymin = ymax = node[2]
                    has_nodes = true
                else
                    xmin = min(xmin, node[1])
                    xmax = max(xmax, node[1])
                    ymin = min(ymin, node[2])
                    ymax = max(ymax, node[2])
                end
            end
        end
        if !has_nodes
            # No contours — default to unit square
            xmin, xmax = -one(T), one(T)
            ymin, ymax = -one(T), one(T)
        end
        pad_x = max(T(0.1) * (xmax - xmin), eps(T))
        pad_y = max(T(0.1) * (ymax - ymin), eps(T))
        if xlims === nothing
            xlims = (xmin - pad_x, xmax + pad_x)
        end
        if ylims === nothing
            ylims = (ymin - pad_y, ymax + pad_y)
        end
    end

    field = zeros(T, nx, ny)
    xs = range(T(xlims[1]), T(xlims[2]), length=nx)
    ys = range(T(ylims[1]), T(ylims[2]), length=ny)

    for c in prob.contours
        for (ix, x) in enumerate(xs)
            for (iy, y) in enumerate(ys)
                if _point_in_polygon(SVector{2,T}(x, y), c.nodes)
                    field[ix, iy] += c.pv
                end
            end
        end
    end
    return field
end

function _point_in_polygon(p::SVector{2,T}, nodes::Vector{SVector{2,T}}) where {T}
    n = length(nodes)
    inside = false
    j = n
    for i in 1:n
        if ((nodes[i][2] > p[2]) != (nodes[j][2] > p[2])) &&
            (p[1] < (nodes[j][1] - nodes[i][1]) * (p[2] - nodes[i][2]) /
                     (nodes[j][2] - nodes[i][2]) + nodes[i][1])
            inside = !inside
        end
        j = i
    end
    return inside
end

end # module
