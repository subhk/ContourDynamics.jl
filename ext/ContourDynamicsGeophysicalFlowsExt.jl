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
                c = PVContour(node_set, pv_jump)
                # Enforce CCW orientation (positive area); the PV sign
                # lives in c.pv, not in the winding direction.
                if vortex_area(c) < zero(T)
                    reverse!(c.nodes)
                end
                push!(contours, c)
            end
        end
    end

    return contours
end

function _marching_squares(field::AbstractMatrix{T}, xs, ys, level::T) where {T}
    nx, ny = size(field)
    contour_sets = Vector{SVector{2,T}}[]
    # Track which edges have been used. Edges are indexed by cell (i,j) and side:
    #   side 1 = bottom (y=ys[j]),  side 2 = right (x=xs[i+1]),
    #   side 3 = top    (y=ys[j+1]), side 4 = left  (x=xs[i])
    visited_edges = falses(nx-1, ny-1, 4)

    for i in 1:(nx-1), j in 1:(ny-1)
        case = _cell_case(field, i, j, level)
        (case == 0 || case == 15) && continue
        # Try tracing from each edge of this cell
        for side in 1:4
            visited_edges[i, j, side] && continue
            _has_crossing(case, side) || continue
            nodes = _trace_contour(field, xs, ys, level, i, j, side, visited_edges)
            length(nodes) >= 3 && push!(contour_sets, nodes)
        end
    end

    return contour_sets
end

# Cell corner layout:  (i,j+1)=f01 --- (i+1,j+1)=f11
#                         |                |
#                      (i,j)=f00  ---  (i+1,j)=f10
# Case = bit0*f00_above + bit1*f10_above + bit2*f11_above + bit3*f01_above
@inline function _cell_case(field, i, j, level)
    return Int(field[i,j] >= level) +
           2 * Int(field[i+1,j] >= level) +
           4 * Int(field[i+1,j+1] >= level) +
           8 * Int(field[i,j+1] >= level)
end

# Which edges does a given marching-squares case cross?
# Edges: 1=bottom, 2=right, 3=top, 4=left
@inline function _has_crossing(case::Int, side::Int)
    # Lookup table: for each case, which edges are crossed.
    # An edge crosses when its two corner values differ (one above, one below level).
    # Edge 1=bottom (f00↔f10), 2=right (f10↔f11), 3=top (f01↔f11), 4=left (f00↔f01).
    # Complement check: case N and case 15-N must cross the same edges.
    #  case  edges    case  edges
    #  1     1,4      14    1,4
    #  2     1,2      13    1,2
    #  3     2,4      12    2,4
    #  4     2,3      11    2,3
    #  5     1,2,3,4  10    1,2,3,4
    #  6     1,3       9    1,3
    #  7     3,4       8    3,4
    crossing = (
        (true,false,false,true),   # 1:  1,4
        (true,true,false,false),   # 2:  1,2
        (false,true,false,true),   # 3:  2,4
        (false,true,true,false),   # 4:  2,3
        (true,true,true,true),     # 5:  saddle (1,2,3,4)
        (true,false,true,false),   # 6:  1,3
        (false,false,true,true),   # 7:  3,4
        (false,false,true,true),   # 8:  3,4
        (true,false,true,false),   # 9:  1,3
        (true,true,true,true),     # 10: saddle (1,2,3,4)
        (false,true,true,false),   # 11: 2,3
        (false,true,false,true),   # 12: 2,4
        (true,true,false,false),   # 13: 1,2
        (true,false,false,true),   # 14: 1,4
    )
    (case < 1 || case > 14) && return false
    return crossing[case][side]
end

@inline function _case_edges(case::Int)
    # Returns the pair(s) of edges crossed for each case
    table = (
        (1,4), (1,2), (2,4), (2,3), (1,2,3,4), (1,3), (3,4),  # 1-7
        (3,4), (1,3), (1,2,3,4), (2,3), (2,4), (1,2), (1,4),  # 8-14
    )
    return table[case]
end

# Neighbor cell when crossing a given edge
@inline function _neighbor(i, j, side)
    side == 1 && return (i, j-1, 3)   # cross bottom → neighbor's top
    side == 2 && return (i+1, j, 4)   # cross right  → neighbor's left
    side == 3 && return (i, j+1, 1)   # cross top    → neighbor's bottom
    return (i-1, j, 2)                # cross left   → neighbor's right
end

function _trace_contour(field::AbstractMatrix{T}, xs, ys, level::T,
                        start_i, start_j, start_side, visited_edges) where {T}
    nodes = SVector{2,T}[]
    nx, ny = size(field)

    # Shift field values by level so we interpolate against zero
    # (done inline in _edge_point_level below)

    i, j, entry_side = start_i, start_j, start_side

    for _ in 1:10000
        (i < 1 || i >= nx || j < 1 || j >= ny) && break

        case = _cell_case(field, i, j, level)
        (case == 0 || case == 15) && break

        # Record the crossing point where we enter this cell
        visited_edges[i, j, entry_side] && length(nodes) > 2 && break
        visited_edges[i, j, entry_side] = true
        push!(nodes, _edge_point_level(T, field, xs, ys, i, j, entry_side, level))

        # Determine exit side
        edges = _case_edges(case)
        if length(edges) == 2
            exit_side = edges[1] == entry_side ? edges[2] : edges[1]
        else
            # Saddle case (5 or 10): disambiguate using center value
            center = (field[i,j] + field[i+1,j] + field[i+1,j+1] + field[i,j+1]) / 4
            if center >= level
                # Connect 1↔2 and 3↔4 for case 5; 1↔4 and 2↔3 for case 10
                if case == 5
                    pairs = ((1,2), (3,4))
                else
                    pairs = ((1,4), (2,3))
                end
            else
                if case == 5
                    pairs = ((1,4), (2,3))
                else
                    pairs = ((1,2), (3,4))
                end
            end
            exit_side = 0
            for (a, b) in pairs
                a == entry_side && (exit_side = b; break)
                b == entry_side && (exit_side = a; break)
            end
            exit_side == 0 && break
        end

        visited_edges[i, j, exit_side] = true

        # Move to neighbor through exit edge
        ni, nj, neighbor_entry = _neighbor(i, j, exit_side)
        i, j, entry_side = ni, nj, neighbor_entry
    end

    return nodes
end

@inline function _edge_point_level(::Type{T}, field, xs, ys, i, j, side, level) where {T}
    f00, f10 = field[i,j] - level, field[i+1,j] - level
    f11, f01 = field[i+1,j+1] - level, field[i,j+1] - level
    x0, x1 = T(xs[i]), T(xs[i+1])
    y0, y1 = T(ys[j]), T(ys[j+1])
    if side == 1      # bottom: y=y0, interp x between f00 and f10
        t = _safe_frac(T, -f00, f10 - f00)
        return SVector{2,T}(x0 + t * (x1 - x0), y0)
    elseif side == 2  # right: x=x1, interp y between f10 and f11
        t = _safe_frac(T, -f10, f11 - f10)
        return SVector{2,T}(x1, y0 + t * (y1 - y0))
    elseif side == 3  # top: y=y1, interp x between f01 and f11
        t = _safe_frac(T, -f01, f11 - f01)
        return SVector{2,T}(x0 + t * (x1 - x0), y1)
    else              # left: x=x0, interp y between f00 and f01
        t = _safe_frac(T, -f00, f01 - f00)
        return SVector{2,T}(x0, y0 + t * (y1 - y0))
    end
end

@inline function _safe_frac(::Type{T}, num, denom) where {T}
    abs(denom) < eps(T) && return T(0.5)
    return clamp(num / denom, zero(T), one(T))
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
        if is_spanning(c)
            # Spanning contours represent a PV jump across a horizontal band.
            # Add PV to all grid points below the contour's y-level.
            # (For a beta staircase, this reconstructs βy via cumulative jumps.)
            y_level = sum(node[2] for node in c.nodes) / length(c.nodes)
            for (ix, _) in enumerate(xs)
                for (iy, y) in enumerate(ys)
                    if y < y_level
                        field[ix, iy] += c.pv
                    end
                end
            end
        else
            for (ix, x) in enumerate(xs)
                for (iy, y) in enumerate(ys)
                    if _point_in_polygon(SVector{2,T}(x, y), c.nodes)
                        field[ix, iy] += c.pv
                    end
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
