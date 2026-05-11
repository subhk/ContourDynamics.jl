function _contour_curve(contour)
    nodes = contour.nodes
    n = length(nodes)
    n == 0 && return Float64[], Float64[]

    xs = Vector{Float64}(undef, n + 1)
    ys = Vector{Float64}(undef, n + 1)
    for i in 1:n
        xs[i] = nodes[i][1]
        ys[i] = nodes[i][2]
    end

    endpoint = is_spanning(contour) ? next_node(contour, n) : nodes[1]
    xs[end] = endpoint[1]
    ys[end] = endpoint[2]
    return xs, ys
end

function _periodic_coordinate(x, lo, hi)
    width = hi - lo
    width > 0 || throw(ArgumentError("periodic bounds must have positive width"))

    y = mod(x - lo, width) + lo
    if isapprox(y, lo; atol=100eps(Float64) * max(1.0, abs(width))) && x > lo
        return hi
    end
    return y
end

function _periodic_curve(xs, ys, periodic_box)
    length(xs) == length(ys) || throw(DimensionMismatch("xs and ys must have the same length"))
    isempty(xs) && return Float64[], Float64[]

    xmin, xmax, ymin, ymax = periodic_box
    jump_x = 0.5 * (xmax - xmin)
    jump_y = 0.5 * (ymax - ymin)

    out_x = Float64[]
    out_y = Float64[]
    sizehint!(out_x, length(xs) + 4)
    sizehint!(out_y, length(ys) + 4)

    prev_x = _periodic_coordinate(xs[1], xmin, xmax)
    prev_y = _periodic_coordinate(ys[1], ymin, ymax)
    push!(out_x, prev_x)
    push!(out_y, prev_y)

    for i in 2:length(xs)
        x = _periodic_coordinate(xs[i], xmin, xmax)
        y = _periodic_coordinate(ys[i], ymin, ymax)
        if abs(x - prev_x) > jump_x || abs(y - prev_y) > jump_y
            push!(out_x, NaN)
            push!(out_y, NaN)
        end
        push!(out_x, x)
        push!(out_y, y)
        prev_x = x
        prev_y = y
    end

    return out_x, out_y
end

function _point_like(reference, x, y)
    try
        return typeof(reference)(x, y)
    catch
        return (x, y)
    end
end

function _periodic_delta(prev, current, periodic_box)
    xmin, xmax, ymin, ymax = periodic_box
    width_x = xmax - xmin
    width_y = ymax - ymin
    width_x > 0 || throw(ArgumentError("periodic x bounds must have positive width"))
    width_y > 0 || throw(ArgumentError("periodic y bounds must have positive width"))

    dx = current[1] - prev[1]
    dy = current[2] - prev[2]
    dx -= width_x * round(dx / width_x)
    dy -= width_y * round(dy / width_y)
    return _point_like(current - prev, dx, dy)
end

function _unwrap_periodic_points(points, periodic_box)
    isempty(points) && return copy(points)

    out = Vector{typeof(first(points))}(undef, length(points))
    out[1] = first(points)
    for i in 2:length(points)
        out[i] = out[i - 1] + _periodic_delta(points[i - 1], points[i], periodic_box)
    end
    return out
end
