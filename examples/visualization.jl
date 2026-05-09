using ContourDynamics
import CairoMakie

# Shared plotting utilities for example scripts that write JLD2 snapshots.
# Snapshot records are intentionally handled structurally: single-layer files
# contain `contours`, while multi-layer files contain `layers`.

const _LAYER_LINESTYLES = (:solid, :dash, :dot, :dashdot, :dashdotdot)

function _foreach_snapshot_contour(f, snapshot)
    # Invoke `f(contour, layer_idx)` for both single-layer and multi-layer
    # snapshot layouts so drawing code can stay layout-agnostic.
    if hasproperty(snapshot, :contours)
        for contour in snapshot.contours
            f(contour, 1)
        end
    elseif hasproperty(snapshot, :layers)
        for (layer_idx, layer) in enumerate(snapshot.layers)
            for contour in layer
                f(contour, layer_idx)
            end
        end
    else
        throw(ArgumentError("Snapshot does not contain `contours` or `layers`."))
    end
    return nothing
end

function _snapshot_limits(snapshots)
    # Compute a square data window over all frames. A fixed window prevents
    # videos from visually zooming as vortices merge, drift, or shed filaments.
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf
    for snapshot in snapshots
        _foreach_snapshot_contour(snapshot) do contour, _
            for node in contour.nodes
                x, y = node
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
            end
        end
    end
    if !isfinite(xmin)
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
    end
    dx = xmax - xmin
    dy = ymax - ymin
    pad_x = max(0.08 * dx, 0.05)
    pad_y = max(0.08 * dy, 0.05)
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2
    half_span = max(xmax - xmin, ymax - ymin) / 2
    return (mid_x - half_span, mid_x + half_span,
            mid_y - half_span, mid_y + half_span)
end

function _closed_curve(contour)
    # Closed vortex patches repeat the first point for line rendering. Spanning
    # contours represent periodic interfaces, so drawing the closing edge would
    # incorrectly connect across the plot.
    nodes = contour.nodes
    n = length(nodes)
    n == 0 && return Float64[], Float64[]
    npts = is_spanning(contour) ? n : n + 1
    xs = Vector{Float64}(undef, npts)
    ys = Vector{Float64}(undef, npts)
    for i in 1:n
        xs[i] = nodes[i][1]
        ys[i] = nodes[i][2]
    end
    if !is_spanning(contour)
        xs[end] = nodes[1][1]
        ys[end] = nodes[1][2]
    end
    return xs, ys
end

function _draw_snapshot!(cm, ax, snapshot; linewidth, fillalpha)
    # Multi-layer snapshots reuse line style for layer identity; PV color is not
    # used here because these example exports are intended as simple black-line
    # validation artifacts.
    _foreach_snapshot_contour(snapshot) do contour, layer_idx
        xs, ys = _closed_curve(contour)
        isempty(xs) && return
        linestyle = _LAYER_LINESTYLES[mod1(layer_idx, length(_LAYER_LINESTYLES))]
        if fillalpha > 0 && !is_spanning(contour)
            cm.poly!(ax, cm.Point2f.(xs, ys); color=(:black, fillalpha), strokewidth=0)
        end
        cm.lines!(ax, xs, ys;
                  color=:black,
                  linestyle=linestyle,
                  linewidth=is_spanning(contour) ? max(1.5, 0.7 * linewidth) : linewidth)
    end
    return nothing
end

function _style_axis!(cm, ax, limits, title)
    # Keep all output formats visually consistent: final PNG/SVG and every
    # animation frame share the same limits, aspect ratio, and labels.
    xmin, xmax, ymin, ymax = limits
    cm.xlims!(ax, xmin, xmax)
    cm.ylims!(ax, ymin, ymax)
    ax.title = title
    ax.xlabel = "x"
    ax.ylabel = "y"
    return nothing
end

function save_animation(basename::AbstractString, snapshots;
                        title::AbstractString=basename,
                        figure_size=(1600, 1200),
                        linewidth::Real=3.0,
                        fillalpha::Real=0.0,
                        framerate::Int=30,
                        px_per_unit::Real=2)
    # Produce both static final-state figures and an MP4 from the same drawing
    # path so example outputs remain comparable.
    isempty(snapshots) && error("No snapshots available for media export.")

    cm = CairoMakie
    limits = _snapshot_limits(snapshots)

    fig = cm.Figure(size=figure_size, backgroundcolor=:white, fontsize=24)
    ax = cm.Axis(fig[1, 1]; aspect=cm.DataAspect())
    _style_axis!(cm, ax, limits, title)
    _draw_snapshot!(cm, ax, snapshots[end]; linewidth, fillalpha)

    pngfile = basename * "_final.png"
    svgfile = basename * "_final.svg"
    mp4file = basename * ".mp4"

    cm.save(pngfile, fig; px_per_unit=px_per_unit)
    cm.save(svgfile, fig)
    println("Saved final-state figures: $pngfile, $svgfile")

    cm.record(fig, mp4file, eachindex(snapshots); framerate=framerate) do i
        cm.empty!(ax)
        _style_axis!(cm, ax, limits, "$(title) (t = $(round(snapshots[i].time; digits=3)))")
        _draw_snapshot!(cm, ax, snapshots[i]; linewidth, fillalpha)
    end
    println("Saved animation: $mp4file")

    return nothing
end
