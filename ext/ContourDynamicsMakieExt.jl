# Makie visualization extension.
#
# The recording functions advance the simulation between requested frame
# indices and redraw the current contour geometry. They intentionally operate on
# the passed problem in-place, matching `evolve!` semantics.
module ContourDynamicsMakieExt

using ContourDynamics
using Makie

# Colour range for the diverging :RdBu map, centred on zero so each PV sign
# keeps its hue and a single PV level is drawn at a saturated end of the map
# rather than at its near-white midpoint.
function _pv_colorrange(pv_vals)
    m = isempty(pv_vals) ? 0.0 : Float64(maximum(abs, pv_vals))
    m > 0 || (m = 1.0)
    return (-m, m)
end

# Frame 0 (initial state), every `frameskip`-th step, and always the final
# state so output videos document the exact requested integration interval.
function _frame_schedule(nsteps::Int, frameskip::Int)
    nsteps >= 0 || throw(ArgumentError("nsteps must be non-negative, got $nsteps"))
    frameskip > 0 || throw(ArgumentError("frameskip must be positive, got $frameskip"))
    frame_indices = vcat([0], collect(frameskip:frameskip:nsteps))
    if frame_indices[end] != nsteps
        push!(frame_indices, nsteps)
    end
    return frame_indices
end

# Advance `prob` in place from the last evolved step (`evolved[]`) to `frame`.
function _evolve_to_frame!(prob, stepper, params, callbacks, evolved::Ref{Int}, frame::Int)
    # Evolve only for frames after the initial state
    frame > 0 || return nothing
    step_offset = evolved[]
    steps_to_take = frame - step_offset
    if steps_to_take > 0
        evolve!(prob, stepper, params; nsteps=steps_to_take, callbacks=callbacks,
                step_offset=step_offset,
                run_initial_callbacks=iszero(step_offset))
        evolved[] = frame
    end
    return nothing
end

# Convert a contour's nodes to plottable coordinate vectors, or `nothing` when
# the contour is empty. Spanning contours represent periodic interfaces and
# should not be closed visually; ordinary patches repeat the first node at the end.
function _contour_line(c)
    nodes = c.nodes
    n = length(nodes)
    n == 0 && return nothing
    spanning = ContourDynamics.is_spanning(c)
    n_pts = spanning ? n : n + 1
    xs = Vector{Float64}(undef, n_pts)
    ys = Vector{Float64}(undef, n_pts)
    for i in 1:n
        xs[i] = nodes[i][1]
        ys[i] = nodes[i][2]
    end
    if !spanning
        xs[n+1] = nodes[1][1]
        ys[n+1] = nodes[1][2]
    end
    return xs, ys
end

# Record one frame per entry of `frame_indices`: advance the problem, clear the
# axis, let `draw!(ax)` redraw the current geometry, then refit the limits.
function _record_frames(draw!, fig, ax, filename, frame_indices, prob, stepper, params, callbacks)
    evolved = Ref(0)
    Makie.record(fig, filename, frame_indices; framerate=30) do frame
        _evolve_to_frame!(prob, stepper, params, callbacks, evolved, frame)
        Makie.empty!(ax)
        draw!(ax)
        # Fit the axis to this frame. Backends that render off screen (e.g.
        # CairoMakie) do not refit limits after `empty!`, which otherwise
        # leaves every frame at the empty axis's default box.
        Makie.reset_limits!(ax)
    end
    return fig
end

"""
    record_evolution(prob::ContourProblem, stepper, params; nsteps, frameskip=10, filename="contour_evolution.mp4", callbacks=nothing)

Record a single-layer contour simulation to a Makie video file while advancing
`prob` in place. The initial state and final requested step are always included,
even when `nsteps` is not an exact multiple of `frameskip`.
"""
function ContourDynamics.record_evolution(prob::ContourProblem, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing)
    frame_indices = _frame_schedule(nsteps, frameskip)

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    # Fix colorrange from initial PV values so colors are consistent across frames.
    pv_lo, pv_hi = _pv_colorrange([c.pv for c in snapshot_contours(prob)])

    return _record_frames(fig, ax, filename, frame_indices, prob, stepper, params, callbacks) do ax
        for c in snapshot_contours(prob)
            line = _contour_line(c)
            line === nothing && continue
            xs, ys = line
            Makie.lines!(ax, xs, ys; color=c.pv, colormap=:RdBu,
                         colorrange=(pv_lo, pv_hi))
        end
    end
end

"""
    record_evolution(prob::MultiLayerContourProblem, stepper, params; nsteps, frameskip=10, filename="contour_evolution.mp4", callbacks=nothing)

Record a multi-layer contour simulation to a Makie video file. Layers share the
same PV colormap and are distinguished by line style.
"""
function ContourDynamics.record_evolution(prob::MultiLayerContourProblem{N}, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing) where {N}
    frame_indices = _frame_schedule(nsteps, frameskip)

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    # Fix colorrange from initial PV values across all layers.
    pv_lo, pv_hi = _pv_colorrange([c.pv for layer in snapshot_contours(prob) for c in layer])

    # Distinct line styles make layer identity visible even when PV colors
    # overlap or are identical across layers.
    layer_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    return _record_frames(fig, ax, filename, frame_indices, prob, stepper, params, callbacks) do ax
        for (li, layer) in enumerate(snapshot_contours(prob))
            style = layer_styles[mod1(li, length(layer_styles))]
            first_in_layer = true
            for c in layer
                line = _contour_line(c)
                line === nothing && continue
                xs, ys = line
                Makie.lines!(ax, xs, ys; color=c.pv, colormap=:RdBu,
                             colorrange=(pv_lo, pv_hi), linestyle=style,
                             label=first_in_layer ? "Layer $li" : nothing)
                first_in_layer = false
            end
        end
    end
end

# Forwarder so the high-level `Problem` wrapper can be recorded directly.
ContourDynamics.record_evolution(prob::ContourDynamics.Problem, stepper, params; kwargs...) =
    ContourDynamics.record_evolution(prob.contour_problem, stepper, params; kwargs...)

end # module
