module ContourDynamicsMakieExt

using ContourDynamics
using Makie

function ContourDynamics.record_evolution(prob::ContourProblem, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing)
    frameskip > 0 || throw(ArgumentError("frameskip must be positive, got $frameskip"))

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    # Fix colorrange from initial PV values so colors are consistent across frames.
    pv_vals = [c.pv for c in prob.contours]
    pv_lo, pv_hi = isempty(pv_vals) ? (-1.0, 1.0) : (minimum(pv_vals), maximum(pv_vals))
    if pv_lo == pv_hi
        pv_lo -= one(pv_lo)
        pv_hi += one(pv_hi)
    end

    # Include frame 0 (initial state), intermediate frames, and always the final state
    frame_indices = vcat([0], collect(frameskip:frameskip:nsteps))
    if frame_indices[end] != nsteps
        push!(frame_indices, nsteps)
    end
    evolved = Ref(0)

    # Pre-allocate reusable coordinate buffers to reduce GC pressure
    xs_buf = Float64[]
    ys_buf = Float64[]

    Makie.record(fig, filename, frame_indices; framerate=30) do frame
        # Evolve only for frames after the initial state
        if frame > 0
            steps_to_take = frame - evolved[]
            evolved[] = frame
            if steps_to_take > 0
                if callbacks !== nothing
                    evolve!(prob, stepper, params; nsteps=steps_to_take, callbacks=callbacks)
                else
                    evolve!(prob, stepper, params; nsteps=steps_to_take)
                end
            end
        end
        Makie.empty!(ax)
        for c in prob.contours
            nodes = c.nodes
            n = length(nodes)
            n_pts = ContourDynamics.is_spanning(c) ? n : n + 1
            resize!(xs_buf, n_pts)
            resize!(ys_buf, n_pts)
            for i in 1:n
                xs_buf[i] = nodes[i][1]
                ys_buf[i] = nodes[i][2]
            end
            if !ContourDynamics.is_spanning(c)
                xs_buf[n+1] = nodes[1][1]
                ys_buf[n+1] = nodes[1][2]
            end
            Makie.lines!(ax, copy(xs_buf), copy(ys_buf); color=c.pv, colormap=:RdBu,
                         colorrange=(pv_lo, pv_hi))
        end
    end

    return fig
end

function ContourDynamics.record_evolution(prob::MultiLayerContourProblem{N}, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing) where {N}
    frameskip > 0 || throw(ArgumentError("frameskip must be positive, got $frameskip"))

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    # Fix colorrange from initial PV values across all layers.
    pv_vals = [c.pv for layer in prob.layers for c in layer]
    pv_lo, pv_hi = isempty(pv_vals) ? (-1.0, 1.0) : (minimum(pv_vals), maximum(pv_vals))
    if pv_lo == pv_hi
        pv_lo -= one(pv_lo)
        pv_hi += one(pv_hi)
    end

    # Distinct line styles for each layer
    layer_styles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

    frame_indices = vcat([0], collect(frameskip:frameskip:nsteps))
    if frame_indices[end] != nsteps
        push!(frame_indices, nsteps)
    end
    evolved = Ref(0)

    xs_buf = Float64[]
    ys_buf = Float64[]

    Makie.record(fig, filename, frame_indices; framerate=30) do frame
        if frame > 0
            steps_to_take = frame - evolved[]
            evolved[] = frame
            if steps_to_take > 0
                if callbacks !== nothing
                    evolve!(prob, stepper, params; nsteps=steps_to_take, callbacks=callbacks)
                else
                    evolve!(prob, stepper, params; nsteps=steps_to_take)
                end
            end
        end
        Makie.empty!(ax)
        for (li, layer) in enumerate(prob.layers)
            style = layer_styles[mod1(li, length(layer_styles))]
            for c in layer
                nodes = c.nodes
                n = length(nodes)
                n_pts = ContourDynamics.is_spanning(c) ? n : n + 1
                resize!(xs_buf, n_pts)
                resize!(ys_buf, n_pts)
                for i in 1:n
                    xs_buf[i] = nodes[i][1]
                    ys_buf[i] = nodes[i][2]
                end
                if !ContourDynamics.is_spanning(c)
                    xs_buf[n+1] = nodes[1][1]
                    ys_buf[n+1] = nodes[1][2]
                end
                Makie.lines!(ax, copy(xs_buf), copy(ys_buf); color=c.pv, colormap=:RdBu,
                             colorrange=(pv_lo, pv_hi), linestyle=style,
                             label=li <= N ? "Layer $li" : nothing)
            end
        end
    end

    return fig
end

end # module
