module ContourDynamicsMakieExt

using ContourDynamics
using Makie

function ContourDynamics.record_evolution(prob::ContourProblem, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    # Include frame 0 (initial state) followed by evolution frames
    frame_indices = vcat([0], collect(frameskip:frameskip:nsteps))

    Makie.record(fig, filename, frame_indices; framerate=30) do frame
        # Evolve only for frames after the initial state
        if frame > 0
            prev_frame = frame - frameskip
            steps_to_take = min(frameskip, nsteps - max(prev_frame, 0))
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
            close_node = nodes[1] + c.wrap  # wrap = (0,0) for closed contours
            xs = [nodes[i][1] for i in 1:n]
            push!(xs, close_node[1])
            ys = [nodes[i][2] for i in 1:n]
            push!(ys, close_node[2])
            Makie.lines!(ax, xs, ys; color=c.pv, colormap=:RdBu)
        end
    end

    return fig
end

function ContourDynamics.record_evolution(prob::MultiLayerContourProblem{N}, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4",
                                          callbacks=nothing) where {N}
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    frame_indices = vcat([0], collect(frameskip:frameskip:nsteps))

    Makie.record(fig, filename, frame_indices; framerate=30) do frame
        if frame > 0
            prev_frame = frame - frameskip
            steps_to_take = min(frameskip, nsteps - max(prev_frame, 0))
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
            for c in layer
                nodes = c.nodes
                n = length(nodes)
                close_node = nodes[1] + c.wrap
                xs = [nodes[i][1] for i in 1:n]
                push!(xs, close_node[1])
                ys = [nodes[i][2] for i in 1:n]
                push!(ys, close_node[2])
                Makie.lines!(ax, xs, ys; color=c.pv, colormap=:RdBu,
                             linestyle=li == 1 ? :solid : :dash)
            end
        end
    end

    return fig
end

end # module
