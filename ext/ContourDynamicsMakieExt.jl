module ContourDynamicsMakieExt

using ContourDynamics
using Makie

function ContourDynamics.record_evolution(prob::ContourProblem, stepper, params;
                                          nsteps::Int, frameskip::Int=10,
                                          filename="contour_evolution.mp4")
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; aspect=Makie.DataAspect())

    Makie.record(fig, filename, 1:frameskip:nsteps; framerate=30) do frame
        steps_to_take = min(frameskip, nsteps - (frame - 1))
        evolve!(prob, stepper, params; nsteps=steps_to_take)
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

end # module
