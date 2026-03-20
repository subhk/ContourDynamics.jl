module ContourDynamicsJLD2Ext

using ContourDynamics
using JLD2
using StaticArrays

"""
    save_snapshot(filename, prob, step; dt=nothing, diagnostics=true)

Save a single snapshot of the simulation state to a JLD2 file.
Each snapshot is stored under a group `step_NNNNNN`.

Saves: contour nodes, PV values, node counts, and optionally
energy, circulation, enstrophy, and angular momentum.
"""
function ContourDynamics.save_snapshot(filename::String,
                                       prob::ContourProblem{K,D,T},
                                       step::Int;
                                       dt::Union{Nothing,T}=nothing,
                                       diagnostics::Bool=true) where {K,D,T}
    group = "step_" * lpad(step, 6, '0')

    jldopen(filename, "a+") do f
        g = JLD2.Group(f, group)

        g["step"] = step
        if dt !== nothing
            g["time"] = step * dt
        end
        g["ncontours"] = length(prob.contours)

        for (ci, c) in enumerate(prob.contours)
            cg = JLD2.Group(g, "contour_" * lpad(ci, 4, '0'))
            # Store as plain arrays for portability
            nodes_x = [c.nodes[i][1] for i in 1:nnodes(c)]
            nodes_y = [c.nodes[i][2] for i in 1:nnodes(c)]
            cg["x"] = nodes_x
            cg["y"] = nodes_y
            cg["pv"] = c.pv
            cg["nnodes"] = nnodes(c)
        end

        if diagnostics
            dg = JLD2.Group(g, "diagnostics")
            dg["energy"] = energy(prob)
            dg["circulation"] = circulation(prob)
            dg["enstrophy"] = enstrophy(prob)
            dg["angular_momentum"] = angular_momentum(prob)
            dg["total_nodes"] = total_nodes(prob)
        end
    end

    return nothing
end

# Multi-layer version
function ContourDynamics.save_snapshot(filename::String,
                                       prob::MultiLayerContourProblem{N,K,D,T},
                                       step::Int;
                                       dt::Union{Nothing,T}=nothing,
                                       diagnostics::Bool=true) where {N,K,D,T}
    group = "step_" * lpad(step, 6, '0')

    jldopen(filename, "a+") do f
        g = JLD2.Group(f, group)

        g["step"] = step
        g["nlayers"] = N
        if dt !== nothing
            g["time"] = step * dt
        end

        for (li, layer) in enumerate(prob.layers)
            lg = JLD2.Group(g, "layer_" * lpad(li, 2, '0'))
            lg["ncontours"] = length(layer)
            for (ci, c) in enumerate(layer)
                cg = JLD2.Group(lg, "contour_" * lpad(ci, 4, '0'))
                cg["x"] = [c.nodes[i][1] for i in 1:nnodes(c)]
                cg["y"] = [c.nodes[i][2] for i in 1:nnodes(c)]
                cg["pv"] = c.pv
                cg["nnodes"] = nnodes(c)
            end
        end

        if diagnostics
            dg = JLD2.Group(g, "diagnostics")
            dg["energy"] = energy(prob)
            dg["circulation"] = circulation(prob)
            dg["enstrophy"] = enstrophy(prob)
            dg["total_nodes"] = total_nodes(prob)
        end
    end

    return nothing
end

"""
    load_snapshot(filename, step) -> (contours, diagnostics)

Load a single snapshot from a JLD2 file.
Returns a named tuple `(contours, diagnostics)` where `contours` is a
`Vector{PVContour{Float64}}` and `diagnostics` is a `NamedTuple`.
"""
function ContourDynamics.load_snapshot(filename::String, step::Int)
    group = "step_" * lpad(step, 6, '0')

    jldopen(filename, "r") do f
        haskey(f, group) || error("Step $step not found in $filename")
        g = f[group]

        nc = g["ncontours"]
        T = Float64
        contours = PVContour{T}[]

        for ci in 1:nc
            cg = g["contour_" * lpad(ci, 4, '0')]
            x = cg["x"]::Vector{T}
            y = cg["y"]::Vector{T}
            pv = cg["pv"]::T
            nodes = [SVector{2,T}(x[i], y[i]) for i in eachindex(x)]
            push!(contours, PVContour(nodes, pv))
        end

        diag = if haskey(g, "diagnostics")
            dg = g["diagnostics"]
            (energy = dg["energy"]::T,
             circulation = dg["circulation"]::T,
             enstrophy = dg["enstrophy"]::T,
             angular_momentum = haskey(dg, "angular_momentum") ? dg["angular_momentum"]::T : zero(T),
             total_nodes = dg["total_nodes"]::Int)
        else
            nothing
        end

        time = haskey(g, "time") ? g["time"] : nothing

        return (contours=contours, diagnostics=diag, step=step, time=time)
    end
end

"""
    load_simulation(filename) -> Vector{NamedTuple}

Load all snapshots from a JLD2 file, sorted by step number.
"""
function ContourDynamics.load_simulation(filename::String)
    snapshots = []
    jldopen(filename, "r") do f
        step_keys = sort(filter(k -> startswith(k, "step_"), keys(f));
                         by = k -> parse(Int, k[6:end]))
        for key in step_keys
            step = parse(Int, key[6:end])
            snap = ContourDynamics.load_snapshot(filename, step)
            push!(snapshots, snap)
        end
    end
    return snapshots
end

"""
    jld2_recorder(filename; save_every=nothing, save_dt=nothing, dt=nothing, diagnostics=true)

Create a callback for `evolve!` that saves snapshots to a JLD2 file.

Specify either:
- `save_every::Int` — save every N iterations
- `save_dt` + `dt` — save every `save_dt` time units (requires the stepper's `dt`)

# Example

```julia
using ContourDynamics, JLD2

recorder = jld2_recorder("output.jld2"; save_every=100)
evolve!(prob, stepper, params; nsteps=10000, callbacks=[recorder])

# Or time-based:
recorder = jld2_recorder("output.jld2"; save_dt=0.5, dt=0.01)
evolve!(prob, stepper, params; nsteps=10000, callbacks=[recorder])
```
"""
function ContourDynamics.jld2_recorder(filename::String;
                                        save_every::Union{Nothing,Int}=nothing,
                                        save_dt=nothing,
                                        dt=nothing,
                                        diagnostics::Bool=true)
    if save_every === nothing && save_dt === nothing
        throw(ArgumentError("Specify either save_every (iterations) or save_dt (time interval)"))
    end
    if save_dt !== nothing && dt === nothing
        throw(ArgumentError("save_dt requires dt (time step size)"))
    end

    # Compute iteration interval from time-based save rate
    interval = if save_every !== nothing
        save_every
    else
        max(1, round(Int, save_dt / dt))
    end

    step_dt = dt  # capture for closure

    return function(prob, step)
        if step % interval == 0 || step == 1
            ContourDynamics.save_snapshot(filename, prob, step;
                                          dt=step_dt, diagnostics=diagnostics)
        end
    end
end

end # module
