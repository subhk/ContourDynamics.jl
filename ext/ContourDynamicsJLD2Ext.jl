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
            cg["wrap_x"] = c.wrap[1]
            cg["wrap_y"] = c.wrap[2]
        end

        if diagnostics
            dg = JLD2.Group(g, "diagnostics")
            dg["circulation"] = circulation(prob)
            dg["enstrophy"] = enstrophy(prob)
            dg["total_nodes"] = total_nodes(prob)
            try
                dg["energy"] = energy(prob)
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
            end
            try
                dg["angular_momentum"] = angular_momentum(prob)
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
            end
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
                cg["wrap_x"] = c.wrap[1]
                cg["wrap_y"] = c.wrap[2]
            end
        end

        if diagnostics
            dg = JLD2.Group(g, "diagnostics")
            dg["circulation"] = circulation(prob)
            dg["enstrophy"] = enstrophy(prob)
            dg["total_nodes"] = total_nodes(prob)
            try
                dg["energy"] = energy(prob)
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
            end
            try
                dg["angular_momentum"] = angular_momentum(prob)
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
            end
        end
    end

    return nothing
end

"""
    load_snapshot(filename, step) -> NamedTuple

Load a single snapshot from a JLD2 file.

For single-layer files the result contains a `contours` field
(`Vector{PVContour}`).  For multi-layer files it contains a `layers` field
(`Tuple` of `Vector{PVContour}` per layer).  Both formats include
`diagnostics`, `step`, and `time`.
"""
function ContourDynamics.load_snapshot(filename::String, step::Int)
    group = "step_" * lpad(step, 6, '0')
    jldopen(filename, "r") do f
        haskey(f, group) || error("Step $step not found in $filename")
        _load_snapshot_from_group(f[group], step)
    end
end

function _load_snapshot_from_group(g, step::Int)
    time = haskey(g, "time") ? g["time"] : nothing
    is_multilayer = haskey(g, "nlayers")

    if is_multilayer
        nlyr = g["nlayers"]::Int
        all_layers = Vector{Any}(undef, nlyr)
        for li in 1:nlyr
            lg = g["layer_" * lpad(li, 2, '0')]
            nc = lg["ncontours"]::Int
            all_layers[li] = _load_contours(lg, nc)
        end
        diag = _load_diagnostics(g)
        return (layers=Tuple(all_layers), diagnostics=diag, step=step, time=time)
    else
        nc = g["ncontours"]::Int
        contours = _load_contours(g, nc)
        diag = _load_diagnostics(g)
        return (contours=contours, diagnostics=diag, step=step, time=time)
    end
end

# ── helpers ──────────────────────────────────────────────────

function _load_contours(g, nc::Int)
    nc == 0 && return PVContour{Float64}[]

    # Peek at first contour to determine element type
    cg1 = g["contour_" * lpad(1, 4, '0')]
    T = eltype(cg1["x"])

    contours = PVContour{T}[]
    for ci in 1:nc
        cg = g["contour_" * lpad(ci, 4, '0')]
        x = cg["x"]
        y = cg["y"]
        pv = T(cg["pv"])
        nodes = [SVector{2,T}(x[i], y[i]) for i in eachindex(x)]
        wrap = if haskey(cg, "wrap_x")
            SVector{2,T}(T(cg["wrap_x"]), T(cg["wrap_y"]))
        else
            zero(SVector{2,T})
        end
        push!(contours, PVContour(nodes, pv, wrap))
    end
    return contours
end

function _load_diagnostics(g)
    if haskey(g, "diagnostics")
        dg = g["diagnostics"]
        (energy = haskey(dg, "energy") ? dg["energy"] : nothing,
         circulation = dg["circulation"],
         enstrophy = dg["enstrophy"],
         angular_momentum = haskey(dg, "angular_momentum") ? dg["angular_momentum"] : nothing,
         total_nodes = dg["total_nodes"]::Int)
    else
        nothing
    end
end

"""
    load_simulation(filename) -> Vector{NamedTuple}

Load all snapshots from a JLD2 file, sorted by step number.
"""
function ContourDynamics.load_simulation(filename::String)
    snapshots = NamedTuple[]
    jldopen(filename, "r") do f
        step_keys = sort(filter(k -> startswith(k, "step_"), keys(f));
                         by = k -> parse(Int, k[6:end]))
        for key in step_keys
            step = parse(Int, key[6:end])
            push!(snapshots, _load_snapshot_from_group(f[key], step))
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
        ratio = save_dt / dt
        rounded = round(Int, ratio)
        if abs(ratio - rounded) / max(ratio, 1) > 0.01
            @warn "jld2_recorder: save_dt/dt = $ratio is not near-integer; rounding to $rounded (effective save_dt = $(rounded * dt))"
        end
        max(1, rounded)
    end

    step_dt = dt  # capture for closure

    return function(prob, step)
        if step % interval == 0 || step == 0
            ContourDynamics.save_snapshot(filename, prob, step;
                                          dt=step_dt, diagnostics=diagnostics)
        end
    end
end

end # module
