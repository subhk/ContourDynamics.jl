# JLD2 persistence extension.
#
# Snapshots are written as portable groups of plain arrays plus metadata rather
# than serialized Julia objects. That makes files easier to inspect and more
# robust across internal type changes.
module ContourDynamicsJLD2Ext

using ContourDynamics
using JLD2
using StaticArrays

# Snapshot copies decoupled from the live contours (exercised by the tests).
_snapshot_contours(prob::ContourProblem) = snapshot_contours(prob)
_snapshot_layers(prob::MultiLayerContourProblem) = snapshot_contours(prob)
_snapshot(prob::ContourProblem) = _snapshot_contours(prob)
_snapshot(prob::MultiLayerContourProblem) = _snapshot_layers(prob)

function _save_contour!(g, c::PVContour, ci::Int)
    cg = JLD2.Group(g, "contour_" * lpad(ci, 4, '0'))
    # Store coordinates as plain arrays for portability and easy inspection
    # from non-Julia tooling.
    cg["x"] = [c.nodes[i][1] for i in 1:nnodes(c)]
    cg["y"] = [c.nodes[i][2] for i in 1:nnodes(c)]
    cg["pv"] = c.pv
    cg["nnodes"] = nnodes(c)
    cg["wrap_x"] = c.wrap[1]
    cg["wrap_y"] = c.wrap[2]
    cg["corners"] = collect(c.corners)
    return nothing
end

function _save_contours!(g, contours::Vector{<:PVContour})
    g["ncontours"] = length(contours)
    for (ci, c) in enumerate(contours)
        _save_contour!(g, c, ci)
    end
    return nothing
end

# Save kernel/domain metadata so snapshots can be restored without external
# information from the original script.
function _save_metadata!(g, kernel::EulerKernel, domain)
    g["kernel_type"] = "EulerKernel"
    _save_domain!(g, domain)
end
function _save_metadata!(g, kernel::QGKernel{T}, domain) where {T}
    g["kernel_type"] = "QGKernel"
    g["kernel_Ld"] = kernel.Ld
    _save_domain!(g, domain)
end
function _save_metadata!(g, kernel::BetaPlaneQGKernel{T}, domain) where {T}
    g["kernel_type"] = "BetaPlaneQGKernel"
    g["kernel_beta"] = kernel.beta
    g["kernel_Ld"] = kernel.Ld
    g["kernel_reference_contours"] = length(kernel.reference_contours)
    # The reference staircase is frozen kernel state, not live contour state.
    # Persist its complete portable geometry so load_problem can reproduce the
    # original inversion after the live staircase has evolved or been remeshed.
    rg = JLD2.Group(g, "kernel_reference_geometry")
    _save_contours!(rg, kernel.reference_contours)
    _save_domain!(g, domain)
end
function _save_metadata!(g, kernel::SQGKernel{T}, domain) where {T}
    g["kernel_type"] = "SQGKernel"
    g["kernel_delta"] = kernel.δ
    _save_domain!(g, domain)
end
function _save_metadata!(g, kernel::MultiLayerQGKernel{N,M,T}, domain) where {N,M,T}
    g["kernel_type"] = "MultiLayerQGKernel"
    g["kernel_Ld"] = collect(kernel.Ld)
    g["kernel_coupling"] = collect(kernel.coupling)
    # The layer depths weight the energy and angular momentum; without them a
    # kernel rebuilt from Ld and the coupling alone infers different depths.
    g["kernel_layer_thicknesses"] = collect(kernel.layer_thicknesses)
    g["kernel_nlayers"] = N
    _save_domain!(g, domain)
end
function _save_domain!(g, ::UnboundedDomain)
    g["domain_type"] = "UnboundedDomain"
end
function _save_domain!(g, domain::PeriodicDomain{T}) where {T}
    g["domain_type"] = "PeriodicDomain"
    g["domain_Lx"] = domain.Lx
    g["domain_Ly"] = domain.Ly
end

"""
    save_snapshot(filename, prob, step; dt=nothing, diagnostics=true)

Save a single snapshot of the simulation state to a JLD2 file.
Each snapshot is stored under a group `step_NNNNNN`.

Saves: contour nodes, PV values, node counts, kernel/domain metadata,
and optionally energy, circulation, enstrophy, and angular momentum.

The file is opened for appending: an existing group for the same `step` is
replaced and all other groups are kept. To record a new run into an existing
file, use [`jld2_recorder`](@ref), which clears earlier step groups when the run
starts, or remove the file first; otherwise [`load_simulation`](@ref) returns
snapshots from both runs.
"""
function ContourDynamics.save_snapshot(filename::String,
                                       prob::Union{ContourProblem, MultiLayerContourProblem},
                                       step::Int;
                                       dt::Union{Nothing,Real}=nothing,
                                       diagnostics::Bool=true)
    group = "step_" * lpad(step, 6, '0')
    snapshot = _snapshot(prob)

    jldopen(filename, "a+") do f
        if haskey(f, group)
            delete!(f, group)
        end
        g = JLD2.Group(f, group)

        g["step"] = step
        _save_layer_count!(g, prob)
        if dt !== nothing
            g["time"] = step * dt
        end
        # Save kernel/domain metadata before contour data so readers can rebuild
        # the right problem type even if loading stops early.
        mg = JLD2.Group(g, "metadata")
        _save_metadata!(mg, prob.kernel, prob.domain)
        mg["coordinate_zero"] = zero(ContourDynamics._problem_float_type(prob))

        _save_snapshot_contours!(g, prob, snapshot)

        diagnostics && _save_diagnostics!(JLD2.Group(g, "diagnostics"), prob)
    end

    return nothing
end

# Only multi-layer snapshots carry a layer count; its presence is what readers
# use to tell the two layouts apart.
_save_layer_count!(g, ::ContourProblem) = nothing
_save_layer_count!(g, ::MultiLayerContourProblem{N}) where {N} = (g["nlayers"] = N; nothing)

# Single-layer contours live directly under the step group; multi-layer
# snapshots store one layer_NN subgroup per layer.
_save_snapshot_contours!(g, ::ContourProblem, contours) = _save_contours!(g, contours)
function _save_snapshot_contours!(g, ::MultiLayerContourProblem, layers)
    for (li, layer) in enumerate(layers)
        lg = JLD2.Group(g, "layer_" * lpad(li, 2, '0'))
        _save_contours!(lg, layer)
    end
    return nothing
end

# Energy and angular momentum are omitted (not written) when undefined for the
# kernel/domain pair; readers treat a missing key as `nothing`.
function _save_diagnostics!(dg, prob)
    dg["circulation"] = circulation(prob)
    dg["enstrophy"] = enstrophy(prob)
    dg["total_nodes"] = total_nodes(prob)
    e = ContourDynamics._try_diagnostic(energy, prob)
    e === nothing || (dg["energy"] = e)
    L = ContourDynamics._try_diagnostic(angular_momentum, prob)
    L === nothing || (dg["angular_momentum"] = L)
    return nothing
end

# Forwarder so the high-level `Problem` wrapper works directly with snapshots.
ContourDynamics.save_snapshot(filename::String, prob::ContourDynamics.Problem,
                              step::Int; kwargs...) =
    ContourDynamics.save_snapshot(filename, prob.contour_problem, step; kwargs...)

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
    fallback_T = haskey(g, "metadata") ? _metadata_float_type(g["metadata"]) : Float64

    if is_multilayer
        # Multi-layer snapshots are stored as layer_N/contour_M groups. Infer a
        # single coordinate element type from the first non-empty layer so empty
        # trailing layers still reconstruct with the same type.
        nlyr = g["nlayers"]::Int
        inferred_T = fallback_T
        for li in 1:nlyr
            lg = g["layer_" * lpad(li, 2, '0')]
            nc = lg["ncontours"]::Int
            if nc > 0
                cg1 = lg["contour_" * lpad(1, 4, '0')]
                inferred_T = eltype(cg1["x"])
                break
            end
        end
        layers = let T = inferred_T
            all_layers = Vector{Vector{PVContour{T}}}(undef, nlyr)
            for li in 1:nlyr
                lg = g["layer_" * lpad(li, 2, '0')]
                nc = lg["ncontours"]::Int
                all_layers[li] = _load_contours(lg, nc; fallback_T=T)
            end
            Tuple(all_layers)
        end
        diag = _load_diagnostics(g)
        return (layers=layers, diagnostics=diag, step=step, time=time)
    else
        # Single-layer snapshots keep contour groups directly under the step
        # group and reconstruct to a flat Vector{PVContour}.
        nc = g["ncontours"]::Int
        contours = _load_contours(g, nc; fallback_T=fallback_T)
        diag = _load_diagnostics(g)
        return (contours=contours, diagnostics=diag, step=step, time=time)
    end
end

# ── helpers ──────────────────────────────────────────────────

function _load_contours(g, nc::Int; fallback_T::Type{<:AbstractFloat}=Float64)
    nc == 0 && return PVContour{fallback_T}[]

    # Infer element type from file data; fallback_T is only used when nc == 0.
    cg1 = g["contour_" * lpad(1, 4, '0')]
    T = eltype(cg1["x"])

    contours = PVContour{T}[]
    for ci in 1:nc
        cg = g["contour_" * lpad(ci, 4, '0')]
        x = cg["x"]
        y = cg["y"]
        pv = T(cg["pv"])
        nodes = [SVector{2,T}(x[i], y[i]) for i in eachindex(x)]
        has_wx, has_wy = haskey(cg, "wrap_x"), haskey(cg, "wrap_y")
        wrap = if has_wx && has_wy
            SVector{2,T}(T(cg["wrap_x"]), T(cg["wrap_y"]))
        elseif has_wx || has_wy
            error("Corrupted snapshot: contour $ci has wrap_x without wrap_y (or vice versa)")
        else
            zero(SVector{2,T})
        end
        corners = haskey(cg, "corners") ? Bool.(cg["corners"]) : falses(length(nodes))
        push!(contours, PVContour(nodes, pv, wrap, corners))
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

# Reconstruct a domain from saved metadata.
function _load_domain(mg)
    dtype = mg["domain_type"]::String
    if dtype == "UnboundedDomain"
        return UnboundedDomain()
    elseif dtype == "PeriodicDomain"
        return PeriodicDomain(mg["domain_Lx"], mg["domain_Ly"])
    else
        error("Unknown domain_type \"$dtype\" in snapshot metadata")
    end
end

# Reconstruct a single-layer kernel from saved metadata. The saved scalar
# parameters already carry the original float type, so the rebuilt kernel
# matches the loaded contours' coordinate type.
function _load_single_layer_kernel(mg)
    ktype = mg["kernel_type"]::String
    if ktype == "EulerKernel"
        return EulerKernel()
    elseif ktype == "QGKernel"
        return QGKernel(mg["kernel_Ld"])
    elseif ktype == "SQGKernel"
        return SQGKernel(mg["kernel_delta"])
    elseif ktype == "BetaPlaneQGKernel"
        reference_key = "kernel_reference_geometry"
        haskey(mg, reference_key) || throw(ArgumentError(
            "load_problem cannot rebuild this BetaPlaneQGKernel because the snapshot " *
            "does not contain its frozen reference-contour geometry. This file was " *
            "written by an older ContourDynamics version; load_snapshot can still " *
            "read its live state, but the original reference geometry must be supplied " *
            "manually."))
        rg = mg[reference_key]
        haskey(rg, "ncontours") || error(
            "Corrupted snapshot: beta-plane reference geometry has no ncontours field")
        nr = rg["ncontours"]::Int
        if haskey(mg, "kernel_reference_contours")
            expected_nr = mg["kernel_reference_contours"]::Int
            nr == expected_nr || error(
                "Corrupted snapshot: beta-plane reference contour count is $nr, " *
                "but metadata records $expected_nr")
        end
        reference = _load_contours(rg, nr; fallback_T=_metadata_float_type(mg))
        return BetaPlaneQGKernel(mg["kernel_beta"], mg["kernel_Ld"], reference)
    elseif ktype == "MultiLayerQGKernel"
        throw(ArgumentError(
            "load_problem rebuilds single-layer problems only. This snapshot is multi-layer; " *
            "use load_snapshot to read the layers and rebuild the MultiLayerContourProblem manually."))
    else
        error("Unknown kernel_type \"$ktype\" in snapshot metadata")
    end
end

"""
    load_problem(filename, step; dev=CPU()) -> ContourProblem

Reconstruct a runnable single-layer [`ContourProblem`](@ref) from the snapshot at
`step`, using the kernel/domain metadata written by [`save_snapshot`](@ref).

Supported kernels: `EulerKernel`, `QGKernel`, `SQGKernel`, and
`BetaPlaneQGKernel`. Beta-plane snapshots include the frozen reference-contour
geometry needed to reproduce their inversion. Legacy beta-plane files that
predate that geometry remain readable through [`load_snapshot`](@ref), but
cannot be reconstructed automatically. `MultiLayerQGKernel` snapshots remain
state-only and must be rebuilt manually. Stepper and surgery state are not
persisted, so the caller recreates the time stepper and [`SurgeryParams`](@ref)
to continue a run.
"""
function ContourDynamics.load_problem(filename::String, step::Int; dev=ContourDynamics.CPU())
    group = "step_" * lpad(step, 6, '0')
    jldopen(filename, "r") do f
        haskey(f, group) || error("Step $step not found in $filename")
        g = f[group]
        haskey(g, "nlayers") && throw(ArgumentError(
            "load_problem rebuilds single-layer problems only; this snapshot is multi-layer. " *
            "Use load_snapshot to read the layers and rebuild the MultiLayerContourProblem manually."))
        haskey(g, "metadata") || error(
            "Snapshot at step $step has no metadata group; cannot rebuild the problem")
        mg = g["metadata"]
        domain = _load_domain(mg)
        kernel = _load_single_layer_kernel(mg)
        nc = g["ncontours"]::Int
        # Use the kernel/domain float type as the fallback so a zero-contour
        # snapshot rebuilds contours matching the kernel type (avoids a spurious
        # float-type mismatch error for empty non-Float64 problems).
        contours = _load_contours(g, nc; fallback_T=_metadata_float_type(mg))
        return ContourProblem(kernel, domain, contours; dev=dev)
    end
end

# Float type recorded in the metadata (matches the original contour coordinate
# type, since construction enforced kernel/domain/contour type agreement).
function _metadata_float_type(mg)
    haskey(mg, "coordinate_zero") && return typeof(mg["coordinate_zero"])
    if haskey(mg, "kernel_Ld")
        Ld = mg["kernel_Ld"]
        return Ld isa AbstractArray ? eltype(Ld) : typeof(Ld)
    end
    haskey(mg, "kernel_delta") && return typeof(mg["kernel_delta"])
    haskey(mg, "domain_Lx") && return typeof(mg["domain_Lx"])
    return Float64
end

"""
    jld2_recorder(filename; save_every=nothing, save_dt=nothing, dt=nothing, diagnostics=true, append=false)

Create a callback for `evolve!` that saves snapshots to a JLD2 file.

Specify either:
- `save_every::Int` — save every N iterations
- `save_dt` + `dt` — save every `save_dt` time units (requires the stepper's `dt`)

A run starts afresh: before its first snapshot, and whenever a snapshot step
does not advance past the previous one (the recorder is reused for a rerun),
the `step_*` groups already in the file are deleted, so
[`load_simulation`](@ref) never mixes two runs. Other groups are kept. Pass
`append=true` to continue a run already recorded in the file instead, for
example after restarting from a checkpoint with `step_offset`.

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
                                        diagnostics::Bool=true,
                                        append::Bool=false)
    if save_every === nothing && save_dt === nothing
        throw(ArgumentError("Specify either save_every (iterations) or save_dt (time interval)"))
    end
    if save_every !== nothing && save_dt !== nothing
        throw(ArgumentError("Specify only one of save_every (iterations) or save_dt (time interval)"))
    end
    if save_dt !== nothing && dt === nothing
        throw(ArgumentError("save_dt requires dt (time step size)"))
    end
    save_every !== nothing && ContourDynamics._require_positive("save_every", save_every)
    dt !== nothing && ContourDynamics._require_positive("dt", dt)
    if save_dt !== nothing
        ContourDynamics._require_positive("save_dt", save_dt)
    end

    # Compute one integer iteration interval from either iteration-based or
    # time-based user input. The callback stays step-based to match evolve!.
    interval = if save_every !== nothing
        save_every
    else
        ratio = save_dt / dt
        (isfinite(ratio) && ratio <= typemax(Int)) || throw(ArgumentError(
            "save_dt/dt must be finite and fit in an Int; got $ratio"))
        steps = max(1, round(Int, ratio))
        if abs(ratio - steps) / max(ratio, 1) > 0.01
            @warn "jld2_recorder: save_dt/dt = $ratio is not a whole number of steps; saving every $steps step(s) (effective save_dt = $(steps * dt))"
        end
        steps
    end

    step_dt = dt  # capture for closure
    started = Ref(append)
    last_step = Ref{Union{Nothing,Int}}(nothing)

    return function(prob, step)
        if step % interval == 0
            previous = last_step[]
            if !started[] || (previous !== nothing && step <= previous)
                _clear_step_groups!(filename)
                started[] = true
            end
            ContourDynamics.save_snapshot(filename, prob, step;
                                          dt=step_dt, diagnostics=diagnostics)
            last_step[] = step
        end
    end
end

# Delete every `step_*` group so a new run does not interleave with snapshots
# left by an earlier one. Other groups in the file are left untouched.
function _clear_step_groups!(filename::String)
    isfile(filename) || return nothing
    jldopen(filename, "a+") do f
        for key in collect(keys(f))
            startswith(key, "step_") && delete!(f, key)
        end
    end
    return nothing
end

end # module
