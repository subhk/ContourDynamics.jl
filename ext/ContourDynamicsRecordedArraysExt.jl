# RecordedArrays diagnostics extension.
#
# The extension builds callback-friendly recorders for the scalar diagnostics
# exposed by the core package. Diagnostics that are unavailable for a specific
# kernel/domain pair are recorded as NaN so one missing diagnostic does not stop
# long simulations.
module ContourDynamicsRecordedArraysExt

using ContourDynamics
using RecordedArrays

function _recording_schedule(::Type{T}, dt::Real, nsteps::Int,
                             record_every::Int) where {T<:AbstractFloat}
    ContourDynamics._require_positive("dt", dt)
    nsteps >= 0 || throw(ArgumentError("nsteps must be non-negative, got $nsteps"))
    ContourDynamics._require_positive("record_every", record_every)
    dt_T = T(dt)
    ContourDynamics._require_positive("dt converted to $T", dt_T)
    tmax = dt_T * T(nsteps)
    isfinite(tmax) || throw(ArgumentError(
        "dt * nsteps must be finite after conversion to $T; got $tmax"))
    return dt_T, tmax
end

"""
    recorded_diagnostics(prob; dt, nsteps, record_every=1)

Create time-stamped diagnostic recorders using RecordedArrays.

Returns a NamedTuple with `energy`, `enstrophy`, `circulation`,
`angular_momentum` (recorded arrays), `clock` (the shared `ContinuousClock`),
and `callback` (for use with `evolve!`).

After the simulation, retrieve the full history via `getentries`, `getts`, `getvs`
from RecordedArrays.

# Example
```julia
using ContourDynamics, RecordedArrays
rec = recorded_diagnostics(prob; dt=0.01, nsteps=10000, record_every=10)
evolve!(prob, stepper, params; nsteps=10000, callbacks=[rec.callback])

# Access history:
e = getentries(rec.energy)
```
"""
function ContourDynamics.recorded_diagnostics(prob::ContourProblem{K,D,T};
                                              dt::Real,
                                              nsteps::Int,
                                              record_every::Int=1) where {K,D,T}
    dt_T, tmax = _recording_schedule(T, dt, nsteps, record_every)
    clock = ContinuousClock(tmax)

    energy_rec = recorded(StaticEntry, clock, T[])
    enstrophy_rec = recorded(StaticEntry, clock, T[])
    circulation_rec = recorded(StaticEntry, clock, T[])
    angmom_rec = recorded(StaticEntry, clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        # The callback receives integer step counts from evolve!. Convert that
        # to monotonically increasing clock time before pushing diagnostic rows.
        if step % record_every == 0
            t = dt_T * T(step)
            advance = t - last_time[]
            if advance > zero(T)
                increase!(clock, advance)
                last_time[] = t
            end
            try
                push!(energy_rec, energy(p))
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
                push!(energy_rec, T(NaN))
            end
            push!(enstrophy_rec, enstrophy(p))
            push!(circulation_rec, circulation(p))
            try
                push!(angmom_rec, angular_momentum(p))
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
                push!(angmom_rec, T(NaN))
            end
        end
    end

    return (energy=energy_rec, enstrophy=enstrophy_rec, circulation=circulation_rec,
            angular_momentum=angmom_rec, clock=clock, callback=callback)
end

"""
    recorded_diagnostics(prob::MultiLayerContourProblem; dt, nsteps, record_every=1)

Create RecordedArrays diagnostic recorders for a multi-layer problem. The
recorded scalar values are layer-summed diagnostics, matching the core
`energy`, `enstrophy`, `circulation`, and `angular_momentum` methods.
"""
function ContourDynamics.recorded_diagnostics(prob::MultiLayerContourProblem{N,K,D,T};
                                              dt::Real,
                                              nsteps::Int,
                                              record_every::Int=1) where {N,K,D,T}
    dt_T, tmax = _recording_schedule(T, dt, nsteps, record_every)
    clock = ContinuousClock(tmax)

    energy_rec = recorded(StaticEntry, clock, T[])
    enstrophy_rec = recorded(StaticEntry, clock, T[])
    circulation_rec = recorded(StaticEntry, clock, T[])
    angmom_rec = recorded(StaticEntry, clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        # Keep the same clock semantics as the single-layer method: callbacks at
        # skipped steps do not advance the clock or allocate entries.
        if step % record_every == 0
            t = dt_T * T(step)
            advance = t - last_time[]
            if advance > zero(T)
                increase!(clock, advance)
                last_time[] = t
            end
            try
                push!(energy_rec, energy(p))
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
                push!(energy_rec, T(NaN))
            end
            push!(enstrophy_rec, enstrophy(p))
            push!(circulation_rec, circulation(p))
            try
                push!(angmom_rec, angular_momentum(p))
            catch e
                e isa Union{MethodError, ArgumentError} || rethrow()
                push!(angmom_rec, T(NaN))
            end
        end
    end

    return (energy=energy_rec, enstrophy=enstrophy_rec, circulation=circulation_rec,
            angular_momentum=angmom_rec, clock=clock, callback=callback)
end

# Forwarder so the high-level `Problem` wrapper can be recorded directly.
ContourDynamics.recorded_diagnostics(prob::ContourDynamics.Problem; kwargs...) =
    ContourDynamics.recorded_diagnostics(prob.contour_problem; kwargs...)

end # module
