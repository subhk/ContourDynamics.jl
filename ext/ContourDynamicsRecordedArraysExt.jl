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

After the simulation, retrieve the full history via `record`, `getts`, and `getvs`
from RecordedArrays.

For a `MultiLayerContourProblem` the recorded scalar values are layer-summed
diagnostics, matching the core `energy`, `enstrophy`, `circulation`, and
`angular_momentum` methods.

# Example
```julia
using ContourDynamics, RecordedArrays
rec = recorded_diagnostics(prob; dt=0.01, nsteps=10000, record_every=10)
evolve!(prob, stepper, params; nsteps=10000, callbacks=[rec.callback])

# Access history:
e = record(rec.energy)
```
"""
function ContourDynamics.recorded_diagnostics(
        prob::Union{ContourProblem, MultiLayerContourProblem};
        dt::Real, nsteps::Int, record_every::Int=1)
    T = ContourDynamics._problem_float_type(prob)
    dt_T, tmax = _recording_schedule(T, dt, nsteps, record_every)
    clock = ContinuousClock(tmax)

    energy_rec = StaticRArray(clock, T[])
    enstrophy_rec = StaticRArray(clock, T[])
    circulation_rec = StaticRArray(clock, T[])
    angmom_rec = StaticRArray(clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        # The callback receives integer step counts from evolve!. Convert that
        # to monotonically increasing clock time before pushing diagnostic rows.
        # Callbacks at skipped steps do not advance the clock or allocate entries.
        if step % record_every == 0
            t = dt_T * T(step)
            advance = t - last_time[]
            if advance > zero(T)
                increase!(clock, advance)
                last_time[] = t
            end
            push!(energy_rec,
                  something(ContourDynamics._try_diagnostic(energy, p), T(NaN)))
            push!(enstrophy_rec, enstrophy(p))
            push!(circulation_rec, circulation(p))
            push!(angmom_rec,
                  something(ContourDynamics._try_diagnostic(angular_momentum, p), T(NaN)))
        end
    end

    return (energy=energy_rec, enstrophy=enstrophy_rec, circulation=circulation_rec,
            angular_momentum=angmom_rec, clock=clock, callback=callback)
end

# Forwarder so the high-level `Problem` wrapper can be recorded directly.
ContourDynamics.recorded_diagnostics(prob::ContourDynamics.Problem; kwargs...) =
    ContourDynamics.recorded_diagnostics(prob.contour_problem; kwargs...)

end # module
