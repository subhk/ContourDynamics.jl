module ContourDynamicsRecordedArraysExt

using ContourDynamics
using RecordedArrays

"""
    recorded_diagnostics(prob; dt, nsteps, dt_record=1)

Create time-stamped diagnostic recorders using RecordedArrays.

Returns a NamedTuple with `energy`, `enstrophy`, `circulation` (recorded arrays),
`clock` (the shared `ContinuousClock`), and `callback` (for use with `evolve!`).

After the simulation, retrieve the full history via `getentries`, `getts`, `getvs`
from RecordedArrays.

# Example
```julia
using ContourDynamics, RecordedArrays
rec = recorded_diagnostics(prob; dt=0.01, nsteps=10000, dt_record=10)
evolve!(prob, stepper, params; nsteps=10000, callbacks=[rec.callback])

# Access history:
e = getentries(rec.energy)
```
"""
function ContourDynamics.recorded_diagnostics(prob::ContourProblem{K,D,T};
                                              dt::T,
                                              nsteps::Int,
                                              dt_record::Int=1) where {K,D,T}
    tmax = dt * T(nsteps)
    clock = ContinuousClock(tmax)

    energy_rec = recorded(StaticEntry, clock, T[])
    enstrophy_rec = recorded(StaticEntry, clock, T[])
    circulation_rec = recorded(StaticEntry, clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        if step % dt_record == 0
            t = dt * T(step)
            advance = t - last_time[]
            if advance > zero(T)
                increase!(clock, advance)
                last_time[] = t
            end
            push!(energy_rec, energy(p))
            push!(enstrophy_rec, enstrophy(p))
            push!(circulation_rec, circulation(p))
        end
    end

    return (energy=energy_rec, enstrophy=enstrophy_rec, circulation=circulation_rec,
            clock=clock, callback=callback)
end

end # module
