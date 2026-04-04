module ContourDynamicsRecordedArraysExt

using ContourDynamics
using RecordedArrays

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
                                              record_every::Int=1,
                                              dt_record::Int=record_every) where {K,D,T}
    dt = T(dt)
    tmax = dt * T(nsteps)
    clock = ContinuousClock(tmax)

    energy_rec = recorded(StaticEntry, clock, T[])
    enstrophy_rec = recorded(StaticEntry, clock, T[])
    circulation_rec = recorded(StaticEntry, clock, T[])
    angmom_rec = recorded(StaticEntry, clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        if step % dt_record == 0
            t = dt * T(step)
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

# Multi-layer version
function ContourDynamics.recorded_diagnostics(prob::MultiLayerContourProblem{N,K,D,T};
                                              dt::Real,
                                              nsteps::Int,
                                              record_every::Int=1,
                                              dt_record::Int=record_every) where {N,K,D,T}
    dt = T(dt)
    tmax = dt * T(nsteps)
    clock = ContinuousClock(tmax)

    energy_rec = recorded(StaticEntry, clock, T[])
    enstrophy_rec = recorded(StaticEntry, clock, T[])
    circulation_rec = recorded(StaticEntry, clock, T[])
    angmom_rec = recorded(StaticEntry, clock, T[])

    last_time = Ref(zero(T))

    function callback(p, step)
        if step % dt_record == 0
            t = dt * T(step)
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

end # module
