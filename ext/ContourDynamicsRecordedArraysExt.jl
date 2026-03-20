module ContourDynamicsRecordedArraysExt

using ContourDynamics
using RecordedArrays

function ContourDynamics.recorded_diagnostics(prob::ContourProblem{K,D,T};
                                              dt_record::Int=1) where {K,D,T}
    energy_rec = T[]
    enstrophy_rec = T[]
    circulation_rec = T[]

    function callback(p, step)
        if step % dt_record == 0
            push!(energy_rec, energy(p))
            push!(enstrophy_rec, enstrophy(p))
            push!(circulation_rec, circulation(p))
        end
    end

    return (energy=energy_rec, enstrophy=enstrophy_rec, circulation=circulation_rec,
            callback=callback)
end

end # module
