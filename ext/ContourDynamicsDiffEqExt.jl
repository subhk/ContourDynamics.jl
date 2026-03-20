module ContourDynamicsDiffEqExt

using ContourDynamics
using OrdinaryDiffEq
using StaticArrays

function ContourDynamics.flatten_nodes(prob::ContourProblem{K,D,T}) where {K,D,T}
    N = total_nodes(prob)
    u = Vector{T}(undef, 2N)
    idx = 1
    for c in prob.contours
        for node in c.nodes
            u[idx] = node[1]
            u[idx+1] = node[2]
            idx += 2
        end
    end
    return u
end

function ContourDynamics.unflatten_nodes!(prob::ContourProblem, u::AbstractVector{T}) where {T}
    idx = 1
    for c in prob.contours
        for i in 1:nnodes(c)
            c.nodes[i] = SVector{2,T}(u[idx], u[idx+1])
            idx += 2
        end
    end
end

function diffeq_rhs!(du, u, p::ContourProblem{K,D,T}, t) where {K,D,T}
    ContourDynamics.unflatten_nodes!(p, u)
    N = total_nodes(p)
    vel = Vector{SVector{2,T}}(undef, N)
    velocity!(vel, p)
    idx = 1
    for i in 1:N
        du[idx] = vel[i][1]
        du[idx+1] = vel[i][2]
        idx += 2
    end
end

function ContourDynamics.to_ode_problem(prob::ContourProblem, tspan)
    u0 = ContourDynamics.flatten_nodes(prob)
    return ODEProblem(diffeq_rhs!, u0, tspan, prob)
end

end # module
