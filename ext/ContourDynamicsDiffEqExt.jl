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

# Build a closure-based RHS that pre-allocates the velocity buffer
function _make_rhs(prob::ContourProblem{K,D,T}) where {K,D,T}
    N = total_nodes(prob)
    vel = Vector{SVector{2,T}}(undef, N)
    function rhs!(du, u, p, t)
        ContourDynamics.unflatten_nodes!(p, u)
        Ncur = total_nodes(p)
        if length(vel) != Ncur
            resize!(vel, Ncur)
        end
        velocity!(vel, p)
        idx = 1
        for i in 1:Ncur
            du[idx] = vel[i][1]
            du[idx+1] = vel[i][2]
            idx += 2
        end
    end
    return rhs!
end

"""
    to_ode_problem(prob::ContourProblem, tspan; surgery_params=nothing)

Wrap a `ContourProblem` as an `ODEProblem` for use with OrdinaryDiffEq solvers.

If `surgery_params::SurgeryParams` is provided, a `DiscreteCallback` is added
that performs contour surgery every `surgery_params.n_surgery` steps.
"""
function ContourDynamics.to_ode_problem(prob::ContourProblem, tspan;
                                         surgery_params=nothing)
    u0 = ContourDynamics.flatten_nodes(prob)
    rhs! = _make_rhs(prob)

    if surgery_params === nothing
        return ODEProblem(rhs!, u0, tspan, prob)
    end

    # Surgery callback: apply every n_surgery accepted steps
    step_count = Ref(0)
    function condition(u, t, integrator)
        step_count[] += 1
        return step_count[] % surgery_params.n_surgery == 0
    end
    function affect!(integrator)
        ContourDynamics.unflatten_nodes!(integrator.p, integrator.u)
        surgery!(integrator.p, surgery_params)
        # Re-flatten after surgery (node count may have changed)
        new_u = ContourDynamics.flatten_nodes(integrator.p)
        resize!(integrator, length(new_u))
        integrator.u .= new_u
    end
    cb = DiscreteCallback(condition, affect!)

    return ODEProblem(rhs!, u0, tspan, prob, callback=cb)
end

end # module
