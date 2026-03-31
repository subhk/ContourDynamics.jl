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

function ContourDynamics.unflatten_nodes!(prob::ContourProblem{K,D,Tc}, u::AbstractVector) where {K,D,Tc}
    expected = 2 * total_nodes(prob)
    length(u) >= expected || throw(DimensionMismatch(
        "u length ($(length(u))) must be >= 2 * total_nodes ($(expected))"))
    idx = 1
    for c in prob.contours
        for i in 1:nnodes(c)
            c.nodes[i] = SVector{2,Tc}(Tc(u[idx]), Tc(u[idx+1]))
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
        length(du) >= 2 * Ncur || throw(DimensionMismatch(
            "du length ($(length(du))) too small for $Ncur nodes (need $(2*Ncur))"))
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
    to_ode_problem(prob::ContourProblem, tspan; surgery_params=nothing, surgery_dt=nothing)

Wrap a `ContourProblem` as an `ODEProblem` for use with OrdinaryDiffEq solvers.

If `surgery_params::SurgeryParams` is provided, a `PeriodicCallback` is added
that performs contour surgery at fixed time intervals.

The surgery interval is determined by:
- `surgery_dt`: explicit time interval between surgery passes.
- If `surgery_dt` is not given, defaults to `surgery_params.n_surgery * dt` where
  `dt = (tspan[2] - tspan[1]) / 1000`. For fixed-step solvers, prefer setting
  `surgery_dt` explicitly to match your step size.

!!! warning "Solver compatibility"
    The RHS closure mutates `prob.contours` in-place on every evaluation.
    This is **only safe** with fixed-step, non-adaptive solvers (e.g. `RK4()`,
    `Euler()` with `adaptive=false`).  Adaptive solvers (e.g. `Tsit5()`) evaluate
    the RHS at multiple trial points with step rejection, causing rejected-step
    state to overwrite accepted-step state.  Always use `adaptive=false` or a
    fixed-step solver.

!!! warning "Thread safety"
    The returned `ODEProblem` captures a pre-allocated velocity buffer in its
    RHS closure.  Do **not** use the same problem with parallel ensemble solvers
    (`EnsembleThreads()`) — each thread would write to the shared buffer
    concurrently.  Create a separate `to_ode_problem` call per thread instead.
"""
function ContourDynamics.to_ode_problem(prob::ContourProblem, tspan;
                                         surgery_params=nothing,
                                         surgery_dt=nothing)
    u0 = ContourDynamics.flatten_nodes(prob)
    rhs! = _make_rhs(prob)

    if surgery_params === nothing
        return ODEProblem(rhs!, u0, tspan, prob)
    end

    # Determine surgery time interval
    dt_surgery = if surgery_dt !== nothing
        surgery_dt
    else
        # Fallback: estimate from tspan
        (tspan[2] - tspan[1]) / 1000 * surgery_params.n_surgery
    end

    # Time-based surgery condition (works with both fixed and adaptive solvers)
    next_surgery_time = Ref(tspan[1] + dt_surgery)
    function condition(u, t, integrator)
        return t >= next_surgery_time[]
    end
    function affect!(integrator)
        ContourDynamics.unflatten_nodes!(integrator.p, integrator.u)
        surgery!(integrator.p, surgery_params)
        new_u = ContourDynamics.flatten_nodes(integrator.p)
        if length(new_u) != length(integrator.u)
            resize!(integrator, length(new_u))
            # Zero-fill before copyto! so that any internal caches the
            # integrator derives from u start from a clean state.
            fill!(integrator.u, zero(eltype(integrator.u)))
        end
        copyto!(integrator.u, new_u)
        # Advance past the current time so that large adaptive steps
        # that skip multiple intervals don't leave the threshold behind.
        while next_surgery_time[] <= integrator.t
            next_surgery_time[] += dt_surgery
        end
    end
    cb = DiscreteCallback(condition, affect!)

    return ODEProblem(rhs!, u0, tspan, prob, callback=cb)
end

end # module
