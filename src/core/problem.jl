# High-level Problem wrapper — bundles ContourProblem + stepper + surgery params
# into a single object for GeophysicalFlows-style convenience.

"""
    Problem{P,S,SP}

Convenience wrapper bundling a [`ContourProblem`](@ref) (or
[`MultiLayerContourProblem`](@ref)), a time stepper, and optional
[`SurgeryParams`](@ref) into a single object.

Construct via the keyword factory [`Problem(; kwargs...)`](@ref) or directly:

    Problem(contour_problem, stepper, surgery_params)
"""
struct Problem{P<:Union{ContourProblem, MultiLayerContourProblem},
               S<:AbstractTimeStepper,
               SP<:Union{SurgeryParams, Nothing}}
    contour_problem::P
    stepper::S
    surgery_params::SP
end

# ── Forwarded accessors ─────────────────────────────────

"""Return the contours of the underlying problem."""
contours(prob::Problem) = contours(prob.contour_problem)

"""Materialize contours on CPU for output, animation, plotting, or inspection."""
materialize_contours(prob::Problem) = materialize_contours(prob.contour_problem)

"""Return the kernel of the underlying problem."""
kernel(prob::Problem) = prob.contour_problem.kernel

"""Return the domain of the underlying problem."""
domain(prob::Problem) = prob.contour_problem.domain

"""Return the total node count of the wrapped contour problem."""
total_nodes(prob::Problem) = total_nodes(prob.contour_problem)

"""Return the energy diagnostic of the wrapped contour problem."""
energy(prob::Problem) = energy(prob.contour_problem)

"""Return the circulation diagnostic of the wrapped contour problem."""
circulation(prob::Problem) = circulation(prob.contour_problem)

"""Return the enstrophy diagnostic of the wrapped contour problem."""
enstrophy(prob::Problem) = enstrophy(prob.contour_problem)

"""Return the angular-momentum diagnostic of the wrapped contour problem."""
angular_momentum(prob::Problem) = angular_momentum(prob.contour_problem)

"""Compute node velocities for the wrapped contour problem into `vel`."""
velocity!(vel, prob::Problem) = velocity!(vel, prob.contour_problem)

"""Compute velocity induced by the wrapped contour problem at point `x`."""
velocity(prob::Problem, x) = velocity(prob.contour_problem, x)

"""Return signed contour areas from the wrapped contour problem."""
vortex_area(prob::Problem) = vortex_area(prob.contour_problem)

"""Return the number of layers when the wrapped contour problem is multi-layer."""
nlayers(prob::Problem) = nlayers(prob.contour_problem)

# ── evolve! overload ────────────────────────────────────

"""
    evolve!(prob::Problem; nsteps, callbacks=nothing)

Run the simulation for `nsteps` time steps. Surgery is applied according to
`prob.surgery_params` (or skipped if `nothing`).
"""
function evolve!(prob::Problem; nsteps::Int, callbacks=nothing)
    evolve!(prob.contour_problem, prob.stepper, prob.surgery_params;
            nsteps, callbacks)
    return prob
end

include("problem_factory.jl")
