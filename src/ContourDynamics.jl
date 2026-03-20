module ContourDynamics

using StaticArrays
using LinearAlgebra
using SpecialFunctions

include("types.jl")
include("domains.jl")
include("contours.jl")
include("surgery.jl")
include("kernels.jl")
include("ewald.jl")
include("diagnostics.jl")
include("timesteppers.jl")

export AbstractKernel, EulerKernel, QGKernel, MultiLayerQGKernel
export PVContour, nnodes
export AbstractDomain, UnboundedDomain, PeriodicDomain
export ContourProblem, MultiLayerContourProblem
export SurgeryParams
export AbstractTimeStepper, RK4Stepper, LeapfrogStepper
export nlayers, total_nodes
export velocity!, velocity, segment_velocity
export vortex_area, centroid, ellipse_moments
export energy, enstrophy, circulation, angular_momentum
export remesh, arc_lengths, surgery!
export EwaldCache, build_ewald_cache, clear_ewald_cache!
export timestep!, resize_buffers!, evolve!

# Extension stubs — implemented by package extensions
function flatten_nodes end
function unflatten_nodes! end
function to_ode_problem end
function record_evolution end
function contours_from_gridfield end
function gridfield_from_contours end
function recorded_diagnostics end

export flatten_nodes, unflatten_nodes!, to_ode_problem, record_evolution
export contours_from_gridfield, gridfield_from_contours
export recorded_diagnostics

end # module
