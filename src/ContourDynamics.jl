module ContourDynamics

using StaticArrays
using LinearAlgebra
using SpecialFunctions

include("core/device.jl")
include("core/types.jl")
include("core/domains.jl")
include("core/contours.jl")
include("core/surgery.jl")
include("velocity/common.jl")
include("velocity/unbounded/single_layer.jl")
include("velocity/periodic/cache.jl")
include("velocity/periodic/single_layer.jl")
include("accel/fmm/tree.jl")
include("accel/fmm/proxy.jl")
include("accel/fmm/translations.jl")
include("accel/fmm/fmm.jl")
include("accel/gpu/common.jl")
include("diagnostics/geometry.jl")
include("diagnostics/unbounded/single_layer.jl")
include("diagnostics/unbounded/multilayer_qg.jl")
include("diagnostics/periodic/common.jl")
include("diagnostics/periodic/single_layer.jl")
include("diagnostics/periodic/multilayer_qg.jl")
include("core/evolution.jl")
include("core/problem.jl")
include("core/show.jl")
include("core/shapes.jl")

export AbstractDevice, CPU, GPU, device_array, device_zeros, to_cpu, to_device
export AbstractKernel, EulerKernel, QGKernel, SQGKernel, MultiLayerQGKernel
export PVContour, nnodes, is_spanning, next_node, beta_staircase
export AbstractDomain, UnboundedDomain, PeriodicDomain, wrap_nodes!
export ContourProblem, MultiLayerContourProblem
export SurgeryParams
export AbstractTimeStepper, RK4Stepper, LeapfrogStepper
export nlayers, total_nodes
export velocity!, velocity, segment_velocity
export vortex_area, centroid, ellipse_moments
export energy, enstrophy, circulation, angular_momentum
export remesh, arc_lengths, surgery!
export circular_patch, elliptical_patch, rankine_vortex
export Problem, contours, kernel, domain
export EwaldCache, build_ewald_cache, setup_ewald_cache!, clear_ewald_cache!
export timestep!, resize_buffers!, evolve!

# Extension stubs — implemented by package extensions
function flatten_nodes end
function unflatten_nodes! end
function to_ode_problem end
function record_evolution end
function recorded_diagnostics end
function save_snapshot end
function load_snapshot end
function jld2_recorder end
function load_simulation end

export flatten_nodes, unflatten_nodes!, to_ode_problem, record_evolution
export recorded_diagnostics
export save_snapshot, load_snapshot, jld2_recorder, load_simulation

end # module
