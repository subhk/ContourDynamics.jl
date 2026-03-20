module ContourDynamics

using StaticArrays
using LinearAlgebra
using SpecialFunctions

include("types.jl")
include("domains.jl")

export AbstractKernel, EulerKernel, QGKernel, MultiLayerQGKernel
export PVContour, nnodes
export AbstractDomain, UnboundedDomain, PeriodicDomain
export ContourProblem, MultiLayerContourProblem
export SurgeryParams
export AbstractTimeStepper, RK4Stepper, LeapfrogStepper
export nlayers, total_nodes

end # module
