# Note: StaticArrays and LinearAlgebra are imported in the parent module (ContourDynamics.jl).
# No `using` statements needed here since this file is `include`d.

# Core object model, split by responsibility so new contributors can locate the
# main types without scrolling through one monolithic file.

include("kernel_types.jl")
include("contour_types.jl")
include("device_state.jl")
include("domain_types.jl")
include("beta_plane_types.jl")
include("problem_types.jl")
include("surgery_types.jl")
include("stepper_types.jl")
