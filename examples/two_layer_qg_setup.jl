# Figure 19 of Polvani, Zabusky & Flierl (1989).
# This setup has no visualization or file-output dependencies.
using ContourDynamics
using StaticArrays

function polvani_upper_layer_merger_problem(; nodes_per_contour=128,
                                             R=1.0,
                                             pv=1.0,
                                             depth_ratio=0.2,
                                             γ=5.0,
                                             initial_distance=2.2,
                                             dt=0.01,
                                             surgery_δ=0.0025,
                                             surgery_mu=0.01,
                                             max_segment=0.12,
                                             surgery_every=25)
    depth_ratio > 0 || throw(ArgumentError("depth_ratio must be positive for the two-layer solver. Use a single-layer QG kernel for the equivalent-barotropic δ = 0 limit."))
    Ld = SVector(1.0 / (γ * sqrt(1 + depth_ratio)))
    coupling = γ^2 * SMatrix{2,2}(-1.0, depth_ratio, 1.0, -depth_ratio)
    # Thicknesses in units of the lower-layer depth. The kernel performs the
    # weighted similarity transform internally.
    H = SVector(depth_ratio, 1.0)

    c_left = circular_patch(R, nodes_per_contour, pv; cx=-initial_distance / 2)
    c_right = circular_patch(R, nodes_per_contour, pv; cx=initial_distance / 2)
    surgery = SurgeryParams(surgery_δ, surgery_mu, max_segment, 1e-8, surgery_every)
    prob = Problem(; kernel=:multilayer_qg, Ld=Ld, coupling=coupling,
                     layer_thicknesses=H,
                     layers=([c_left, c_right], PVContour{Float64}[]),
                     dt=dt, surgery=surgery)
    return prob, Ld, coupling
end
