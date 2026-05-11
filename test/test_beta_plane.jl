using Test
using LinearAlgebra
using StaticArrays
using ContourDynamics

function _copy_contours_for_test(contours)
    return [PVContour(copy(c.nodes), c.pv, c.wrap, copy(c.corners)) for c in contours]
end

@testset "Beta-plane contour QG kernel" begin
    beta = 1.0
    Ld = 1.0
    domain = PeriodicDomain(3.0, 3.0)
    staircase = beta_staircase(beta, domain, 6; nodes_per_contour=8)

    kernel = BetaPlaneQGKernel(beta, Ld, staircase)
    @test kernel.beta == beta
    @test kernel.Ld == Ld
    @test length(kernel.reference_contours) == length(staircase)
    @test kernel.reference_contours[1].nodes == staircase[1].nodes
    @test kernel.reference_contours[1].nodes !== staircase[1].nodes

    straight = ContourProblem(kernel, domain, _copy_contours_for_test(staircase))
    vel = fill(SVector(0.0, 0.0), total_nodes(straight))
    velocity!(vel, straight)
    @test maximum(norm, vel) < 1e-12
    @test norm(velocity(straight, SVector(0.37, -0.22))) < 1e-12

    deformed_staircase = _copy_contours_for_test(staircase)
    deformed_staircase[2].nodes[3] += SVector(0.0, 0.1)
    deformed = ContourProblem(kernel, domain, deformed_staircase)
    deformed_vel = fill(SVector(0.0, 0.0), total_nodes(deformed))
    velocity!(deformed_vel, deformed)
    @test maximum(norm, deformed_vel) > 1e-6

    wrapped = Problem(; contours=_copy_contours_for_test(staircase),
                       dt=0.01,
                       kernel=:beta_plane_qg,
                       beta=beta,
                       Ld=Ld,
                       domain=:periodic,
                       Lx=domain.Lx,
                       Ly=domain.Ly,
                       surgery=:none)
    @test ContourDynamics.kernel(wrapped) isa BetaPlaneQGKernel
    @test length(ContourDynamics.kernel(wrapped).reference_contours) == length(staircase)

    vortex = circular_patch(0.3, 16, 1.0)
    @test_throws ArgumentError Problem(; contours=[vortex],
                                        dt=0.01,
                                        kernel=:beta_plane_qg,
                                        beta=beta,
                                        Ld=Ld,
                                        domain=:periodic,
                                        Lx=domain.Lx,
                                        Ly=domain.Ly,
                                        surgery=:none)

    wrong_beta_staircase = beta_staircase(2beta, domain, 6; nodes_per_contour=8)
    @test_throws ArgumentError Problem(; contours=wrong_beta_staircase,
                                        dt=0.01,
                                        kernel=:beta_plane_qg,
                                        beta=beta,
                                        Ld=Ld,
                                        domain=:periodic,
                                        Lx=domain.Lx,
                                        Ly=domain.Ly,
                                        surgery=:none)

    shifted_staircase = _copy_contours_for_test(staircase)
    shifted_staircase[2] = PVContour([node + SVector(0.0, 0.25) for node in shifted_staircase[2].nodes],
                                     shifted_staircase[2].pv,
                                     shifted_staircase[2].wrap,
                                     copy(shifted_staircase[2].corners))
    @test_throws ArgumentError Problem(; contours=shifted_staircase,
                                        dt=0.01,
                                        kernel=:beta_plane_qg,
                                        beta=beta,
                                        Ld=Ld,
                                        domain=:periodic,
                                        Lx=domain.Lx,
                                        Ly=domain.Ly,
                                        surgery=:none)

    @test_throws ArgumentError ContourProblem(kernel, UnboundedDomain(),
                                              _copy_contours_for_test(staircase))
end
