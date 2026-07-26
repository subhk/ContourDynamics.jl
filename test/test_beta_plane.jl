using Test
using LinearAlgebra
using StaticArrays
using ContourDynamics

function _copy_contours_for_test(contours)
    return [PVContour(copy(c.nodes), c.pv, c.wrap, copy(c.corners)) for c in contours]
end

# Independent high-precision reference for the sawtooth zonal jet, which solves
# (∂yy − κ²)ψ = −βξ periodically across one staircase step, with u = −∂yψ:
#     u(ξ) = (β/κ²) [ (κ dy/2) cosh(κξ) / sinh(κ dy/2) − 1 ]
# Evaluated in BigFloat so it is free of the cancellation that makes the
# double-precision closed form unusable at small κ·dy. Deliberately NOT a copy
# of the implementation — mirroring it here would make the comparisons below
# tautological.
function _beta_plane_sawtooth_velocity(beta, Ld, domain, n_beta, x)
    dy = 2 * domain.Ly / n_beta
    ξ = mod(x[2] + domain.Ly + dy / 2, dy) - dy / 2
    b = BigFloat(beta); κ = inv(BigFloat(Ld)); D = BigFloat(dy); X = BigFloat(ξ)
    u = -b / κ^2 + b * D * cosh(κ * X) / (2 * κ * sinh(κ * D / 2))
    return SVector(Float64(u), 0.0)
end

@testset "Beta-plane contour QG kernel" begin
    beta = 1.0
    Ld = 1.0
    domain = PeriodicDomain(3.0, 3.0)
    n_beta = 6
    staircase = beta_staircase(beta, domain, n_beta; nodes_per_contour=8)

    kernel = BetaPlaneQGKernel(beta, Ld, staircase)
    @test kernel.beta == beta
    @test kernel.Ld == Ld
    @test length(kernel.reference_contours) == n_beta
    @test kernel.reference_contours[1].nodes == staircase[1].nodes
    @test kernel.reference_contours[1].nodes !== staircase[1].nodes
    @test [c.nodes[1][2] for c in staircase] ≈ [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]

    straight = ContourProblem(kernel, domain, _copy_contours_for_test(staircase))
    vel = fill(SVector(0.0, 0.0), total_nodes(straight))
    velocity!(vel, straight)
    @test maximum(norm, vel) > 1e-3
    @test vel[1] ≈ _beta_plane_sawtooth_velocity(beta, Ld, domain, n_beta, staircase[1].nodes[1]) atol=1e-12

    probe = SVector(0.37, -0.22)
    @test velocity(straight, probe) ≈ _beta_plane_sawtooth_velocity(beta, Ld, domain, n_beta, probe) atol=1e-12

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

@testset "Beta-plane sawtooth jet is accurate at all deformation radii" begin
    # u(ξ) = (β/κ²)[(κ dy/2) cosh(κξ)/sinh(κ dy/2) − 1] is evaluated in double
    # precision as a difference of two O(β/κ²) terms, so it cancels
    # catastrophically once κ·dy is small — exactly the near-barotropic regime
    # (deformation radius ≫ staircase step) that beta-plane studies care about.
    # A large-Ld run must still reproduce the barotropic jet, not collapse to
    # zero.
    beta = 1.3
    n_beta = 4
    dy = 1.0
    Ly = n_beta * dy / 2
    domain = PeriodicDomain(1.0, Ly)
    staircase = beta_staircase(beta, domain, n_beta; nodes_per_contour=4)

    tread_y = -Ly + dy          # midway between two risers: ξ = 0
    riser_y = staircase[1].nodes[1][2]

    for κdy in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 3.0)
        Ld = dy / κdy
        kernel = BetaPlaneQGKernel(beta, Ld, staircase)
        for y in (tread_y, riser_y)
            got = ContourDynamics._beta_plane_sawtooth_velocity(
                kernel, domain, SVector(0.0, y))[1]
            want = _beta_plane_sawtooth_velocity(beta, Ld, domain, n_beta,
                                                 SVector(0.0, y))[1]
            @test isapprox(got, want; rtol=1e-9, atol=1e-14)
        end
    end

    # Barotropic limit (Dritschel & McIntyre 2008): eastward βdy²/12 on the PV
    # risers, westward −βdy²/24 on the treads between them.
    kernel_baro = BetaPlaneQGKernel(beta, dy / 1e-8, staircase)
    @test ContourDynamics._beta_plane_sawtooth_velocity(
        kernel_baro, domain, SVector(0.0, riser_y))[1] ≈ beta * dy^2 / 12 rtol=1e-12
    @test ContourDynamics._beta_plane_sawtooth_velocity(
        kernel_baro, domain, SVector(0.0, tread_y))[1] ≈ -beta * dy^2 / 24 rtol=1e-12

    # No jump where the implementation switches between its small-κ·dy and
    # general branches.
    for κdy in (0.049, 0.051)
        k = BetaPlaneQGKernel(beta, dy / κdy, staircase)
        vals = [ContourDynamics._beta_plane_sawtooth_velocity(k, domain, SVector(0.0, y))[1]
                for y in (tread_y, riser_y)]
        ref = [_beta_plane_sawtooth_velocity(beta, dy / κdy, domain, n_beta, SVector(0.0, y))[1]
               for y in (tread_y, riser_y)]
        @test isapprox(vals, ref; rtol=1e-10)
    end
end
