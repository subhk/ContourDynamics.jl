# Implementation-independent oracles for the periodic velocity path.
#
# The stored arrays in test_periodic_velocity_regression.jl pin the periodic
# implementation against its own past output, so regenerating that baseline
# from the implementation cannot detect an error introduced *before* the
# regeneration. These tests instead compare periodic (Ewald) velocities
# against direct lattice-image sums built from the *unbounded* velocity path —
# a different formula family (Bessel/log segment integrals, no Ewald split) —
# so they remain valid across baseline regenerations.
#
# Configurations are chosen so the image sums converge unconditionally:
#   * QG decays exponentially, so a single patch and a 7×7 image block are
#     exact to machine precision; the tolerance is set by the periodic
#     implementation's algebraically-truncated correction series (~1/n_fourier²).
#   * Euler and SQG kernels decay algebraically, so the test uses a neutral
#     quadrupole (zero net circulation AND zero dipole moment) to remove the
#     conditional-convergence ambiguity of 2D lattice sums; the residual
#     converges absolutely.
@testset "Periodic velocity vs image-sum oracle" begin
    # Velocity at `x` induced by an image of the unbounded problem displaced
    # by `s` equals the unbounded velocity at `x - s`.
    image_sum(prob_unb, x, Lx, Ly, M) = sum(
        velocity(prob_unb, x - SVector(2Lx * mx, 2Ly * my))
        for mx in -M:M, my in -M:M)

    probes = [SVector(0.9, 0.3), SVector(-0.7, 1.1)]

    quadrupole() = [circular_patch(0.3, 64, 1.0; cx=-0.6, cy=-0.6),
                    circular_patch(0.3, 64, -1.0; cx=0.6, cy=-0.6),
                    circular_patch(0.3, 64, 1.0; cx=0.6, cy=0.6),
                    circular_patch(0.3, 64, -1.0; cx=-0.6, cy=0.6)]

    Lx = Ly = 2.0

    @testset "QG single patch (exponential image decay)" begin
        Ld = 0.3
        patch = circular_patch(0.5, 64, 1.0)
        dom = PeriodicDomain(Lx, Ly)
        # The QG correction series converges ~1/n_fourier²; n_fourier=32 puts
        # the implementation within ~5e-5 of the exact periodic solution while
        # the image-sum oracle is exact to ~exp(-16/Ld) ≈ 1e-23.
        clear_ewald_cache!()
        setup_ewald_cache!(dom, QGKernel(Ld); n_fourier=32, n_images=2)
        prob_p = ContourProblem(QGKernel(Ld), dom, [patch])
        prob_u = ContourProblem(QGKernel(Ld), UnboundedDomain(), [patch])
        for x in probes
            vp = velocity(prob_p, x)
            vo = image_sum(prob_u, x, Lx, Ly, 3)
            @test isapprox(vp, vo; rtol=2e-4)
        end
        clear_ewald_cache!()
    end

    @testset "Euler neutral quadrupole" begin
        cs = quadrupole()
        prob_p = ContourProblem(EulerKernel(), PeriodicDomain(Lx, Ly), cs)
        prob_u = ContourProblem(EulerKernel(), UnboundedDomain(), cs)
        for x in probes
            vp = velocity(prob_p, x)
            vo = image_sum(prob_u, x, Lx, Ly, 40)   # converged to ~6e-6
            @test isapprox(vp, vo; rtol=5e-5)
        end
    end

    @testset "SQG neutral quadrupole" begin
        cs = quadrupole()
        prob_p = ContourProblem(SQGKernel(0.02), PeriodicDomain(Lx, Ly), cs)
        prob_u = ContourProblem(SQGKernel(0.02), UnboundedDomain(), cs)
        for x in probes
            vp = velocity(prob_p, x)
            vo = image_sum(prob_u, x, Lx, Ly, 20)   # converged to ~1e-7
            @test isapprox(vp, vo; rtol=1e-6)
        end
    end
end
