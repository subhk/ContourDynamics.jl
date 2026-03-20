@testset "Conservation" begin
    @testset "Circular Patch Steady State (Euler)" begin
        # A single circular patch is an exact steady state of 2D Euler
        # All diagnostics should be conserved to machine precision (modulo RK4 truncation)
        R = 1.0
        pv_val = 1.0
        c = circular_patch(R, 128, pv_val)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        dt = 0.01
        nsteps = 10000  # 10^4 timesteps per spec
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        E0 = energy(prob)
        A0 = vortex_area(prob.contours[1])
        G0 = circulation(prob)
        c0 = centroid(prob.contours[1])

        evolve!(prob, stepper, params; nsteps=nsteps)

        E1 = energy(prob)
        A1 = vortex_area(prob.contours[1])
        G1 = circulation(prob)
        c1 = centroid(prob.contours[1])

        # Energy drift should be O(dt^4) for RK4
        @test abs(E1 - E0) / abs(E0) < 1e-6

        # Area should be conserved very precisely
        @test A1 ≈ A0 rtol=1e-8

        # Circulation conserved
        @test G1 ≈ G0 rtol=1e-8

        # Centroid should remain at origin
        @test sqrt(c1[1]^2 + c1[2]^2) < 1e-8
    end

    @testset "Circular Patch Steady State (QG)" begin
        R = 1.0
        pv_val = 1.0
        c = circular_patch(R, 128, pv_val)
        prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])

        dt = 0.01
        nsteps = 500
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        A0 = vortex_area(prob.contours[1])
        G0 = circulation(prob)

        evolve!(prob, stepper, params; nsteps=nsteps)

        A1 = vortex_area(prob.contours[1])
        G1 = circulation(prob)

        @test A1 ≈ A0 rtol=1e-8
        @test G1 ≈ G0 rtol=1e-8
    end
end
