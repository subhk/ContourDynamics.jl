extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "Conservation" begin
    @testset "Circular Patch Steady State (Euler)" begin
        R = 1.0
        pv_val = 1.0
        N_nodes = extended ? 128 : 32
        c = circular_patch(R, N_nodes, pv_val)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        dt = 0.01
        nsteps = extended ? 10000 : 50
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

        energy_tol = extended ? 1e-7 : 1e-4
        @test abs(E1 - E0) / abs(E0) < energy_tol
        @test A1 ≈ A0 rtol=1e-6
        @test G1 ≈ G0 rtol=1e-6
        @test sqrt((c1[1] - c0[1])^2 + (c1[2] - c0[2])^2) < 1e-6
    end

    @testset "Circular Patch Steady State (QG)" begin
        R = 1.0
        pv_val = 1.0
        N_nodes = extended ? 128 : 32
        c = circular_patch(R, N_nodes, pv_val)
        prob = ContourProblem(QGKernel(2.0), UnboundedDomain(), [c])

        dt = 0.01
        nsteps = extended ? 500 : 20
        stepper = RK4Stepper(dt, total_nodes(prob))
        params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

        A0 = vortex_area(prob.contours[1])
        G0 = circulation(prob)
        E0 = energy(prob)

        evolve!(prob, stepper, params; nsteps=nsteps)

        A1 = vortex_area(prob.contours[1])
        G1 = circulation(prob)
        E1 = energy(prob)

        @test A1 ≈ A0 rtol=1e-6
        @test G1 ≈ G0 rtol=1e-6
        qg_energy_tol = extended ? 1e-6 : 1e-4
        @test abs(E1 - E0) / abs(E0) < qg_energy_tol
    end
end
