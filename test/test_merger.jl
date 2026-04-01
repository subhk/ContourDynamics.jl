using Test, ContourDynamics, StaticArrays

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "Vortex Merger" begin
    R = 1.0
    sep = 3.0
    pv = 1.0

    if extended
        # Full merger test: 64 nodes, 200 steps, check conservation
        N_nodes = 64
        c1_nodes = [SVector(R * cos(2π * i / N_nodes) - sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c2_nodes = [SVector(R * cos(2π * i / N_nodes) + sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c1 = PVContour(c1_nodes, pv)
        c2 = PVContour(c2_nodes, pv)

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        stepper = RK4Stepper(0.05, total_nodes(prob))
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-4, 5)

        circ_initial = circulation(prob)
        evolve!(prob, stepper, params; nsteps=200)
        circ_final = circulation(prob)

        @test circ_final ≈ circ_initial rtol=0.05
        @test length(prob.contours) <= 2
    else
        # Smoke test: verify merger pipeline runs without error
        N_nodes = 32
        c1_nodes = [SVector(R * cos(2π * i / N_nodes) - sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c2_nodes = [SVector(R * cos(2π * i / N_nodes) + sep/2, R * sin(2π * i / N_nodes)) for i in 0:(N_nodes-1)]
        c1 = PVContour(c1_nodes, pv)
        c2 = PVContour(c2_nodes, pv)

        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        stepper = RK4Stepper(0.05, total_nodes(prob))
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-4, 5)

        circ_initial = circulation(prob)
        evolve!(prob, stepper, params; nsteps=10)

        @test total_nodes(prob) > 0
        @test length(prob.contours) >= 1
        @test circulation(prob) ≈ circ_initial rtol=0.15
    end
end
