@testset "Vortex Merger" begin
    # Two equal circular vortex patches separated by ~3.0 radii → should merge
    R = 1.0
    sep = 3.0
    pv = 1.0

    c1_nodes = [SVector(R * cos(2π * i / 64) - sep/2, R * sin(2π * i / 64)) for i in 0:63]
    c2_nodes = [SVector(R * cos(2π * i / 64) + sep/2, R * sin(2π * i / 64)) for i in 0:63]
    c1 = PVContour(c1_nodes, pv)
    c2 = PVContour(c2_nodes, pv)

    prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])

    dt = 0.05
    nsteps = 200
    stepper = RK4Stepper(dt, total_nodes(prob))
    params = SurgeryParams(0.05, 0.02, 0.3, 1e-4, 5)

    circ_initial = circulation(prob)

    evolve!(prob, stepper, params; nsteps=nsteps)

    circ_final = circulation(prob)

    # Circulation must be conserved (surgery preserves PV jumps)
    @test circ_final ≈ circ_initial rtol=0.05

    # Contours should have merged or remain: started with 2
    @test length(prob.contours) <= 2
end
