using Test, ContourDynamics

extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"

@testset "Kirchhoff Ellipse" begin
    a = 2.0
    b = 1.0
    pv = 1.0
    N_nodes = extended ? 64 : 32

    c = elliptical_patch(a, b, N_nodes, pv)
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

    Omega = a * b * pv / (a + b)^2
    T_period = 2π / Omega

    nsteps = extended ? 500 : 100
    dt = T_period / nsteps
    stepper = RK4Stepper(dt, total_nodes(prob))
    params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)

    initial_nodes = copy(prob.contours[1].nodes)
    initial_area = vortex_area(prob.contours[1])
    initial_circ = circulation(prob)

    evolve!(prob, stepper, params; nsteps=nsteps)

    final_area = vortex_area(prob.contours[1])
    final_circ = circulation(prob)

    @test final_area ≈ initial_area rtol=1e-3
    @test final_circ ≈ initial_circ rtol=1e-4

    # After one full period, nodes should return near initial positions
    node_tol = extended ? 0.08 : 0.10
    for i in 1:N_nodes
        @test prob.contours[1].nodes[i] ≈ initial_nodes[i] atol=node_tol
    end

    ratio_final, _ = ellipse_moments(prob.contours[1])
    @test ratio_final ≈ a / b rtol=0.05
end
