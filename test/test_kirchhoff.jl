@testset "Kirchhoff Ellipse" begin
    # Kirchhoff ellipse: an elliptical vortex patch in 2D Euler
    # rotates steadily with angular velocity Ω = ab/(a+b)² * pv
    # Period T = 2π(a+b)²/(ab * pv)

    a = 2.0  # semi-major axis
    b = 1.0  # semi-minor axis
    pv = 1.0
    N_nodes = 128

    c = elliptical_patch(a, b, N_nodes, pv)
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

    # Analytic angular velocity
    Omega = a * b * pv / (a + b)^2
    T_period = 2π / Omega

    extended = get(ENV, "CONTOURDYNAMICS_EXTENDED_TESTS", "false") == "true"
    nsteps = extended ? 4000 : 400
    dt = T_period / nsteps
    stepper = RK4Stepper(dt, total_nodes(prob))
    params = SurgeryParams(0.001, 0.01, 0.2, 1e-8, nsteps + 1)  # no surgery during test

    # Record initial state
    initial_nodes = copy(prob.contours[1].nodes)
    initial_area = vortex_area(prob.contours[1])
    initial_circ = circulation(prob)

    evolve!(prob, stepper, params; nsteps=nsteps)

    final_area = vortex_area(prob.contours[1])
    final_circ = circulation(prob)

    # Area must be conserved
    @test final_area ≈ initial_area rtol=1e-4

    # Circulation must be conserved
    @test final_circ ≈ initial_circ rtol=1e-6

    # After one full period, nodes should return to initial positions
    node_tol = extended ? 0.05 : 0.1
    for i in 1:N_nodes
        @test prob.contours[1].nodes[i] ≈ initial_nodes[i] atol=node_tol
    end

    # Aspect ratio should be preserved
    ratio_final, _ = ellipse_moments(prob.contours[1])
    @test ratio_final ≈ a / b rtol=0.05
end
