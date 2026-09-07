using Test, ContourDynamics, OrdinaryDiffEq

@testset "OrdinaryDiffEq Extension" begin
    contour = circular_patch(0.25, 16, 1.0)
    prob = ContourProblem(EulerKernel(), UnboundedDomain(), [contour])

    flat = flatten_nodes(prob)
    @test length(flat) == 2 * total_nodes(prob)
    shifted = flat .+ 0.125
    unflatten_nodes!(prob, shifted)
    @test flatten_nodes(prob) == shifted
    @test_throws DimensionMismatch unflatten_nodes!(prob, shifted[1:end-1])

    sp = SurgeryParams(0.001, 0.01, 0.1, 1e-8, 5)
    @test_throws ArgumentError to_ode_problem(
        prob, (0.0, 1.0); surgery_params=sp, surgery_dt=0.0)
    @test_throws ArgumentError to_ode_problem(
        prob, (0.0, 1.0); surgery_params=sp, surgery_dt=-0.1)
    @test_throws ArgumentError to_ode_problem(
        prob, (0.0, 1.0); surgery_params=sp, surgery_dt=Inf)
    @test_throws ArgumentError to_ode_problem(
        prob, (1.0, 0.0); surgery_params=sp, surgery_dt=0.1)
    @test_throws ArgumentError to_ode_problem(
        prob, (1.0e20, 2.0e20); surgery_params=sp, surgery_dt=1.0)

    wrapped = to_ode_problem(
        prob, (0.0, 1.0); surgery_params=sp, surgery_dt=0.1)
    @test keys(wrapped) == (:ode_prob, :callback)
    @test wrapped.ode_prob.p === prob

    plain = to_ode_problem(prob, (0.0, 0.1))
    @test plain.p === prob

    @testset "Fixed-step solve" begin
        solve_prob = ContourProblem(
            EulerKernel(), UnboundedDomain(), [circular_patch(0.25, 16, 1.0)])
        u0 = flatten_nodes(solve_prob)
        ode_prob = to_ode_problem(solve_prob, (0.0, 0.01))

        sol = solve(ode_prob, Tsit5(); dt=0.01, adaptive=false)

        @test sol.t[end] == 0.01
        @test length(sol.u[end]) == length(u0)
        @test all(isfinite, sol.u[end])
    end

    @testset "Fixed-step solve with surgery callback" begin
        resizing_sp = SurgeryParams(0.001, 0.01, 0.1, 1e-8, 1)
        solve_prob = ContourProblem(
            EulerKernel(), UnboundedDomain(), [circular_patch(1.0, 8, 1.0)])
        u0 = flatten_nodes(solve_prob)
        wrapped = to_ode_problem(
            solve_prob, (0.0, 0.02);
            surgery_params=resizing_sp, surgery_dt=0.01)

        sol = solve(
            wrapped.ode_prob, Tsit5(); dt=0.01, adaptive=false,
            callback=wrapped.callback)

        @test sol.t[end] == 0.02
        @test length(sol.u[end]) > length(u0)
        @test length(sol.u[end]) == length(flatten_nodes(solve_prob))
        @test all(isfinite, sol.u[end])
    end
end
