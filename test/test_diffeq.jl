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

    @testset "Solving the same surgery problem twice gives the same result" begin
        base = ContourProblem(EulerKernel(), UnboundedDomain(),
                              [circular_patch(1.0, 64, 1.0),
                               circular_patch(0.3, 32, -1.0; cx=3.0)])
        wrapped = to_ode_problem(base, (0.0, 0.2);
                                 surgery_params=SurgeryParams(0.005, 0.05, 0.2, 1e-8, 5),
                                 surgery_dt=0.1)
        solve_once() = solve(wrapped.ode_prob, Tsit5(); dt=0.01, adaptive=false,
                             callback=wrapped.callback)
        first_solution = solve_once()
        first_final = copy(first_solution.u[end])
        first_circulation = circulation(base)
        second_solution = solve_once()
        @test second_solution.u[end] == first_final
        @test circulation(base) ≈ first_circulation rtol=1e-12
        @test_throws DimensionMismatch unflatten_nodes!(base, vcat(first_final, 0.0, 0.0))
    end

    @testset "Problem wrapper keeps its bundled surgery" begin
        p = Problem(; contours=[circular_patch(1.0, 8, 1.0)], dt=0.01)
        wrapped = to_ode_problem(p, (0.0, 0.05))
        @test keys(wrapped) == (:ode_prob, :callback)
        sol = solve(wrapped.ode_prob, Tsit5(); dt=0.01, adaptive=false,
                    callback=wrapped.callback)
        # Surgery (every n_surgery = 5 steps of dt) refined the coarse polygon.
        @test length(sol.u[end]) > 2 * 8
        plain = to_ode_problem(p, (0.0, 0.05); surgery_params=nothing)
        @test plain isa ODEProblem
    end
end
