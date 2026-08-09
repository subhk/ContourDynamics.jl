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
end
