using Test

include("test_utils.jl")

@testset "ContourDynamics.jl" begin
    @testset "Core Types" begin
        # PVContour construction
        c = circular_patch(1.0, 64, 1.0)
        @test nnodes(c) == 64
        @test c.pv == 1.0

        # EulerKernel
        k = EulerKernel()
        @test k isa AbstractKernel

        # QGKernel validation
        @test_throws ArgumentError QGKernel(-1.0)
        qg = QGKernel(2.5)
        @test qg.Ld == 2.5

        # Domains
        d = UnboundedDomain()
        @test d isa AbstractDomain
        pd = PeriodicDomain(π, π)
        @test pd.Lx == π
        @test_throws ArgumentError PeriodicDomain(-1.0, 1.0)

        # ContourProblem
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test total_nodes(prob) == 64

        # SurgeryParams validation
        sp = SurgeryParams(0.01, 0.005, 0.1, 1e-6, 10)
        @test sp.n_surgery == 10
        @test_throws ArgumentError SurgeryParams(0.01, 0.005, 0.003, 1e-6, 10)  # Delta_max < mu

        # RK4Stepper construction
        rk = RK4Stepper(0.01, 64)
        @test rk.dt == 0.01
        @test length(rk.k1) == 64
    end
end
