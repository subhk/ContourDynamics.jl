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

    @testset "Euler Kernel" begin
        c = circular_patch(1.0, 128, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2, Float64}, total_nodes(prob))
        velocity!(vel, prob)

        # Velocity at each node: tangential, magnitude = pv*R/2 = 0.5
        expected_speed = 0.5
        for i in 1:nnodes(c)
            speed = sqrt(vel[i][1]^2 + vel[i][2]^2)
            @test speed ≈ expected_speed rtol=0.02
        end

        # Check tangential direction: velocity perpendicular to position
        for i in 1:nnodes(c)
            pos = c.nodes[i]
            @test abs(vel[i][1]*pos[1] + vel[i][2]*pos[2]) < 0.02
        end
    end

    @testset "QG Kernel" begin
        c = circular_patch(1.0, 128, 1.0)
        prob_euler = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        prob_qg = ContourProblem(QGKernel(100.0), UnboundedDomain(), [c])

        vel_euler = zeros(SVector{2, Float64}, 128)
        vel_qg = zeros(SVector{2, Float64}, 128)
        velocity!(vel_euler, prob_euler)
        velocity!(vel_qg, prob_qg)

        # Large Ld → QG approaches Euler
        for i in 1:128
            @test vel_qg[i] ≈ vel_euler[i] rtol=0.05
        end

        # Small Ld → QG velocity weaker than Euler
        prob_qg_small = ContourProblem(QGKernel(0.5), UnboundedDomain(), [c])
        vel_qg_small = zeros(SVector{2, Float64}, 128)
        velocity!(vel_qg_small, prob_qg_small)

        euler_speed = sqrt(vel_euler[1][1]^2 + vel_euler[1][2]^2)
        qg_speed = sqrt(vel_qg_small[1][1]^2 + vel_qg_small[1][2]^2)
        @test qg_speed < euler_speed
    end

    @testset "Per-Contour Diagnostics" begin
        c = circular_patch(1.0, 256, 1.0)
        @test vortex_area(c) ≈ π rtol=2e-4

        cx = centroid(c)
        @test abs(cx[1]) < 1e-10
        @test abs(cx[2]) < 1e-10

        e = elliptical_patch(2.0, 1.0, 256, 1.0)
        @test vortex_area(e) ≈ 2π rtol=2e-4

        ratio, angle = ellipse_moments(e)
        @test ratio ≈ 2.0 rtol=0.01
        @test abs(angle) < 0.05
    end

    @testset "Problem-Level Diagnostics" begin
        c = circular_patch(1.0, 128, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])

        @test circulation(prob) ≈ π rtol=1e-3
        @test enstrophy(prob) ≈ π / 2 rtol=1e-3

        E = energy(prob)
        @test E > 0
        @test isfinite(E)

        L = angular_momentum(prob)
        @test L ≈ π / 2 rtol=1e-3
    end

    @testset "Node Management" begin
        nodes = SVector{2, Float64}[
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.002, 0.0),
            SVector(1.0, 0.0), SVector(1.0, 1.0), SVector(0.0, 1.0),
        ]
        c = PVContour(nodes, 1.0)
        params = SurgeryParams(0.01, 0.01, 0.05, 1e-6, 10)

        c_new = remesh(c, params)

        for i in 1:nnodes(c_new)
            j = mod1(i + 1, nnodes(c_new))
            d = c_new.nodes[j] - c_new.nodes[i]
            spacing = sqrt(d[1]^2 + d[2]^2)
            @test spacing >= params.mu * 0.9
            @test spacing <= params.Delta_max * 1.1
        end

        @test vortex_area(c_new) ≈ vortex_area(c) rtol=0.1
    end

    include("test_surgery.jl")
end
