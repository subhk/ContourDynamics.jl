using Test, ContourDynamics, StaticArrays

@testset "Edge Cases & Error Handling" begin
    @testset "Invalid PVContour construction" begin
        # Fewer than 3 nodes should still construct (remesh/surgery handle cleanup)
        c2 = PVContour([SVector(0.0, 0.0), SVector(1.0, 0.0)], 1.0)
        @test nnodes(c2) == 2

        c1 = PVContour([SVector(0.0, 0.0)], 1.0)
        @test nnodes(c1) == 1

        # Empty node vector
        c0 = PVContour(SVector{2,Float64}[], 1.0)
        @test nnodes(c0) == 0
    end

    @testset "Invalid SurgeryParams" begin
        # Delta_max < mu
        @test_throws ArgumentError SurgeryParams(0.01, 0.1, 0.05, 1e-6, 10)
        # Negative values
        @test_throws ArgumentError SurgeryParams(-0.01, 0.05, 0.1, 1e-6, 10)
        @test_throws ArgumentError SurgeryParams(0.01, -0.05, 0.1, 1e-6, 10)
        @test_throws ArgumentError SurgeryParams(0.01, 0.05, 0.1, -1e-6, 10)
        # n_surgery < 1
        @test_throws ArgumentError SurgeryParams(0.01, 0.05, 0.1, 1e-6, 0)
    end

    @testset "Invalid kernel parameters" begin
        @test_throws ArgumentError QGKernel(0.0)
        @test_throws ArgumentError QGKernel(-1.0)
        @test_throws ArgumentError SQGKernel(-0.1)
    end

    @testset "Invalid domain parameters" begin
        @test_throws ArgumentError PeriodicDomain(0.0, 1.0)
        @test_throws ArgumentError PeriodicDomain(1.0, 0.0)
        @test_throws ArgumentError PeriodicDomain(-1.0, 1.0)
    end

    @testset "Degenerate contour diagnostics" begin
        # All nodes at the same point — degenerate contour
        nodes = [SVector(1.0, 1.0), SVector(1.0, 1.0), SVector(1.0, 1.0)]
        c = PVContour(nodes, 1.0)
        @test vortex_area(c) ≈ 0.0 atol=1e-15
        ctr = centroid(c)
        @test ctr ≈ SVector(1.0, 1.0) atol=1e-15

        # ellipse_moments should return (1.0, 0.0) for degenerate contour
        ratio, angle = ellipse_moments(c)
        @test ratio == 1.0
        @test angle == 0.0
    end

    @testset "Collinear nodes" begin
        # All nodes on a line — zero area
        nodes = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(2.0, 0.0)]
        c = PVContour(nodes, 1.0)
        @test abs(vortex_area(c)) < 1e-15
    end

    @testset "Very small contour" begin
        # Contour much smaller than surgery params → filament removal
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(1e-6, 0.0), SVector(5e-7, 1e-6)
        ], 1.0)
        normal = circular_patch(1.0, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [normal, tiny])
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-8, 10)
        surgery!(prob, params)
        @test length(prob.contours) == 1  # tiny removed
    end

    @testset "Velocity with zero-area contour" begin
        # Degenerate contour shouldn't produce NaN velocities
        nodes = [SVector(0.0, 0.0), SVector(1e-15, 0.0), SVector(0.0, 1e-15)]
        c = PVContour(nodes, 1.0)
        normal = circular_patch(1.0, 16, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [normal, c])
        vel = zeros(SVector{2, Float64}, total_nodes(prob))
        velocity!(vel, prob)
        @test all(v -> all(isfinite, v), vel)
    end

    @testset "Remesh edge cases" begin
        # Contour with uniform spacing within [mu, Delta_max] — should be unchanged
        c = circular_patch(1.0, 64, 1.0)
        params = SurgeryParams(0.001, 0.05, 0.2, 1e-6, 10)
        c_new = remesh(c, params)
        @test nnodes(c_new) == nnodes(c)

        # Contour with very few nodes
        c3 = PVContour([SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(0.5, 1.0)], 1.0)
        params_wide = SurgeryParams(0.001, 0.01, 5.0, 1e-6, 10)
        c3_new = remesh(c3, params_wide)
        @test nnodes(c3_new) >= 3
    end

    @testset "Float32 support" begin
        # Verify core operations work with Float32
        c = PVContour([SVector{2,Float32}(cos(2π*i/16), sin(2π*i/16)) for i in 0:15], 1.0f0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2, Float32}, 16)
        velocity!(vel, prob)
        @test all(v -> all(isfinite, v), vel)
        @test eltype(vel) == SVector{2, Float32}

        @test isfinite(vortex_area(c))
        @test isfinite(circulation(prob))
    end

    @testset "Negative PV values" begin
        # CW contour with negative PV — should have negative area
        c = circular_patch(1.0, 64, -1.0)
        @test c.pv == -1.0
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2, Float64}, 64)
        velocity!(vel, prob)
        # Velocity should be reversed compared to positive PV
        c_pos = circular_patch(1.0, 64, 1.0)
        prob_pos = ContourProblem(EulerKernel(), UnboundedDomain(), [c_pos])
        vel_pos = zeros(SVector{2, Float64}, 64)
        velocity!(vel_pos, prob_pos)
        @test vel[1] ≈ -vel_pos[1] rtol=1e-10
    end

    @testset "beta_staircase validation" begin
        domain = PeriodicDomain(1.0, 1.0)
        @test_throws ArgumentError beta_staircase(1.0, domain, 1)   # n_steps < 2
        @test_throws ArgumentError beta_staircase(1.0, domain, 4; nodes_per_contour=2)  # < 3
    end
end
