using ContourDynamics
using StaticArrays
using Test

# Guard against double-include when run from runtests.jl
@isdefined(circular_patch) || include("test_utils.jl")

# Disable scalar indexing on GPU arrays to catch accidental cu_array[i] access.
# Only activates when CUDA is actually loaded.
try
    using CUDA
    CUDA.allowscalar(false)
catch
end

@testset "Device abstraction" begin
    @testset "ContourProblem defaults to CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        @test prob.dev === CPU()
    end

    @testset "ContourProblem accepts dev keyword" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        @test prob.dev === CPU()
    end

    @testset "GPU() without CUDA gives helpful error" begin
        @test_throws ErrorException device_array(GPU())
    end

    @testset "GPU() with unsupported kernel/domain errors at construction" begin
        c = circular_patch(0.5, 32, 1.0)
        @test_throws ArgumentError ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
        @test_throws ArgumentError ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
    end

    @testset "CPU device_array returns Array" begin
        @test device_array(CPU()) === Array
    end

    @testset "to_cpu is identity for Array" begin
        x = [1.0, 2.0, 3.0]
        @test to_cpu(x) === x
    end

    @testset "device_zeros CPU" begin
        z = device_zeros(CPU(), Float64, 5)
        @test z == zeros(5)
        @test z isa Vector{Float64}
    end

    @testset "Full evolve! with dev=CPU() matches existing behavior" begin
        c = circular_patch(0.5, 64, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        stepper = RK4Stepper(0.01, total_nodes(prob))
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        circ_before = circulation(prob)
        evolve!(prob, stepper, params; nsteps=10)
        circ_after = circulation(prob)
        @test isapprox(circ_before, circ_after; rtol=1e-6)
    end

    @testset "pack_segments round-trip" begin
        c1 = circular_patch(0.5, 16, 1.0)
        c2 = circular_patch(0.3, 8, -0.5)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c1, c2])
        seg = ContourDynamics.pack_segments(prob, CPU())
        @test length(seg.ax) == total_nodes(prob)
        @test length(seg.pv) == total_nodes(prob)
        # First segment of c1
        @test seg.ax[1] ≈ c1.nodes[1][1]
        @test seg.ay[1] ≈ c1.nodes[1][2]
        @test seg.bx[1] ≈ c1.nodes[2][1]
        @test seg.by[1] ≈ c1.nodes[2][2]
        @test seg.pv[1] ≈ c1.pv
    end

    @testset "KA Euler velocity matches direct CPU" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        N = total_nodes(prob)

        # CPU reference
        vel_cpu = zeros(SVector{2,Float64}, N)
        ContourDynamics._direct_velocity!(vel_cpu, prob)

        # KA CPU kernel path
        vel_ka_x = zeros(Float64, N)
        vel_ka_y = zeros(Float64, N)
        seg = ContourDynamics.pack_segments(prob, CPU())
        target_x = Float64[c.nodes[i][1] for c in prob.contours for i in 1:nnodes(c)]
        target_y = Float64[c.nodes[i][2] for c in prob.contours for i in 1:nnodes(c)]
        ContourDynamics._ka_euler_velocity!(vel_ka_x, vel_ka_y, target_x, target_y, seg, CPU())

        for i in 1:N
            @test isapprox(vel_ka_x[i], vel_cpu[i][1]; atol=1e-12)
            @test isapprox(vel_ka_y[i], vel_cpu[i][2]; atol=1e-12)
        end
    end

    @testset "CPU velocity! uses KA path without changing results" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        N = total_nodes(prob)

        vel_ref = zeros(SVector{2,Float64}, N)
        vel = similar(vel_ref)
        ContourDynamics._direct_velocity!(vel_ref, prob)
        velocity!(vel, prob)

        for i in 1:N
            @test isapprox(vel[i][1], vel_ref[i][1]; atol=1e-12)
            @test isapprox(vel[i][2], vel_ref[i][2]; atol=1e-12)
        end
    end
end
