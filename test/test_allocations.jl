using SpecialFunctions

@testset "Allocation Regressions" begin
    allocation_bytes(f) = (f(); f(); @allocated f())

    @testset "Bessel K0 approximation matches SpecialFunctions" begin
        xs = exp10.(range(-12, 4; length=2_000))
        approx = ContourDynamics._besselk0_approx_scalar.(xs)
        exact = besselk.(0, xs)
        abs_errors = abs.(approx .- exact)
        rel_errors = abs_errors ./ max.(abs.(exact), eps(Float64))

        @test maximum(abs_errors) <= 5e-8
        @test maximum(rel_errors) <= 2e-7

        small_xs = exp10.(range(-12, log10(0.5); length=1_000))
        correction = ContourDynamics._besselk0_correction.(small_xs) .+ log(2.0) .-
                     Base.MathConstants.eulergamma
        expected = besselk.(0, small_xs) .+ log.(small_xs)

        @test maximum(abs.(correction .- expected)) <= 1e-12
    end

    @testset "Unbounded QG velocity is allocation-light after warm-up" begin
        c = circular_patch(0.5, 32, 2π)
        prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c])
        vel = zeros(SVector{2,Float64}, total_nodes(prob))

        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 8_192
    end

    @testset "Unbounded QG energy is allocation-light after warm-up" begin
        c = circular_patch(0.5, 32, 2π)
        prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c])

        alloc = allocation_bytes(() -> energy(prob))
        @test alloc <= 8_192
    end

    @testset "Multilayer velocity avoids quadratic allocation growth" begin
        N = 16
        F = 0.5
        coupling = SMatrix{2,2,Float64}(-F, F, F, -F)
        c1 = PVContour([SVector(0.5cos(2π*k/N), 0.5sin(2π*k/N)) for k in 0:N-1], 1.0)
        c2 = PVContour([SVector(0.5cos(2π*k/N), 1.0 + 0.5sin(2π*k/N)) for k in 0:N-1], -1.0)
        prob = MultiLayerContourProblem(
            MultiLayerQGKernel(SVector(1 / sqrt(2F)), coupling),
            UnboundedDomain(),
            ([c1], [c2]),
        )
        vel = (zeros(SVector{2,Float64}, N), zeros(SVector{2,Float64}, N))

        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 4_000
    end

    @testset "Single-point velocity is allocation-light after warm-up" begin
        c = circular_patch(0.5, 32, 2π)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        x = SVector(0.1, 0.2)

        alloc = allocation_bytes(() -> velocity(prob, x))
        @test alloc <= 64
    end

    @testset "Cached node ranges do not allocate on cache hits" begin
        c = circular_patch(0.5, 32, 2π)
        prob = Problem(; contours=[c], dt=0.01, surgery=:none)
        ContourDynamics._ensure_node_ranges!(prob.stepper, prob.contour_problem)

        alloc = allocation_bytes(() -> ContourDynamics._ensure_node_ranges!(prob.stepper, prob.contour_problem))
        @test alloc <= 64
    end

    @testset "Segment packing reuses curvature buffers" begin
        c = circular_patch(0.5, 32, 2π)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        N = total_nodes(prob)
        ax = Vector{Float64}(undef, N)
        ay = similar(ax)
        bx = similar(ax)
        by = similar(ax)
        pv = similar(ax)
        ka = similar(ax)
        kb = similar(ax)

        alloc = allocation_bytes(() -> ContourDynamics._fill_segment_bufs!(ax, ay, bx, by, pv, ka, kb, prob))
        @test alloc <= 64
    end
end
