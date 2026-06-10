using SpecialFunctions
using LinearAlgebra: norm

@testset "Allocation Regressions" begin
    allocation_bytes(f) = (f(); f(); @allocated f())

    # The tight bounds below assume serial execution. On the threaded velocity
    # branch (node count >= 128) `Threads.@threads` adds O(nthreads) task-spawn
    # overhead, which is not a per-node leak — allow slack for it when the suite
    # runs multithreaded so the serial regression check stays strict at -t1.
    thread_slack(per_thread=2048) = Threads.nthreads() == 1 ? 0 : Threads.nthreads() * per_thread

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
        @test alloc <= 64
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
        @test alloc <= 64
    end

    @testset "Single-layer direct velocity reuses curvature scratch" begin
        N = 64
        c = circular_patch(0.5, N, 2π)
        prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c])
        vel = zeros(SVector{2,Float64}, total_nodes(prob))

        @test @inferred(velocity!(vel, prob)) === vel
        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 128
    end

    @testset "Unbounded Euler CPU velocity is allocation-free after warm-up" begin
        N = 64
        c = circular_patch(0.5, N, 2π)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c])
        vel = zeros(SVector{2,Float64}, total_nodes(prob))

        # CPU Euler/Unbounded must take the direct evaluator (no KA launch),
        # matching the QG/SQG paths. The KA result is the reference.
        reference = zeros(SVector{2,Float64}, total_nodes(prob))
        ContourDynamics._ka_velocity!(reference, prob, prob.dev)

        @test @inferred(velocity!(vel, prob)) === vel
        @test maximum(norm.(vel .- reference)) <= 1e-12

        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 128
    end

    @testset "Periodic Euler velocity does not box the Ewald cache lookup" begin
        N = 64
        domain = PeriodicDomain(4.0, 4.0)
        setup_ewald_cache!(domain, EulerKernel())
        c = circular_patch(0.5, N, 2π)
        prob = ContourProblem(EulerKernel(), domain, [c])
        vel = zeros(SVector{2,Float64}, total_nodes(prob))

        # The cache fast-path read must not allocate a `lock(f) do ... end`
        # closure on warm cache hits.
        @test @inferred(velocity!(vel, prob)) === vel
        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 16
    end

    @testset "Beta-plane velocity reuses live and reference curvature scratch" begin
        N = 8
        domain = PeriodicDomain(4.0, 4.0)
        setup_ewald_cache!(domain, QGKernel(1.0); n_fourier=1, n_images=0)
        reference = beta_staircase(0.2, domain, 2)
        vortex = circular_patch(0.4, N, 2π)
        prob = ContourProblem(BetaPlaneQGKernel(0.2, 1.0, reference),
                              domain, vcat(reference, [vortex]))
        vel = zeros(SVector{2,Float64}, total_nodes(prob))

        # total_nodes here is 136 (>= 128), so this hits the threaded branch when
        # the suite runs multithreaded — allow for @threads task overhead.
        @test @inferred(velocity!(vel, prob)) === vel
        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 256 + thread_slack()
    end

    @testset "Multilayer CPU velocity reuses modal scratch" begin
        N = 64
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

        @test @inferred(velocity!(vel, prob)) === vel
        alloc = allocation_bytes(() -> velocity!(vel, prob))
        @test alloc <= 32
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
