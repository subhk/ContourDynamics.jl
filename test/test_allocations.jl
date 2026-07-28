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
        # Julia 1.12's escape analysis elides a 64-byte box that 1.10/1.11
        # still allocate on this path; keep the strict bound where the
        # compiler supports it.
        @test alloc <= (VERSION >= v"1.12" ? 16 : 64)
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

    @testset "Device-resident multilayer velocity reuses workspace" begin
        # The GPU-tagged modal velocity kernel is device-generic, so exercise it on
        # a CPU device. Before task-local workspace caching it allocated 13 device
        # arrays + a ranges Vector (~23.5 KB) every evaluation; it must now stay
        # flat across RK stages (residual is KA CPU-backend launch overhead).
        F = 0.5
        coupling = SMatrix{2,2,Float64}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), coupling)
        dev = ContourDynamics.CPU()
        states = (ContourDynamics.DeviceContourState([circular_patch(1.0, 64, 1.0)], dev),
                  ContourDynamics.DeviceContourState([circular_patch(1.2, 64, -1.0; cx=0.5)], dev))
        vel = zeros(SVector{2,Float64}, 128)
        f() = ContourDynamics._ka_multilayer_velocity_from_states!(vel, states, kernel,
                                                                   UnboundedDomain(), dev)
        alloc = allocation_bytes(f)
        @test alloc <= 12_288
    end

    @testset "Multilayer device host output allocation is node-count independent" begin
        F = 0.5
        coupling = SMatrix{2,2,Float64}(-F, F, F, -F)
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), coupling)
        dev = CPU()

        layers = ([circular_patch(0.45, 24, 1.0)],
                  [circular_patch(0.35, 24, -0.5; cx=0.6)])
        states = ntuple(i -> DeviceContourState(layers[i], dev), 2)
        actual = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 24))
        reference = (similar(actual[1]), similar(actual[2]))
        reference_prob = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)

        ContourDynamics._ka_multilayer_velocity_to_host!(
            actual, states, kernel, UnboundedDomain(), dev)
        velocity!(reference, reference_prob)
        @test maximum(norm.(actual[1] .- reference[1])) <= 1e-12
        @test maximum(norm.(actual[2] .- reference[2])) <= 1e-12

        function host_output_alloc(n)
            layers = ([circular_patch(0.45, n, 1.0)],
                      [circular_patch(0.35, n, -0.5; cx=0.6)])
            states = ntuple(i -> DeviceContourState(layers[i], dev), 2)
            vel = (zeros(SVector{2,Float64}, n), zeros(SVector{2,Float64}, n))
            f() = ContourDynamics._ka_multilayer_velocity_to_host!(
                vel, states, kernel, UnboundedDomain(), dev)
            return allocation_bytes(f)
        end

        small = host_output_alloc(16)
        large = host_output_alloc(256)
        @test small <= 16_384
        # KA's CPU backend has a small fixed launch-bookkeeping variation; the
        # tolerance is far below even one node-sized replacement buffer here.
        @test large <= small + 256 + thread_slack()
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

    @testset "Beta-plane device velocity allocation is independent of node count" begin
        # The reference staircase is packed once into the cached workspace and
        # the live half is repacked in place, so per-call allocation must be
        # fixed KA launch overhead — never proportional to the node count.
        domain = PeriodicDomain(2.0, 2.0)
        staircase = beta_staircase(0.4, domain, 4; nodes_per_contour=8)
        kernel = BetaPlaneQGKernel(0.4, 1.0, staircase)

        function beta_alloc(nnode)
            live = vcat(deepcopy(staircase), [circular_patch(0.25, nnode, 2π)])
            state = ContourDynamics.DeviceContourState(live, CPU())
            N = ContourDynamics._device_state_nnodes(state)
            vel = zeros(SVector{2,Float64}, N)
            return allocation_bytes(() -> ContourDynamics._ka_velocity_from_state!(
                vel, state, kernel, domain, CPU()))
        end

        small = beta_alloc(16)
        large = beta_alloc(256)
        @test small <= 4_096
        @test large <= small + thread_slack()
    end

    @testset "Multilayer device energy packs segments once, not once per mode" begin
        # The modal loop only changes each segment's PV weight, so the segment
        # packing must be hoisted out of it. Packing per mode costs
        # O(n_modes × n_layers) full packings — quadratic in layer count — and
        # throws away an intermediate per-layer copy every time.
        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        states = (ContourDynamics.DeviceContourState([circular_patch(0.3, 16, 1.0)], CPU()),
                  ContourDynamics.DeviceContourState([circular_patch(0.25, 16, -0.5; cx=0.2)], CPU()))

        alloc_u = allocation_bytes(() -> ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, UnboundedDomain(), CPU()))
        @test alloc_u <= 14_000

        domain = PeriodicDomain(2.0, 2.0)
        setup_ewald_cache!(domain, EulerKernel())
        alloc_p = allocation_bytes(() -> ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, domain, CPU()))
        @test alloc_p <= 17_000
    end

    @testset "Device-state energy allocation is node-count independent" begin
        dev = CPU()

        function state_energy_alloc(n)
            contours = [circular_patch(0.4, n, 1.0)]
            state = DeviceContourState(contours, dev)
            f() = ContourDynamics._ka_energy_from_state(
                state, EulerKernel(), UnboundedDomain(), dev)
            return f(), allocation_bytes(f)
        end

        small_energy, small = state_energy_alloc(16)
        large_energy, large = state_energy_alloc(128)
        @test isfinite(small_energy)
        @test isfinite(large_energy)
        @test small <= 16_384
        @test large <= small + 512 + thread_slack()
    end
end
