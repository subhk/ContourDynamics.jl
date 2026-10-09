using ContourDynamics
using StaticArrays
using Test
using Logging

# Guard against double-include when run from runtests.jl
@isdefined(circular_patch) || include("test_utils.jl")
@isdefined(_full_rewrite_output_layout) || include("device_rewrite_layout_oracle.jl")

# Disable scalar indexing on GPU arrays to catch accidental cu_array[i] access.
# Only activates when CUDA is actually loaded.
const _TEST_CUDA_LOADED = Ref(false)
try
    using CUDA
    CUDA.allowscalar(false)
    _TEST_CUDA_LOADED[] = true
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

    @testset "CUDA extension hooks do not overwrite core methods" begin
        if !_TEST_CUDA_LOADED[]
            dispatch_device(f, types) =
                Base.unwrap_unionall(which(f, types).sig).parameters[2]
            @test dispatch_device(device_array, Tuple{GPU}) === AbstractDevice
            @test dispatch_device(device_zeros, Tuple{GPU,Type{Float64},Int}) === AbstractDevice
            @test dispatch_device(to_device, Tuple{GPU,Vector{Float64}}) === AbstractDevice
            @test dispatch_device(ContourDynamics._ka_backend, Tuple{GPU}) === AbstractDevice
        end
    end

    @testset "GPU velocity! without CUDA gives helpful error" begin
        c = circular_patch(0.5, 32, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
            vel = zeros(SVector{2,Float64}, total_nodes(prob))
            velocity!(vel, prob)
            @test all(v -> all(isfinite, v), vel)
        else
            @test_throws ErrorException ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        end
    end

    @testset "GPU() accepts supported single-layer periodic and unbounded cases" begin
        c = circular_patch(0.5, 32, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa ContourDynamics.DeviceContourState

            prob_periodic = ContourProblem(QGKernel(1.0), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic.dev === GPU()
            @test prob_periodic.device_state isa ContourDynamics.DeviceContourState

            prob_periodic_euler = ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic_euler.dev === GPU()
            @test prob_periodic_euler.device_state isa ContourDynamics.DeviceContourState

            prob = ContourProblem(SQGKernel(0.01), UnboundedDomain(), [c]; dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa ContourDynamics.DeviceContourState
            prob_periodic_sqg = ContourProblem(SQGKernel(0.01), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test prob_periodic_sqg.dev === GPU()
            @test prob_periodic_sqg.device_state isa ContourDynamics.DeviceContourState
        else
            @test_throws ErrorException ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(QGKernel(1.0), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(EulerKernel(), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(SQGKernel(0.01), UnboundedDomain(), [c]; dev=GPU())
            @test_throws ErrorException ContourProblem(SQGKernel(0.01), PeriodicDomain(1.0, 1.0), [c]; dev=GPU())
        end
    end

    @testset "GPU() accepts multi-layer QG problems" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c = circular_patch(0.5, 32, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                            ([c], PVContour{Float64}[]); dev=GPU())
            @test prob.dev === GPU()
            @test prob.device_state isa Tuple
            @test all(s -> s isa ContourDynamics.DeviceContourState, prob.device_state)
        else
            @test_throws ErrorException MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                                                 ([c], PVContour{Float64}[]); dev=GPU())
        end
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

    @testset "DeviceContourState packs and materializes flat topology" begin
        c1 = circular_patch(0.5, 8, 1.25)
        c2_nodes = [SVector(0.1, 0.2), SVector(0.7, 0.2),
                    SVector(0.7, 0.8), SVector(0.1, 0.8)]
        c2 = PVContour(c2_nodes, -0.5, SVector(1.0, 0.0),
                       Bool[true, false, true, false])
        source = [c1, c2]
        state = ContourDynamics.DeviceContourState(source, CPU())

        expected_x = Float64[p[1] for c in source for p in c.nodes]
        expected_y = Float64[p[2] for c in source for p in c.nodes]
        @test to_cpu(state.x) == expected_x
        @test to_cpu(state.y) == expected_y
        @test to_cpu(state.pv) == [c1.pv, c2.pv]
        @test to_cpu(state.wrapx) == [c1.wrap[1], c2.wrap[1]]
        @test to_cpu(state.wrapy) == [c1.wrap[2], c2.wrap[2]]
        @test to_cpu(state.offsets) == [1, nnodes(c1) + 1]
        @test to_cpu(state.lengths) == [nnodes(c1), nnodes(c2)]
        @test to_cpu(state.contour_of_node) == vcat(fill(1, nnodes(c1)), fill(2, nnodes(c2)))
        @test to_cpu(state.local_index) == vcat(collect(1:nnodes(c1)), collect(1:nnodes(c2)))
        @test to_cpu(state.corners) == UInt8[c.corners[i] ? 1 : 0 for c in source for i in 1:nnodes(c)]

        materialized = ContourDynamics.materialize_contours(state)
        @test length(materialized) == length(source)
        @test all(zip(materialized, source)) do (actual, expected)
            actual.nodes == expected.nodes &&
                actual.pv == expected.pv &&
                actual.wrap == expected.wrap &&
                actual.corners == expected.corners
        end
    end

    @testset "GPU contour access requires explicit materialization" begin
        c = circular_patch(0.5, 12, 1.0)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [deepcopy(c)]; dev=GPU())
            @test prob.device_state isa ContourDynamics.DeviceContourState
            @test !(prob.device_state.x isa Vector)
            @test_throws ErrorException contours(prob)

            before = materialize_contours(prob)
            prob.contours[1].nodes[1] = SVector(99.0, 99.0)
            after = materialize_contours(prob)
            @test after[1].nodes[1] == before[1].nodes[1]
            @test after[1].nodes[1] == c.nodes[1]
        else
            @test_throws ErrorException ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
        end
    end

    @testset "DeviceContourState builds velocity segments on backend" begin
        c1 = circular_patch(0.5, 12, 1.0)
        c2 = PVContour([SVector(0.1, 0.0), SVector(0.5, 0.2),
                        SVector(0.4, 0.8), SVector(0.0, 0.7)],
                       -0.25, SVector(1.0, 0.0),
                       Bool[true, false, false, true])
        contours_in = [c1, c2]
        state = DeviceContourState(contours_in, CPU())
        seg = ContourDynamics._state_segment_data(state, CPU())
        # c2 spans a period-1 domain; packing itself does not depend on the domain.
        packed = ContourDynamics.pack_segments(
            ContourProblem(EulerKernel(), PeriodicDomain(0.5, 2.0), contours_in), CPU())

        for name in (:ax, :ay, :bx, :by, :pv, :ka, :kb)
            @test to_cpu(getproperty(seg, name)) ≈ to_cpu(getproperty(packed, name))
        end
    end

    @testset "DeviceContourState velocity ignores stale host contour shadow" begin
        c1 = circular_patch(0.45, 16, 1.0)
        c2 = PVContour([p + SVector(0.9, -0.2) for p in circular_patch(0.2, 10, -0.5).nodes], -0.5)
        original = [c1, c2]
        ref_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(original); dev=CPU())
        ref = zeros(SVector{2,Float64}, total_nodes(ref_prob))
        ContourDynamics._direct_velocity!(ref, ref_prob)

        state_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(original); dev=CPU())
        state = DeviceContourState(deepcopy(original), CPU())
        state_prob.contours[1].nodes[1] = SVector(99.0, 99.0)

        vel = zeros(SVector{2,Float64}, length(ref))
        ContourDynamics._ka_velocity_from_state!(vel, state, state_prob.kernel,
                                                 state_prob.domain, CPU())
        @test all(eachindex(ref)) do i
            isapprox(vel[i][1], ref[i][1]; rtol=1e-10, atol=1e-10) &&
                isapprox(vel[i][2], ref[i][2]; rtol=1e-10, atol=1e-10)
        end
    end

    @testset "DeviceContourState point velocity uses authoritative coordinates" begin
        contours_in = [circular_patch(0.5, 24, 1.0)]
        state = DeviceContourState(deepcopy(contours_in), CPU())
        state.x .+= 0.75
        x = SVector(0.1, 0.2)

        current = ContourProblem(EulerKernel(), UnboundedDomain(),
                                 materialize_contours(state))
        stale = ContourProblem(EulerKernel(), UnboundedDomain(), contours_in)
        expected = velocity(current, x)
        actual = ContourDynamics._ka_velocity_at_state(state, EulerKernel(),
                                                       UnboundedDomain(), x, CPU())

        @test actual ≈ expected rtol=1e-12 atol=1e-12
        @test !isapprox(actual, velocity(stale, x); rtol=1e-8, atol=1e-8)

        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)),
                                    SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = ([circular_patch(0.35, 16, 1.0)],
                  [circular_patch(0.25, 12, -0.5; cx=0.4)])
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        states[1].x .-= 0.6
        current_layers = ntuple(i -> materialize_contours(states[i]), 2)
        current_multi = MultiLayerContourProblem(kernel, UnboundedDomain(), current_layers)
        expected_multi = velocity(current_multi, x)
        actual_multi = ContourDynamics._ka_multilayer_velocity_at_states(
            states, kernel, UnboundedDomain(), x, CPU())

        @test all(isapprox.(actual_multi, expected_multi; rtol=1e-12, atol=1e-12))
    end

    @testset "Mixed-precision point probes retain device-specific dispatch" begin
        contours_in = [circular_patch(0.5, 24, 1.0)]
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), contours_in)
        state_type = typeof(cpu_prob.device_state)
        gpu_prob_type = ContourProblem{
            EulerKernel,UnboundedDomain,Float64,GPU,state_type}
        cpu_method = which(velocity, (typeof(cpu_prob), SVector{2,Float32}))
        gpu_method = which(velocity, (gpu_prob_type, SVector{2,Float32}))

        @test gpu_method !== cpu_method
        @test velocity(cpu_prob, SVector{2,Float32}(0.1, 0.2)) isa
              SVector{2,Float64}

        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)),
                                    SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = ([circular_patch(0.35, 16, 1.0)],
                  [circular_patch(0.25, 12, -0.5; cx=0.4)])
        cpu_multi = MultiLayerContourProblem(kernel, UnboundedDomain(), layers)
        multi_state_type = typeof(cpu_multi.device_state)
        gpu_multi_type = MultiLayerContourProblem{
            2,typeof(kernel),UnboundedDomain,Float64,GPU,multi_state_type}
        cpu_multi_method = which(
            velocity, (typeof(cpu_multi), SVector{2,Float32}))
        gpu_multi_method = which(
            velocity, (gpu_multi_type, SVector{2,Float32}))

        @test gpu_multi_method !== cpu_multi_method
        @test all(v -> v isa SVector{2,Float64},
                  velocity(cpu_multi, SVector{2,Float32}(0.1, 0.2)))
    end

    @testset "DeviceContourState RK4 step matches CPU contour RK4" begin
        contours_in = [
            circular_patch(0.35, 18, 1.0),
            PVContour([p + SVector(0.7, -0.25) for p in circular_patch(0.18, 12, -0.4).nodes], -0.4),
        ]
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours_in); dev=CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())
        dt = 0.002
        cpu_stepper = RK4Stepper(dt, total_nodes(cpu_prob); dev=CPU())
        state_stepper = RK4Stepper(dt, total_nodes(cpu_prob); dev=CPU())

        timestep!(cpu_prob, cpu_stepper)
        ContourDynamics._rk4_state_step!(state, EulerKernel(), UnboundedDomain(),
                                         state_stepper, CPU())
        state_contours = materialize_contours(state)

        @test all(zip(state_contours, cpu_prob.contours)) do (actual, expected)
            all(isapprox.(actual.nodes, expected.nodes; rtol=1e-10, atol=1e-10))
        end
    end

    @testset "DeviceContourState periodic wrapping matches CPU contour wrapping" begin
        domain = PeriodicDomain(1.0, 1.0)
        shifted = PVContour([p + SVector(2.2, -2.1) for p in circular_patch(0.2, 12, 1.0).nodes], 1.0)
        spanning = PVContour([SVector(-1.0, -0.5), SVector(1.0, -0.5),
                              SVector(1.0, 0.5), SVector(-1.0, 0.5)],
                             -0.25, SVector(2.0, 0.0))
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy([shifted, spanning]); dev=CPU())
        state = DeviceContourState(deepcopy([shifted, spanning]), CPU())

        wrap_nodes!(cpu_prob)
        ContourDynamics._wrap_state_nodes!(state, domain, CPU())
        state_contours = materialize_contours(state)

        @test all(zip(state_contours, cpu_prob.contours)) do (actual, expected)
            all(isapprox.(actual.nodes, expected.nodes; rtol=1e-12, atol=1e-12))
        end
    end

    @testset "Stepper buffers honor selected device" begin
        rk_cpu = RK4Stepper(0.01, 8; dev=CPU())
        @test rk_cpu.k1 isa Vector{SVector{2,Float64}}

        if _TEST_CUDA_LOADED[]
            rk_gpu = RK4Stepper(0.01, 8; dev=GPU())
            @test !(rk_gpu.k1 isa Vector)
            @test !(rk_gpu.nodes_buf isa Vector)
        else
            @test_throws ErrorException RK4Stepper(0.01, 8; dev=GPU())
        end
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

    @testset "CPU velocity! agrees with the KA evaluator" begin
        c = circular_patch(0.5, 32, 1.0)
        prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=CPU())
        N = total_nodes(prob)

        # CPU velocity! uses the allocation-free direct evaluator; it must still
        # match the KA workspace path (reserved for GPU) to machine precision.
        vel_ka = zeros(SVector{2,Float64}, N)
        vel = similar(vel_ka)
        ContourDynamics._ka_velocity!(vel_ka, prob, prob.dev)
        velocity!(vel, prob)

        for i in 1:N
            @test isapprox(vel[i][1], vel_ka[i][1]; atol=1e-12)
            @test isapprox(vel[i][2], vel_ka[i][2]; atol=1e-12)
        end
    end

    @testset "KA single-layer energy matches scalar diagnostics on CPU backend" begin
        c = circular_patch(0.25, 8, 1.0)
        probs = (
            ContourProblem(EulerKernel(), UnboundedDomain(), [c]),
            ContourProblem(QGKernel(1.0), UnboundedDomain(), [c]),
            ContourProblem(SQGKernel(0.02), UnboundedDomain(), [c]),
            ContourProblem(EulerKernel(), PeriodicDomain(2.0, 2.0), [c]),
            ContourProblem(QGKernel(1.0), PeriodicDomain(2.0, 2.0), [c]),
            ContourProblem(SQGKernel(0.02), PeriodicDomain(2.0, 2.0), [c]),
        )

        for prob in probs
            E_scalar = energy(prob)
            E_ka = ContourDynamics._ka_energy(prob, CPU())
            @test isfinite(E_ka)
            @test E_ka ≈ E_scalar rtol=1e-7 atol=1e-10
        end
    end

    @testset "DeviceContourState geometry diagnostics match CPU diagnostics" begin
        closed1 = circular_patch(0.35, 24, 1.0)
        closed2 = PVContour([p + SVector(0.8, -0.25) for p in circular_patch(0.18, 16, -0.4).nodes], -0.4)
        spanning = PVContour([SVector(-1.0, -0.2), SVector(1.0, -0.2),
                              SVector(1.0, 0.2), SVector(-1.0, 0.2)],
                             0.25, SVector(2.0, 0.0))
        contours_in = [closed1, closed2, spanning]
        prob = ContourProblem(EulerKernel(), PeriodicDomain(1.0, 2.0), deepcopy(contours_in); dev=CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        @test to_cpu(ContourDynamics._state_vortex_area(state, CPU())) ≈ vortex_area(prob)
        @test ContourDynamics._state_circulation(state, CPU()) ≈ circulation(prob)
        @test ContourDynamics._state_enstrophy(state, CPU()) ≈ enstrophy(prob)
        @test ContourDynamics._state_angular_momentum(state, CPU()) ≈ angular_momentum(prob)
    end

    @testset "DeviceContourState energy diagnostics match KA CPU diagnostics" begin
        contours_in = [
            circular_patch(0.35, 20, 1.0),
            PVContour([p + SVector(0.8, -0.25) for p in circular_patch(0.18, 14, -0.4).nodes], -0.4),
        ]
        domains = (UnboundedDomain(), PeriodicDomain(2.0, 2.0))
        kernels = (EulerKernel(), QGKernel(1.25), SQGKernel(0.02))
        for domain in domains, kernel in kernels
            clear_ewald_cache!()
            prob = ContourProblem(kernel, domain, deepcopy(contours_in); dev=CPU())
            state = DeviceContourState(deepcopy(contours_in), CPU())
            @test ContourDynamics._ka_energy_from_state(state, kernel, domain, CPU()) ≈
                  ContourDynamics._ka_energy(prob, CPU()) rtol=1e-7 atol=1e-10
        end
    end

    @testset "Energy packing excludes invalid contours in both precisions" begin
        for T in (Float32, Float64)
            closed = circular_patch(0.25, 6, one(T); T=T)
            short = PVContour(closed.nodes[1:2], T(3))
            spanning = PVContour(copy(closed.nodes), T(2), SVector{2,T}(4, 0))
            invalid = [short, spanning]
            mixed = [short, closed, spanning]
            tolerance = T === Float32 ? T(2e-5) : T(1e-7)
            for domain in (UnboundedDomain(), PeriodicDomain(T(2), T(2))),
                kernel in (EulerKernel(), QGKernel(T(1.25)), SQGKernel(T(0.02)))
                expected = energy(ContourProblem(kernel, domain, [closed]))
                workspace = ExecutionWorkspace(T)
                for (contours_in, reference) in ((mixed, expected), (invalid, zero(T)),
                                                  (PVContour{T}[], zero(T)))
                    state = DeviceContourState(contours_in, CPU())
                    @test ContourDynamics._ka_energy_from_state(
                        contours_in, kernel, domain, CPU(); workspace) ≈
                          reference rtol=tolerance atol=eps(T)
                    @test ContourDynamics._ka_energy_from_state(
                        state, kernel, domain, CPU(); workspace) ≈
                          reference rtol=tolerance atol=eps(T)
                end
            end
        end
    end

    @testset "State-based point velocity stays on the KA backend" begin
        point = SVector(0.17, -0.11)
        contours_in = [
            circular_patch(0.35, 20, 1.0),
            PVContour([p + SVector(0.7, -0.25) for p in circular_patch(0.16, 12, -0.4).nodes], -0.4),
        ]
        for domain in (UnboundedDomain(), PeriodicDomain(2.0, 2.0)),
            kernel in (EulerKernel(), QGKernel(1.25), SQGKernel(0.02))
            clear_ewald_cache!()
            prob = ContourProblem(kernel, domain, deepcopy(contours_in); dev=CPU())
            state = DeviceContourState(deepcopy(contours_in), CPU())
            @test ContourDynamics._ka_velocity_at_state(
                state, kernel, domain, point, CPU()) ≈ velocity(prob, point) rtol=1e-8 atol=1e-10
        end

        domain = PeriodicDomain(2.0, 2.0)
        reference = beta_staircase(0.4, domain, 4; nodes_per_contour=8)
        kernel = BetaPlaneQGKernel(0.4, 1.0, reference)
        live = vcat(deepcopy(reference), [circular_patch(0.25, 16, 2π; cy=0.5)])
        prob = ContourProblem(kernel, domain, deepcopy(live); dev=CPU())
        state = DeviceContourState(deepcopy(live), CPU())
        @test ContourDynamics._ka_velocity_at_state(
            state, kernel, domain, point, CPU()) ≈ velocity(prob, point) rtol=1e-8 atol=1e-10
    end

    @testset "Batched point velocity matches per-point probes" begin
        points = [SVector(0.17, -0.11), SVector(-0.42, 0.33), SVector(0.61, 0.05),
                  SVector(-0.05, -0.58), SVector(0.9, 0.9)]
        contours_in = [
            circular_patch(0.35, 20, 1.0),
            PVContour([p + SVector(0.7, -0.25) for p in circular_patch(0.16, 12, -0.4).nodes], -0.4),
        ]
        cases = (
            (EulerKernel(), UnboundedDomain()),
            (QGKernel(1.25), PeriodicDomain(2.0, 2.0)),
            (SQGKernel(0.02), UnboundedDomain()),
        )
        for (kernel, domain) in cases
            clear_ewald_cache!()
            domain isa PeriodicDomain && setup_ewald_cache!(domain, kernel)
            prob = ContourProblem(kernel, domain, deepcopy(contours_in); dev=CPU())
            state = DeviceContourState(deepcopy(contours_in), CPU())
            expected = [velocity(prob, x) for x in points]

            # CPU problem: the batched method maps the single-point probe.
            batched = velocity(prob, points)
            @test batched isa Vector{SVector{2,Float64}}
            @test length(batched) == length(points)
            @test all(isapprox.(batched, expected; rtol=1e-12, atol=1e-12))

            # KA backend: one pack and one launch over all targets must agree
            # with the per-point launches.
            ka_batched = ContourDynamics._ka_velocity_at_state(
                state, kernel, domain, points, CPU())
            ka_single = [ContourDynamics._ka_velocity_at_state(
                             state, kernel, domain, x, CPU()) for x in points]
            @test ka_batched isa Vector{SVector{2,Float64}}
            @test all(isapprox.(ka_batched, ka_single; rtol=1e-12, atol=1e-12))
            @test all(isapprox.(ka_batched, expected; rtol=1e-8, atol=1e-10))

            # Mixed precision targets are promoted to the problem precision.
            @test velocity(prob, SVector{2,Float32}.(points)) isa Vector{SVector{2,Float64}}
            @test isempty(velocity(prob, SVector{2,Float64}[]))
        end

        # Multi-layer: batched modal projection equals the per-point one.
        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)),
                                    SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = ([circular_patch(0.35, 16, 1.0)],
                  [circular_patch(0.25, 12, -0.5; cx=0.4)])
        multi = MultiLayerContourProblem(kernel, UnboundedDomain(), deepcopy(layers))
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        expected_multi = [velocity(multi, x) for x in points]
        batched_multi = velocity(multi, points)
        ka_multi = ContourDynamics._ka_multilayer_velocity_at_states(
            states, kernel, UnboundedDomain(), points, CPU())
        @test batched_multi isa Vector{NTuple{2,SVector{2,Float64}}}
        @test all(zip(batched_multi, expected_multi)) do (a, b)
            all(isapprox.(a, b; rtol=1e-12, atol=1e-12))
        end
        @test all(zip(ka_multi, expected_multi)) do (a, b)
            all(isapprox.(a, b; rtol=1e-12, atol=1e-12))
        end
    end

    # (name, kernel, domain, patch, isapprox keywords). `atol` alone implies rtol=0 in
    # isapprox, so the keyword sets are kept verbatim rather than normalised.
    ka_velocity_cases = [
        ("KA SQG velocity matches direct CPU",
         SQGKernel(0.02), UnboundedDomain(), circular_patch(0.5, 32, 1.0), (atol=1e-12,)),
        ("KA QG velocity matches direct CPU",
         QGKernel(1.25), UnboundedDomain(), circular_patch(0.5, 32, 1.0), (atol=1e-8, rtol=1e-8)),
        ("KA periodic Euler velocity matches direct CPU",
         EulerKernel(), PeriodicDomain(2.0, 2.0), circular_patch(0.35, 24, 1.0), (atol=1e-12, rtol=1e-12)),
        ("KA periodic QG velocity matches direct CPU",
         QGKernel(1.1), PeriodicDomain(2.0, 2.0), circular_patch(0.35, 24, 1.0), (atol=1e-12, rtol=1e-12)),
        ("KA periodic SQG velocity matches direct CPU",
         SQGKernel(0.02), PeriodicDomain(2.0, 2.0), circular_patch(0.35, 24, 1.0), (atol=1e-12, rtol=1e-12)),
    ]

    for (name, kernel, domain, c, tol) in ka_velocity_cases
        @testset "$name" begin
            within_tol(a, b) = isapprox(a, b; tol...)
            domain isa PeriodicDomain && clear_ewald_cache!()
            prob = ContourProblem(kernel, domain, [c])
            N = total_nodes(prob)

            vel_ref = zeros(SVector{2,Float64}, N)
            vel_ka = similar(vel_ref)
            ContourDynamics._direct_velocity!(vel_ref, prob)
            ContourDynamics._ka_velocity!(vel_ka, prob, CPU())

            for i in 1:N
                @test within_tol(vel_ka[i][1], vel_ref[i][1])
                @test within_tol(vel_ka[i][2], vel_ref[i][2])
            end
        end
    end

    @testset "State-based beta-plane velocity matches CPU reference" begin
        domain = PeriodicDomain(2.0, 2.0)
        reference = beta_staircase(0.4, domain, 4; nodes_per_contour=8)
        kernel = BetaPlaneQGKernel(0.4, 1.0, reference)
        live = vcat(deepcopy(reference), [circular_patch(0.25, 16, 2π; cy=0.5)])
        prob = ContourProblem(kernel, domain, deepcopy(live))
        state = DeviceContourState(deepcopy(live), CPU())
        expected = zeros(SVector{2,Float64}, total_nodes(prob))
        actual = similar(expected)

        velocity!(expected, prob)
        ContourDynamics._ka_velocity_from_state!(
            actual, state, kernel, domain, CPU())
        @test all(isapprox.(actual, expected; rtol=1e-8, atol=1e-10))
    end

    @testset "KA multi-layer velocity matches direct CPU" begin
        Ld = SVector(1.0)
        F = 1.0 / (2 * Ld[1]^2)
        coupling = SMatrix{2,2}(-F, F, F, -F)
        c1 = circular_patch(0.35, 24, 1.0)
        c2 = circular_patch(0.2, 16, -0.5)
        prob = MultiLayerContourProblem(MultiLayerQGKernel(Ld, coupling), UnboundedDomain(),
                                        ([c1], [c2]))

        vel_ref = (zeros(SVector{2,Float64}, nnodes(c1)),
                   zeros(SVector{2,Float64}, nnodes(c2)))
        vel_ka = (similar(vel_ref[1]), similar(vel_ref[2]))

        ContourDynamics._direct_velocity!(vel_ref, prob)
        let _states = (ContourDynamics.DeviceContourState(prob.layers[1], CPU()),
                       ContourDynamics.DeviceContourState(prob.layers[2], CPU()))
            _flat = zeros(SVector{2,Float64}, nnodes(c1) + nnodes(c2))
            ContourDynamics._ka_multilayer_velocity_from_states!(_flat, _states, prob.kernel, prob.domain, CPU())
            vel_ka[1] .= _flat[1:nnodes(c1)]
            vel_ka[2] .= _flat[nnodes(c1)+1:end]
        end

        for i in eachindex(vel_ref[1])
            @test isapprox(vel_ka[1][i][1], vel_ref[1][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[1][i][2], vel_ref[1][i][2]; atol=1e-8, rtol=1e-8)
        end
        for i in eachindex(vel_ref[2])
            @test isapprox(vel_ka[2][i][1], vel_ref[2][i][1]; atol=1e-8, rtol=1e-8)
            @test isapprox(vel_ka[2][i][2], vel_ref[2][i][2]; atol=1e-8, rtol=1e-8)
        end
    end


    @testset "Advertised GPU dispatch contains no CPU fallback" begin
        root = normpath(joinpath(@__DIR__, ".."))
        velocity_source = read(joinpath(root, "src", "velocity", "common.jl"), String)
        surgery_source = read(
            joinpath(root, "src", "accel", "ka", "surgery", "driver.jl"), String)
        device_docs = read(joinpath(root, "docs", "src", "api", "devices.md"), String)
        velocity_docs = read(joinpath(root, "docs", "src", "api", "velocity.md"), String)
        architecture_docs = read(joinpath(root, "docs", "src", "architecture.md"), String)

        @test !occursin("ContourProblem(kernel, domain, materialize_contours", velocity_source)
        @test !occursin("MultiLayerContourProblem(kernel, domain, host_layers", velocity_source)
        @test !occursin("# GPU fallback", velocity_source)
        @test !occursin("_host_boundary_surgery!", surgery_source)
        @test !occursin("runs the CPU surgery pass", device_docs)
        @test !occursin("periodic surgery intentionally materializes", velocity_docs)
        @test !occursin("runs the scalar direct evaluator", velocity_docs)
        @test !occursin("periodic surgery materializes", architecture_docs)
    end
end

@testset "Multi-layer device state" begin
    @testset "Scaled segment pack multiplies PV only" begin
        c = circular_patch(0.4, 12, 2.0)
        state = ContourDynamics.DeviceContourState([c], CPU())
        seg1 = ContourDynamics._state_segment_data(state, CPU())
        n = ContourDynamics._device_state_nnodes(state)

        ax = zeros(n); ay = zeros(n); bx = zeros(n); by = zeros(n)
        pv = zeros(n); ka = zeros(n); kb = zeros(n)
        ContourDynamics.@_ka_launch CPU() n ContourDynamics._state_segment_data_kernel!(
            ax, ay, bx, by, pv, ka, kb,
            state.x, state.y, state.pv, state.wrapx, state.wrapy,
            state.offsets, state.lengths, state.corners,
            state.contour_of_node, state.local_index, 0.5, n)

        @test pv ≈ 0.5 .* seg1.pv
        @test ax == seg1.ax && ay == seg1.ay && bx == seg1.bx && by == seg1.by
        @test ka == seg1.ka && kb == seg1.kb
    end

    @testset "Layer state ranges partition flat index space" begin
        c1 = circular_patch(0.5, 24, 1.0)
        c2 = circular_patch(0.3, 16, -0.7)
        states = (ContourDynamics.DeviceContourState([c1], CPU()),
                  ContourDynamics.DeviceContourState([c2], CPU()))
        ranges = ContourDynamics._layer_state_ranges(states)
        @test ranges isa NTuple{2, UnitRange{Int}}  # tuple, not Vector: allocation-free per-step path
        @test ranges == (1:24, 25:40)

        # A layer with 0 nodes produces an empty range and the next layer starts at the same cursor.
        states3 = (ContourDynamics.DeviceContourState(PVContour{Float64}[], CPU()),
                   ContourDynamics.DeviceContourState([circular_patch(0.2, 8, 1.0)], CPU()))
        ranges3 = ContourDynamics._layer_state_ranges(states3)
        @test ranges3[1] == 1:0
        @test ranges3[2] == 1:8

        F = 0.5
        kernel = MultiLayerQGKernel(
            SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        layers3 = (PVContour{Float64}[], [circular_patch(0.2, 8, 1.0)])
        prob3 = MultiLayerContourProblem(kernel, UnboundedDomain(), layers3)
        point = SVector(0.17, -0.11)
        @test all(isapprox.(
            ContourDynamics._ka_multilayer_velocity_at_states(
                states3, kernel, UnboundedDomain(), point, CPU()),
            velocity(prob3, point); rtol=1e-8, atol=1e-10))
    end

    @testset "Multi-layer output validation uses authoritative state sizes" begin
        layers = (
            [circular_patch(0.3, 8, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.4)],
        )
        states = ntuple(i -> DeviceContourState(layers[i], CPU()), 2)
        vel = (zeros(SVector{2,Float64}, 8), zeros(SVector{2,Float64}, 12))

        @test ContourDynamics._validate_multilayer_state_velocity_buffer!(vel, states) === vel
        undersized = (zeros(SVector{2,Float64}, 7), vel[2])
        @test_throws DimensionMismatch ContourDynamics._validate_multilayer_state_velocity_buffer!(
            undersized, states)
    end

    @testset "State-based multi-layer velocity matches CPU modal velocity (unbounded)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16))
        velocity!(vel_t, cpu_prob)
        expected = vcat(vel_t[1], vel_t[2])

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        flat = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                             UnboundedDomain(), CPU())
        @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
    end

    @testset "State-based multi-layer velocity matches CPU modal velocity (periodic)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )
        domain = PeriodicDomain(4.0, 4.0)
        setup_ewald_cache!(domain, EulerKernel())

        cpu_prob = MultiLayerContourProblem(ml_kernel, domain, deepcopy(ml_layers))
        vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16))
        velocity!(vel_t, cpu_prob)
        expected = vcat(vel_t[1], vel_t[2])

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        flat = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                             domain, CPU())
        @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
        point = SVector(0.17, -0.11)
        @test all(isapprox.(
            ContourDynamics._ka_multilayer_velocity_at_states(
                states, ml_kernel, domain, point, CPU()),
            velocity(cpu_prob, point); rtol=1e-8, atol=1e-10))
    end

    @testset "State-based 3-layer velocity matches CPU modal velocity (asymmetric P)" begin
        # Distinct tridiagonal couplings give an orthogonal but NON-symmetric
        # eigenvector matrix, so P != P_inv and a swapped/transposed projection
        # would fail this test (the 2-layer equal-F fixture cannot catch that).
        #
        # Coupling matrix is the tridiagonal nearest-neighbor form
        #   C = [-F1  F1   0 ]
        #       [ F1 -(F1+F2)  F2]
        #       [  0   F2  -F2]
        # with F1 != F2.  F1, F2 are chosen so that the two non-zero
        # eigenvalues are exactly -1 and -4, consistent with Ld = [1.0, 0.5].
        # (F1+F2 = 2.5, F1*F2 = 4/3 → F = (2.5 ± √(11/12)) / 2)
        let F1 = (2.5 + sqrt(11.0/12.0)) / 2,
            F2 = (2.5 - sqrt(11.0/12.0)) / 2
            ml_Ld      = SVector(1.0, 0.5)
            ml_coupling = SMatrix{3,3,Float64}(
                -F1,  F1,       0,
                 F1, -(F1+F2), F2,
                  0,  F2,      -F2)
            ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
            P = ml_kernel.eigenvectors
            @test !(P ≈ ml_kernel.eigenvectors_inv)  # fixture sanity: orientation distinguishable

            ml_layers = (
                [circular_patch(0.5, 24, 1.0)],
                [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
                [PVContour([p + SVector(-0.4, 0.5) for p in circular_patch(0.25, 12, 0.9).nodes], 0.9)],
            )

            cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
            vel_t = (zeros(SVector{2,Float64}, 24), zeros(SVector{2,Float64}, 16), zeros(SVector{2,Float64}, 12))
            velocity!(vel_t, cpu_prob)
            expected = vcat(vel_t[1], vel_t[2], vel_t[3])

            states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                      ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()),
                      ContourDynamics.DeviceContourState(deepcopy(ml_layers[3]), CPU()))
            flat = zeros(SVector{2,Float64}, 52)
            ContourDynamics._ka_multilayer_velocity_from_states!(flat, states, ml_kernel,
                                                                 UnboundedDomain(), CPU())
            @test all(isapprox.(flat, expected; rtol=1e-8, atol=1e-10))
            point = SVector(0.17, -0.11)
            @test all(isapprox.(
                ContourDynamics._ka_multilayer_velocity_at_states(
                    states, ml_kernel, UnboundedDomain(), point, CPU()),
                velocity(cpu_prob, point); rtol=1e-8, atol=1e-10))
        end
    end

    @testset "State-based multi-layer energy matches CPU modal energy" begin
        F = 0.5
        kernel = MultiLayerQGKernel(
            SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        layers = (
            [circular_patch(0.3, 16, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.4)],
        )
        states = ntuple(i -> DeviceContourState(layers[i], CPU()), 2)

        for domain in (UnboundedDomain(), PeriodicDomain(2.0, 2.0))
            clear_ewald_cache!()
            prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
            actual = ContourDynamics._ka_multilayer_energy_from_states(
                states, kernel, domain, CPU())
            @test actual ≈ energy(prob) rtol=1e-7 atol=1e-10
        end

        F1 = (2.5 + sqrt(11.0 / 12.0)) / 2
        F2 = (2.5 - sqrt(11.0 / 12.0)) / 2
        kernel3 = MultiLayerQGKernel(
            SVector(1.0, 0.5),
            SMatrix{3,3,Float64}(
                -F1, F1, 0,
                F1, -(F1 + F2), F2,
                0, F2, -F2))
        layers3 = (
            [circular_patch(0.25, 10, 1.0)],
            [circular_patch(0.2, 8, -0.6; cx=0.4)],
            [circular_patch(0.15, 8, 0.8; cx=-0.3, cy=0.35)],
        )
        states3 = ntuple(i -> DeviceContourState(layers3[i], CPU()), 3)
        prob3 = MultiLayerContourProblem(kernel3, UnboundedDomain(), deepcopy(layers3))
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states3, kernel3, UnboundedDomain(), CPU()) ≈ energy(prob3) rtol=1e-7 atol=1e-10
    end

    @testset "Multi-layer state RK4 matches CPU multi-layer RK4" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        ml_layers = (
            [circular_patch(0.5, 24, 1.0)],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)],
        )
        dt = 0.002

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        cpu_stepper = RK4Stepper(dt, total_nodes(cpu_prob))
        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        state_stepper = RK4Stepper(dt, 40; dev=CPU())

        for _ in 1:3
            timestep!(cpu_prob, cpu_stepper)
            ContourDynamics._rk4_multilayer_state_step!(states, ml_kernel,
                                                        UnboundedDomain(), state_stepper, CPU())
        end

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test all(zip(actual, expected)) do (a, e)
                all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end
    end

    @testset "State-based multi-layer velocity ignores stale host contours" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        layer1 = [circular_patch(0.5, 24, 1.0)]
        layer2 = [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 16, -0.7).nodes], -0.7)]

        states = (ContourDynamics.DeviceContourState(layer1, CPU()),
                  ContourDynamics.DeviceContourState(layer2, CPU()))
        flat_before = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat_before, states, ml_kernel,
                                                             UnboundedDomain(), CPU())

        # Mutate the host contours the states were built from.
        # DeviceContourState copies node data into its own arrays, so the state
        # is immune to host-side mutations — flat_after must equal flat_before.
        for i in eachindex(layer1[1].nodes)
            layer1[1].nodes[i] += SVector(10.0, 10.0)
        end

        flat_after = zeros(SVector{2,Float64}, 40)
        ContourDynamics._ka_multilayer_velocity_from_states!(flat_after, states, ml_kernel,
                                                             UnboundedDomain(), CPU())
        @test flat_after == flat_before
    end

    @testset "Multi-layer state periodic wrap matches CPU wrap" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        domain = PeriodicDomain(2.0, 2.0)
        # Patches whose centroids sit outside the box, so wrapping moves them.
        ml_layers = (
            [PVContour([p + SVector(2.5, 0.0) for p in circular_patch(0.3, 16, 1.0).nodes], 1.0)],
            [PVContour([p + SVector(0.0, -2.5) for p in circular_patch(0.2, 12, -0.5).nodes], -0.5)],
        )

        cpu_prob = MultiLayerContourProblem(ml_kernel, domain, deepcopy(ml_layers))
        wrap_nodes!(cpu_prob)
        # Guard against double-no-op: wrapping must actually displace both layers.
        @test abs(cpu_prob.layers[1][1].nodes[1][1] - ml_layers[1][1].nodes[1][1]) > 0.5
        @test abs(cpu_prob.layers[2][1].nodes[1][2] - ml_layers[2][1].nodes[1][2]) > 0.5

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        for s in states
            ContourDynamics._wrap_state_nodes!(s, domain, CPU())
        end

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test all(zip(actual, expected)) do (a, e)
                all(isapprox.(a.nodes, e.nodes; rtol=1e-12, atol=1e-12))
            end
        end
    end
end
