using ContourDynamics
using StaticArrays
using Test
using Logging

# Guard against double-include when run from runtests.jl
@isdefined(circular_patch) || include("test_utils.jl")
@isdefined(_full_rewrite_output_layout) || include("device_rewrite_layout_oracle.jl")

function _assert_device_layout_matches_host(contours, plan)
    host = _full_rewrite_output_layout(contours, plan, CPU())
    dev = ContourDynamics._device_full_rewrite_output_layout(contours, plan, CPU())
    @test dev.total_nodes == host.total_nodes
    for name in (:offsets, :lengths, :op_index, :source_contour, :part,
                 :pv, :wrapx, :wrapy, :out_node_contour)
        @test to_cpu(getproperty(dev, name)) == to_cpu(getproperty(host, name))
    end
end

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

    @testset "GPU surgery! without CUDA gives helpful error" begin
        c = circular_patch(0.5, 32, 1.0)
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        if _TEST_CUDA_LOADED[]
            prob = ContourProblem(EulerKernel(), UnboundedDomain(), [c]; dev=GPU())
            surgery!(prob, params)
            @test materialize_contours(prob) isa Vector{PVContour{Float64}}
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

    @testset "Device surgery filament flags match Dritschel cleanup predicates" begin
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.0005, 0.001)
        ], 1.0)
        normal = circular_patch(1.0, 64, 1.0)
        corner_nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(1.0, 1.0),
            SVector(0.0, 1.0),
        ]
        corner_flags = falses(length(corner_nodes))
        corner_flags[1] = true
        too_few_corner = PVContour(corner_nodes, 1.0, zero(SVector{2,Float64}), corner_flags)
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)

        flags = ContourDynamics._device_filament_flags([normal, tiny, too_few_corner], params, CPU())

        @test flags == [false, true, true]

        contours = [normal, tiny, too_few_corner]
        ContourDynamics._device_remove_filaments!(contours, params, CPU())
        @test length(contours) == 1
        @test contours[1].nodes == normal.nodes
    end

    @testset "DeviceContourState filament removal rewrites authoritative state" begin
        tiny = PVContour([
            SVector(0.0, 0.0), SVector(0.001, 0.0), SVector(0.0005, 0.001)
        ], 1.0)
        normal = circular_patch(1.0, 64, 1.0)
        params = SurgeryParams(0.001, 0.005, 0.1, 1e-4, 10)
        state = DeviceContourState([normal, tiny], CPU())

        ContourDynamics._device_remove_filaments!(state, params, CPU())
        actual = materialize_contours(state)

        @test length(actual) == 1
        @test actual[1].nodes == normal.nodes
        @test actual[1].pv == normal.pv
    end

    @testset "DeviceContourState corner promotion matches CPU corner rules" begin
        nodes = SVector{2,Float64}[
            SVector(0.0, 0.0),
            SVector(1.0, 0.0),
            SVector(2.0, 0.2),
            SVector(0.3, 0.8),
            SVector(1.1, -0.15),
        ]
        corners = falses(length(nodes))
        corners[2] = true
        contours_in = [PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)]
        expected = deepcopy(contours_in)
        ContourDynamics._demote_obtuse_corners!(expected)
        ContourDynamics._promote_high_curvature_corners!(expected, 1.0)
        state = DeviceContourState(deepcopy(contours_in), CPU())

        ContourDynamics._demote_obtuse_corners!(state, CPU())
        ContourDynamics._promote_high_curvature_corners!(state, 1.0, CPU())
        actual = materialize_contours(state)

        @test actual[1].corners == expected[1].corners
    end

    @testset "Device remeshing is independent of coordinate units" begin
        for T in (Float32, Float64)
            c = elliptical_patch(2, 1, 96, 1; T=T)
            params = SurgeryParams(T(0.01), T(0.05), T(0.5), T(1e-8), 5)
            baseline = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
            tolerance = T === Float32 ? T(2e-4) : T(1e-10)
            for scale in (T(0.01), T(100))
                scaled = PVContour([scale * x for x in c.nodes], c.pv)
                scaled_params = SurgeryParams(params.δ * scale, params.μ * scale,
                    params.Δ_max * scale, params.area_min * scale^2, params.n_surgery)
                actual = only(ContourDynamics._device_remesh_contours([scaled], scaled_params, CPU()))
                host = remesh(scaled, scaled_params)

                @test nnodes(actual) == nnodes(baseline) == nnodes(host)
                @test actual.nodes ./ scale ≈ baseline.nodes rtol=tolerance
                @test actual.nodes ≈ host.nodes rtol=tolerance
            end
        end
    end

    @testset "Device closed remesh matches CPU weighted remesh" begin
        contours = [
            elliptical_patch(1.0, 0.6, 40, 1.0),
            circular_patch(0.35, 24, 0.5),
        ]
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)

        density_sources = copy(contours)
        cpu_remeshed = [
            remesh(c, params; _density_sources=density_sources)
            for c in contours
        ]
        device_remeshed = ContourDynamics._device_remesh_contours(contours, params, CPU())

        @test device_remeshed !== nothing
        @test nnodes.(device_remeshed) == nnodes.(cpu_remeshed)
        @test getproperty.(device_remeshed, :pv) == getproperty.(cpu_remeshed, :pv)
        @test getproperty.(device_remeshed, :wrap) == getproperty.(cpu_remeshed, :wrap)
        @test all(zip(device_remeshed, cpu_remeshed)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
        end
        @test all(c -> isempty(corner_indices(c)), device_remeshed)

        cornered = deepcopy(contours[1])
        cornered.corners[1] = true
        cornered.corners[21] = true
        cpu_cornered = remesh(cornered, params; _density_sources=[cornered])
        device_cornered = only(ContourDynamics._device_remesh_contours([cornered], params, CPU()))
        @test nnodes(device_cornered) == nnodes(cpu_cornered)
        @test corner_indices(device_cornered) == corner_indices(cpu_cornered)
        @test all(isapprox.(device_cornered.nodes, cpu_cornered.nodes; atol=1e-10, rtol=1e-10))
    end

    @testset "DeviceContourState remesh rewrites authoritative state" begin
        contours_in = [
            elliptical_patch(0.9, 0.45, 30, 1.0),
            PVContour([p + SVector(1.2, -0.3) for p in circular_patch(0.25, 18, -0.5).nodes], -0.5),
        ]
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)
        expected = ContourDynamics._device_remesh_contours(deepcopy(contours_in), params, CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        @test expected !== nothing
        ContourDynamics._device_remesh_state!(state, params, CPU())
        actual = materialize_contours(state)

        @test nnodes.(actual) == nnodes.(expected)
        @test getproperty.(actual, :pv) == getproperty.(expected, :pv)
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "DeviceContourState rewrite preserves unchanged contours" begin
        δ = 0.02
        contours_in = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            PVContour([p + SVector(4.0, 0.0) for p in circular_patch(0.2, 16, 0.5).nodes], 0.5),
        ]
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours_in, δ, UnboundedDomain(), CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            contours_in, candidates, CPU())
        expected = ContourDynamics._device_rewrite_contours(contours_in, selected, CPU())
        state = DeviceContourState(deepcopy(contours_in), CPU())

        ContourDynamics._device_rewrite_state!(state, selected, CPU())
        actual = materialize_contours(state)

        @test nnodes.(actual) == nnodes.(expected)
        @test getproperty.(actual, :pv) == getproperty.(expected, :pv)
        @test actual[end].nodes == contours_in[end].nodes
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "DeviceContourState reconnect mutates state not host contours" begin
        δ = 0.02
        contours_in = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
        ]
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            contours_in, δ, UnboundedDomain(), CPU())
        expected = deepcopy(contours_in)
        @test ContourDynamics._device_reconnect!(expected, candidates, CPU())
        state_shadow = deepcopy(contours_in)
        state = DeviceContourState(state_shadow, CPU())

        @test ContourDynamics._device_reconnect!(state, candidates, CPU())
        actual = materialize_contours(state)

        @test all(zip(state_shadow, contours_in)) do (shadow_c, source_c)
            shadow_c.nodes == source_c.nodes &&
                shadow_c.pv == source_c.pv &&
                shadow_c.wrap == source_c.wrap &&
                shadow_c.corners == source_c.corners
        end
        @test nnodes.(actual) == nnodes.(expected)
        @test all(zip(actual, expected)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12)) &&
                dev_c.corners == cpu_c.corners
        end
    end

    @testset "Device close-pair candidates match CPU surgery on simple merge" begin
        δ = 0.02
        contours_same = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
        ]
        idx = ContourDynamics.build_spatial_index(contours_same, δ)
        cpu_pairs = Set(ContourDynamics.find_close_segments(contours_same, idx, δ))
        buffer = ContourDynamics._device_close_pair_candidate_buffer(contours_same, δ, CPU())
        buffered_pairs = Set(ContourDynamics._unpack_close_pair_candidates(buffer))
        dev_pairs = Set(ContourDynamics._device_close_pair_candidates(contours_same, δ, CPU()))
        admissible_buffer = ContourDynamics._device_admissible_close_segment_buffer(
            contours_same, δ, UnboundedDomain(), CPU())
        admissible_pairs = Set(ContourDynamics._unpack_close_pair_candidates(admissible_buffer))

        # Raw candidates include right-angle contacts at the squares' corners,
        # where the two parts' far sides lie in different fluid; admissibility
        # drops those on both backends and keeps every facing-side contact.
        facing = Set(p for p in dev_pairs if 7 <= p[2] <= 12 && 19 <= p[4] <= 24)
        @test !isempty(cpu_pairs)
        @test cpu_pairs == admissible_pairs
        @test issubset(cpu_pairs, dev_pairs)
        @test issubset(facing, cpu_pairs)
        @test buffered_pairs == dev_pairs
        @test length(to_cpu(buffer.ci)) == length(dev_pairs)

        contours_different_pv = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 2.0),
        ]
        @test isempty(ContourDynamics._device_close_pair_candidates(contours_different_pv, δ, CPU()))

        spanning_nodes = [SVector(cospi(2k / 256), sinpi(2k / 256))
                          for k in 0:255]
        spanning = PVContour(spanning_nodes, 1.0, SVector(4.0, 0.0))
        with_spanning = vcat(contours_same, [spanning])
        flat = ContourDynamics._pack_flat_topology(with_spanning, CPU())
        eligible = ContourDynamics._device_eligible_surgery_segment_indices(
            flat, CPU())
        @test length(eligible) == sum(nnodes, contours_same)
        @test Set(ContourDynamics._device_close_pair_candidates(
            with_spanning, δ, CPU())) == dev_pairs
    end

    @testset "Chunked pair scan matches CPU reference across chunk boundary" begin
        # 1200 eligible segments → npairs = 1.44e6 > _PAIR_SCAN_CHUNK, so the
        # candidate sweep spans two chunks; results must match the CPU
        # spatial-index reference exactly.
        δ = 0.05
        contours = [circular_patch(1.0, 600, 1.0),
                    circular_patch(1.0, 600, 1.0; cx=2.03)]
        @test 600 * 600 * 4 > ContourDynamics._PAIR_SCAN_CHUNK
        idx = ContourDynamics.build_spatial_index(contours, δ)
        cpu_pairs = Set(ContourDynamics.find_close_segments(contours, idx, δ))
        buffer = ContourDynamics._device_close_pair_candidate_buffer(contours, δ, CPU())
        buffered = Set(ContourDynamics._unpack_close_pair_candidates(buffer))
        @test !isempty(cpu_pairs)
        @test cpu_pairs == buffered
    end

    @testset "Device close-pair admissibility uses periodic minimum images" begin
        domain = PeriodicDomain(2.0, 2.0)
        δ = 0.03
        contours_in = [
            rectangle_patch(1.2, 1.99, -0.5, 0.5, 6, 1.0),
            rectangle_patch(-1.99, -1.2, -0.5, 0.5, 6, 1.0 + 1e-8),
        ]
        index = ContourDynamics.build_spatial_index(contours_in, δ, domain)
        expected = ContourDynamics.find_close_segments(contours_in, index, δ, domain)
        state = DeviceContourState(deepcopy(contours_in), CPU())

        actual = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, domain, CPU())
        unbounded = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, UnboundedDomain(), CPU())

        @test !isempty(expected)
        @test Set(ContourDynamics._unpack_close_pair_candidates(actual)) == Set(expected)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(unbounded))

        expected_merge = deepcopy(contours_in)
        ContourDynamics.reconnect!(expected_merge, expected, domain)
        rewrite_state = DeviceContourState(deepcopy(contours_in), CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            rewrite_state, actual, domain, CPU())
        ContourDynamics._device_rewrite_state!(
            rewrite_state, selected, domain, CPU())
        actual_merge = materialize_contours(rewrite_state)
        @test length(actual_merge) == length(expected_merge)
        @test all(zip(actual_merge, expected_merge)) do (a, b)
            a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
        end

        nested = [
            PVContour([p + SVector(1.8, 0.0) for p in circular_patch(1.0, 96, 1.0).nodes], 1.0),
            PVContour([p + SVector(1.8, 0.0) for p in circular_patch(0.98, 96, 1.0).nodes], 1.0),
        ]
        nested_state = DeviceContourState(deepcopy(nested), CPU())
        raw_nested = ContourDynamics._device_close_pair_candidate_buffer(
            nested_state, 0.05, domain, CPU())
        admissible_nested = ContourDynamics._device_admissible_close_segment_buffer(
            nested_state, 0.05, domain, CPU())
        nested_index = ContourDynamics.build_spatial_index(nested, 0.05, domain)
        expected_nested = ContourDynamics.find_close_segments(
            nested, nested_index, 0.05, domain)
        @test !isempty(ContourDynamics._unpack_close_pair_candidates(raw_nested))
        @test isempty(expected_nested)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(admissible_nested))
    end

    @testset "Device surgery preserves weak PV levels and periodic containment" begin
        for T in (Float32, Float64)
            δ = T(0.02)
            for q in (one(T), T(1e-10)), factor in (-one(T), zero(T), one(T), T(2))
                cs = [circular_patch(0.2, 32, q; cx=-0.205, T=T),
                      circular_patch(0.2, 32, factor * q; cx=0.205, T=T)]
                candidates = ContourDynamics._device_admissible_close_segment_buffer(
                    cs, δ, UnboundedDomain(), CPU())
                pairs = ContourDynamics._unpack_close_pair_candidates(candidates)
                @test any(p -> p[1] != p[3], pairs) == (factor == one(T))
            end
            nested = [circular_patch(1, 64, 1e-10; T=T),
                      circular_patch(0.99, 64, 1e-10; T=T)]
            candidates = ContourDynamics._device_admissible_close_segment_buffer(
                nested, δ, UnboundedDomain(), CPU())
            @test isempty(ContourDynamics._unpack_close_pair_candidates(candidates))

            cs = [circular_patch(0.2, 64, 2; T=T),
                  circular_patch(0.05, 64, 1; cx=0.745, T=T),
                  circular_patch(0.05, 64, 1; cx=0.855, T=T)]
            state = DeviceContourState(cs, CPU())
            pairs_by_domain = map((UnboundedDomain(), PeriodicDomain(one(T)))) do d
                candidates = ContourDynamics._device_admissible_close_segment_buffer(state, δ, d, CPU())
                pairs = ContourDynamics._unpack_close_pair_candidates(candidates)
                Set(p for p in pairs if p[1] == 2 && p[3] == 3)
            end
            @test !isempty(pairs_by_domain[1])
            @test pairs_by_domain[2] == pairs_by_domain[1]
        end
    end

    @testset "Device close-pair admissibility honors interior vorticity" begin
        δ = 0.05
        outer = circular_patch(1.0, 96, 1.0)
        inner = circular_patch(0.98, 96, 1.0)
        contours = [outer, inner]

        raw_pairs = ContourDynamics._device_close_pair_candidates(contours, δ, CPU())
        admissible = ContourDynamics._device_admissible_close_segments(
            contours, δ, UnboundedDomain(), CPU())
        admissible_buffer = ContourDynamics._device_admissible_close_segment_buffer(
            contours, δ, UnboundedDomain(), CPU())

        @test !isempty(raw_pairs)
        @test isempty(admissible)
        @test isempty(ContourDynamics._unpack_close_pair_candidates(admissible_buffer))
    end

    @testset "Device reconnection planner matches CPU independent-pair selection" begin
        δ = 0.02
        contours = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(3.0, 4.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(4.012, 5.0, 0.0, 1.0, 6, 1.0),
        ]
        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)

        cpu_selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        device_selected = _device_select_reconnection_pairs(contours, close_pairs, CPU())
        candidate_buffer = ContourDynamics._device_close_pair_candidate_buffer(contours, δ, CPU())
        buffer_pairs = ContourDynamics._unpack_close_pair_candidates(candidate_buffer)
        buffer_selected = _device_select_reconnection_pairs(contours, candidate_buffer, CPU())
        selected_buffer = ContourDynamics._device_select_reconnection_pair_buffer(
            contours, candidate_buffer, CPU())
        selected_buffer_pairs = ContourDynamics._unpack_close_pair_candidates(selected_buffer)
        plan = ContourDynamics._device_reconnection_plan(contours, close_pairs, CPU())
        buffer_plan = ContourDynamics._device_reconnection_plan(contours, candidate_buffer, CPU())
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, device_selected, CPU())
        buffer_rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected_buffer, CPU())
        _assert_device_layout_matches_host(contours, rewrite)
        materialized = ContourDynamics._unpack_rewrite_outputs(
            _device_materialize_rewrite_outputs(contours, device_selected, CPU()))
        expected_merge_lengths = [
            nnodes(contours[pair[1]]) + nnodes(contours[pair[3]]) + 1
            for pair in device_selected
        ]

        admissible_pairs = ContourDynamics._unpack_close_pair_candidates(
            ContourDynamics._device_admissible_close_segment_buffer(
                contours, δ, UnboundedDomain(), CPU()))
        @test !isempty(close_pairs)
        @test Set(admissible_pairs) == Set(close_pairs)
        @test issubset(Set(close_pairs), Set(buffer_pairs))
        @test device_selected == cpu_selected
        @test Set(buffer_selected) == Set(device_selected)
        @test Set(selected_buffer_pairs) == Set(device_selected)
        @test Set(to_cpu(plan.op)) == Set(UInt8[2])
        @test Set(to_cpu(buffer_plan.op)) == Set(UInt8[2])
        @test all(isfinite, to_cpu(plan.distance2))
        @test all(isfinite, to_cpu(buffer_plan.distance2))
        @test all(to_cpu(rewrite.op) .== UInt8(2))
        @test to_cpu(buffer_rewrite.op) == to_cpu(rewrite.op)
        @test to_cpu(buffer_rewrite.out_len1) == to_cpu(rewrite.out_len1)
        @test all(to_cpu(rewrite.valid) .== UInt8(1))
        @test to_cpu(rewrite.out_len1) == expected_merge_lengths
        @test all(isfinite, to_cpu(rewrite.stitch_x))
        @test all(isfinite, to_cpu(rewrite.stitch_y))
        @test sort(nnodes.(materialized)) == sort(expected_merge_lengths)
        @test all(c -> length(corner_indices(c)) >= 2, materialized)

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, device_selected)
        once_contours = deepcopy(contours)
        @test ContourDynamics._device_reconnect_once!(once_contours, δ, UnboundedDomain(), CPU())
        full_materialized = ContourDynamics._device_rewrite_contours(contours, device_selected, CPU())
        buffer_materialized = ContourDynamics._device_rewrite_contours(contours, selected_buffer, CPU())
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test nnodes.(buffer_materialized) == nnodes.(cpu_contours)
        @test nnodes.(once_contours) == nnodes.(cpu_contours)
        @test getproperty.(full_materialized, :pv) == getproperty.(cpu_contours, :pv)
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
        @test all(zip(buffer_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
        @test all(zip(once_contours, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end
    end

    @testset "Device topology rewrite matches CPU split output geometry" begin
        gap = 0.002
        δ = 0.01
        N_half = 30
        nodes = SVector{2,Float64}[]
        for k in 0:N_half
            θ = -π/2 + π * k / N_half
            push!(nodes, SVector(1.0 + cos(θ), sin(θ)))
        end
        for k in 1:5
            x = 1.0 - k * 2.0 / 6
            push!(nodes, SVector(x, gap))
        end
        for k in 0:N_half
            θ = π/2 + π * k / N_half
            push!(nodes, SVector(-1.0 + cos(θ), sin(θ)))
        end
        for k in 1:5
            x = -1.0 + k * 2.0 / 6
            push!(nodes, SVector(x, -gap))
        end

        contours = [PVContour(nodes, 1.0)]
        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected, CPU())
        _assert_device_layout_matches_host(contours, rewrite)
        materialized = ContourDynamics._unpack_rewrite_outputs(
            _device_materialize_rewrite_outputs(contours, selected, CPU()))

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, selected)
        cpu_lengths = sort(nnodes.(cpu_contours))
        device_lengths = sort([to_cpu(rewrite.out_len1)[1], to_cpu(rewrite.out_len2)[1]])
        materialized_lengths = sort(nnodes.(materialized))

        @test length(selected) == 1
        @test to_cpu(rewrite.op) == UInt8[1]
        @test to_cpu(rewrite.valid) == UInt8[1]
        @test to_cpu(rewrite.out_count) == [2]
        @test device_lengths == cpu_lengths
        @test materialized_lengths == cpu_lengths
        @test all(zip(materialized, cpu_contours)) do (dev_c, cpu_c)
            isapprox(vortex_area(dev_c), vortex_area(cpu_c); atol=1e-12, rtol=1e-12)
        end
        @test all(c -> all(p -> all(isfinite, p), c.nodes), materialized)
        @test all(c -> !isempty(corner_indices(c)), materialized)

        full_materialized = ContourDynamics._device_rewrite_contours(contours, selected, CPU())
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
        end

        domain = PeriodicDomain(4.0, 4.0)
        periodic_state = DeviceContourState(deepcopy(contours), CPU())
        periodic_candidates = ContourDynamics._device_admissible_close_segment_buffer(
            periodic_state, δ, domain, CPU())
        periodic_selected = ContourDynamics._device_select_reconnection_pair_buffer(
            periodic_state, periodic_candidates, domain, CPU())
        ContourDynamics._device_rewrite_state!(
            periodic_state, periodic_selected, domain, CPU())
        periodic_actual = materialize_contours(periodic_state)
        periodic_expected = deepcopy(contours)
        periodic_index = ContourDynamics.build_spatial_index(
            periodic_expected, δ, domain)
        periodic_pairs = ContourDynamics.find_close_segments(
            periodic_expected, periodic_index, δ, domain)
        ContourDynamics.reconnect!(periodic_expected, periodic_pairs, domain)
        @test nnodes.(periodic_actual) == nnodes.(periodic_expected)
        @test all(zip(periodic_actual, periodic_expected)) do (a, b)
            a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
        end
    end

    @testset "Device full topology rewrite preserves unchanged contours" begin
        δ = 0.02
        contours = [
            rectangle_patch(0.0, 1.0, 0.0, 1.0, 6, 1.0),
            rectangle_patch(1.01, 2.0, 0.0, 1.0, 6, 1.0),
            circular_patch(0.2, 16, 0.5),
        ]
        contours[3] = PVContour([p + SVector(4.0, 0.0) for p in contours[3].nodes],
                                contours[3].pv, contours[3].wrap, contours[3].corners)

        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        rewrite = ContourDynamics._device_topology_rewrite_plan(contours, selected, CPU())
        _assert_device_layout_matches_host(contours, rewrite)

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, selected)
        full_materialized = ContourDynamics._device_rewrite_contours(contours, selected, CPU())

        @test length(full_materialized) == length(cpu_contours) == 2
        @test nnodes.(full_materialized) == nnodes.(cpu_contours)
        @test full_materialized[2].nodes == contours[3].nodes
        @test all(zip(full_materialized, cpu_contours)) do (dev_c, cpu_c)
            all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-12, rtol=1e-12))
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

    @testset "Multi-layer device surgery matches CPU multi-layer surgery (unbounded)" begin
        ml_Ld = SVector(1.0)
        ml_F = 1.0 / (2 * ml_Ld[1]^2)
        ml_coupling = SMatrix{2,2,Float64}(-ml_F, ml_F, ml_F, -ml_F)
        ml_kernel = MultiLayerQGKernel(ml_Ld, ml_coupling)
        # Layer 1: a healthy patch plus a tiny filament that surgery must remove.
        # Layer 2: a single healthy patch.
        tiny = PVContour([SVector(2.0, 0.0), SVector(2.0 + 1e-6, 0.0), SVector(2.0, 1e-6)], 1.0)
        ml_layers = (
            [circular_patch(0.5, 32, 1.0), tiny],
            [PVContour([p + SVector(0.6, -0.2) for p in circular_patch(0.3, 24, -0.7).nodes], -0.7)],
        )
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)

        cpu_prob = MultiLayerContourProblem(ml_kernel, UnboundedDomain(), deepcopy(ml_layers))
        surgery!(cpu_prob, params)

        states = (ContourDynamics.DeviceContourState(deepcopy(ml_layers[1]), CPU()),
                  ContourDynamics.DeviceContourState(deepcopy(ml_layers[2]), CPU()))
        ContourDynamics._device_multilayer_surgery!(states, params, UnboundedDomain(), CPU())

        for ℓ in 1:2
            actual = materialize_contours(states[ℓ])
            expected = cpu_prob.layers[ℓ]
            @test length(actual) == length(expected)
            @test all(zip(actual, expected)) do (a, e)
                length(a.nodes) == length(e.nodes) &&
                    all(isapprox.(a.nodes, e.nodes; rtol=1e-8, atol=1e-10))
            end
        end
    end

    @testset "Periodic multi-layer device surgery matches CPU" begin
        F = 0.5
        kernel = MultiLayerQGKernel(
            SVector(1.0), SMatrix{2,2,Float64}(-F, F, F, -F))
        domain = PeriodicDomain(2.0, 2.0)
        layers = (
            [circular_patch(0.3, 16, 1.0)],
            [circular_patch(0.2, 12, -0.5; cx=0.6)],
        )
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        cpu_prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)

        surgery!(cpu_prob, params)
        ContourDynamics._device_multilayer_surgery!(states, params, domain, CPU())
        for layer in 1:2
            actual = materialize_contours(states[layer])
            @test length(actual) == length(cpu_prob.layers[layer])
            @test all(zip(actual, cpu_prob.layers[layer])) do (a, b)
                a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                    all(isapprox.(a.nodes, b.nodes; rtol=1e-12, atol=1e-12))
            end
        end
    end

    @testset "Periodic device surgery handles cross-seam reconnect and cleanup" begin
        domain = PeriodicDomain(2.0, 2.0)
        contours_in = [
            rectangle_patch(1.2, 1.99, -0.5, 0.5, 8, 1.0),
            rectangle_patch(-1.99, -1.2, -0.5, 0.5, 8, 1.0),
            PVContour([SVector(0.0, 1.5), SVector(1e-6, 1.5),
                       SVector(0.0, 1.5 + 1e-6)], 1.0),
        ]
        params = SurgeryParams(0.03, 0.12, 0.25, 1e-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours_in))
        state = DeviceContourState(deepcopy(contours_in), CPU())

        surgery!(cpu_prob, params)
        ContourDynamics._device_surgery_pipeline!(state, params, domain, CPU())
        actual = materialize_contours(state)

        @test length(actual) == length(cpu_prob.contours) == 1
        @test nnodes.(actual) == nnodes.(cpu_prob.contours)
        @test all(zip(actual, cpu_prob.contours)) do (a, b)
            a.pv == b.pv && a.wrap == b.wrap && a.corners == b.corners &&
                all(isapprox.(a.nodes, b.nodes; rtol=1e-8, atol=1e-10))
        end
    end
end

@testset "Device surgery and remesh regressions" begin
    circle_nodes(cx, cy, R, n) = [SVector(cx + R * cos(2π * k / n), cy + R * sin(2π * k / n))
                                  for k in 0:(n - 1)]
    function device_surgery(cs, params, domain)
        state = DeviceContourState(deepcopy(cs), CPU())
        ContourDynamics._device_surgery_pipeline!(state, params, domain, CPU())
        return materialize_contours(state)
    end
    net_circulation(cs) = sum(c.pv * vortex_area(c) for c in cs; init=0.0)

    @testset "merges compare physical far-side PV levels" begin
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-6, 1)
        opposite = [PVContour(circle_nodes(-0.5015, 0.0, 0.5, 128), 1.0),
                    PVContour(reverse(circle_nodes(0.5015, 0.0, 0.5, 128)), 1.0)]
        out = device_surgery(opposite, params, UnboundedDomain())
        @test length(out) == 2
        @test abs(net_circulation(out)) < 1e-12

        ring = [PVContour(circle_nodes(0.0, 0.0, 1.0, 256), 1.0),
                PVContour(reverse(circle_nodes(0.197, 0.0, 0.8, 256)), 1.0)]
        out = device_surgery(ring, params, UnboundedDomain())
        @test length(out) == 1
        @test net_circulation(out) ≈ net_circulation(ring) rtol=2e-3
    end

    @testset "splits keep the trapped hole clockwise" begin
        # C-shape with a narrow gap between its tips.
        function cshape(r1, r2, θ0, h)
            nodes = SVector{2,Float64}[]
            nout = ceil(Int, r2 * (2π - 2θ0) / h)
            for k in 0:(nout - 1)
                θ = θ0 + (2π - 2θ0) * k / nout
                push!(nodes, SVector(r2 * cos(θ), r2 * sin(θ)))
            end
            ntip = max(2, ceil(Int, (r2 - r1) / h))
            for k in 0:(ntip - 1)
                ρ = r2 - (r2 - r1) * k / ntip
                push!(nodes, SVector(ρ * cos(2π - θ0), ρ * sin(2π - θ0)))
            end
            nin = ceil(Int, r1 * (2π - 2θ0) / h)
            for k in 0:(nin - 1)
                θ = (2π - θ0) - (2π - 2θ0) * k / nin
                push!(nodes, SVector(r1 * cos(θ), r1 * sin(θ)))
            end
            for k in 0:(ntip - 1)
                ρ = r1 + (r2 - r1) * k / ntip
                push!(nodes, SVector(ρ * cos(θ0), ρ * sin(θ0)))
            end
            return PVContour(nodes, 1.0)
        end
        c = cshape(0.5, 0.8, 0.01, 0.03)
        params = SurgeryParams(0.02, 0.08, 0.1, 1e-6, 10)
        out = device_surgery([c], params, UnboundedDomain())
        host = ContourProblem(EulerKernel(), UnboundedDomain(), [deepcopy(c)])
        surgery!(host, params)
        @test length(out) == 2
        @test count(d -> vortex_area(d) < 0, out) == 1
        @test net_circulation(out) ≈ net_circulation([c]) rtol=5e-3
        @test sort(vortex_area.(out)) ≈ sort(vortex_area.(contours(host))) rtol=1e-10
    end

    @testset "periodic self-image contacts are left alone" begin
        nodes = SVector{2,Float64}[]
        for k in 0:49; push!(nodes, SVector(-0.995 + 1.99 * k / 50, -0.1)); end
        for k in 0:4; push!(nodes, SVector(0.995, -0.1 + 0.2 * k / 5)); end
        for k in 0:49; push!(nodes, SVector(0.995 - 1.99 * k / 50, 0.1)); end
        for k in 0:4; push!(nodes, SVector(-0.995, 0.1 - 0.2 * k / 5)); end
        band = PVContour(nodes, 1.0)
        out = device_surgery([band], SurgeryParams(0.02, 0.08, 0.1, 1e-6, 10),
                             PeriodicDomain(1.0, 1.0))
        @test length(out) == 1
        @test net_circulation(out) ≈ net_circulation([band]) rtol=1e-6
    end

    @testset "remesh leaves contours too short for a curve unchanged" begin
        domain_period = SVector(2.0, 0.0)
        short = PVContour([SVector(-0.5, 0.1), SVector(0.5, 0.15)], 1.0, domain_period)
        empty_spanning = PVContour(SVector{2,Float64}[], 1.0, domain_period)
        patch = circular_patch(0.3, 48, 1.0)
        params = SurgeryParams(0.005, 0.02, 0.1, 1e-6, 1)
        state = DeviceContourState([short, empty_spanning, patch], CPU())
        ContourDynamics._device_remesh_state!(state, params, CPU())
        out = materialize_contours(state)
        host = deepcopy([short, empty_spanning, patch])
        ContourDynamics._remesh_all!(host, params, SVector{2,Float64}[], Float64[],
                                     SVector{2,Float64}[])
        @test nnodes(out[1]) == 2
        @test out[1].nodes == short.nodes
        @test nnodes(out[2]) == 0
        @test nnodes.(out) == nnodes.(host)
    end

    @testset "multi-layer periodic energy reads each mode's cache" begin
        # setup_ewald_cache! for a single-layer QG problem also configures the
        # domain's Euler cache; the baroclinic mode below keeps its own cache.
        clear_ewald_cache!()
        domain = PeriodicDomain(Float64(π))
        setup_ewald_cache!(domain, QGKernel(0.7); n_fourier=16, n_images=4)
        F = 0.5
        kernel = MultiLayerQGKernel(SVector(1 / sqrt(2F)), SMatrix{2,2}(-F, F, F, -F))
        layers = ([circular_patch(0.4, 48, 1.0; cx=0.3)],
                  [circular_patch(0.3, 40, -0.8; cx=-0.4, cy=0.2)])
        prob = MultiLayerContourProblem(kernel, domain, deepcopy(layers))
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        @test ContourDynamics._ka_multilayer_energy_from_states(
            states, kernel, domain, CPU()) ≈ energy(prob) rtol=1e-12
        clear_ewald_cache!()
    end
end

@testset "Device surgery parity edge cases" begin
    # Two pinched "dumbbell" contours whose necks differ in gap. Both split in
    # the same reconnect round; the CPU appends split daughters in proximity
    # order, and the device must lay its pair buffer out the same way so the
    # contour vectors agree element for element, not only as sets.
    function dumbbell(x0, gap; T=Float64)
        N_half = 30
        nodes = SVector{2,T}[]
        for k in 0:N_half
            θ = -π/2 + π * k / N_half
            push!(nodes, SVector{2,T}(x0 + 1 + cos(θ), sin(θ)))
        end
        for k in 1:5
            push!(nodes, SVector{2,T}(x0 + 1 - k * 2 / 6, gap))
        end
        for k in 0:N_half
            θ = π/2 + π * k / N_half
            push!(nodes, SVector{2,T}(x0 - 1 + cos(θ), sin(θ)))
        end
        for k in 1:5
            push!(nodes, SVector{2,T}(x0 - 1 + k * 2 / 6, -gap))
        end
        return PVContour(nodes, T(1))
    end
    same_contours(a, b; rtol=1e-12, atol=1e-12) =
        length(a) == length(b) && all(zip(a, b)) do (p, q)
            p.pv == q.pv && p.wrap == q.wrap && p.corners == q.corners &&
                nnodes(p) == nnodes(q) &&
                all(isapprox.(p.nodes, q.nodes; rtol=rtol, atol=atol))
        end

    @testset "Simultaneous splits keep CPU daughter order" begin
        δ = 0.01
        # The second contour has the closer neck, so proximity order differs
        # from candidate-buffer (contour index) order.
        contours = [dumbbell(0.0, 0.004), dumbbell(6.0, 0.001)]
        domain = UnboundedDomain()

        idx = ContourDynamics.build_spatial_index(contours, δ)
        close_pairs = ContourDynamics.find_close_segments(contours, idx, δ)
        cpu_selected = ContourDynamics._select_reconnection_pairs(contours, close_pairs)
        @test length(cpu_selected) == 2
        @test cpu_selected[1][1] == 2  # closer neck first

        state = DeviceContourState(deepcopy(contours), CPU())
        candidates = ContourDynamics._device_admissible_close_segment_buffer(
            state, δ, domain, CPU())
        selected = ContourDynamics._device_select_reconnection_pair_buffer(
            state, candidates, domain, CPU())
        device_selected = collect(zip(to_cpu(selected.ci), to_cpu(selected.i),
                                      to_cpu(selected.cj), to_cpu(selected.j)))
        @test device_selected == cpu_selected

        cpu_contours = deepcopy(contours)
        ContourDynamics.reconnect!(cpu_contours, cpu_selected, domain)
        ContourDynamics._device_rewrite_state!(state, selected, domain, CPU())
        @test length(cpu_contours) == 4
        @test same_contours(materialize_contours(state), cpu_contours)

        # The full pipelines must agree in order too.
        params = SurgeryParams(δ, 0.05, 0.3, 1e-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), domain, deepcopy(contours))
        surgery!(cpu_prob, params)
        pipeline_state = DeviceContourState(deepcopy(contours), CPU())
        ContourDynamics._device_surgery_pipeline!(pipeline_state, params, domain, CPU())
        @test length(cpu_prob.contours) == 4
        @test same_contours(materialize_contours(pipeline_state), cpu_prob.contours;
                            rtol=1e-10, atol=1e-10)
    end

    @testset "Float32 end-to-end surgery parity" begin
        δ = 0.01f0
        contours = [dumbbell(0f0, 0.004f0; T=Float32), dumbbell(6f0, 0.001f0; T=Float32)]
        params = SurgeryParams(δ, 0.05f0, 0.3f0, 1f-8, 10)
        cpu_prob = ContourProblem(EulerKernel(), UnboundedDomain(), deepcopy(contours))
        surgery!(cpu_prob, params)
        state = DeviceContourState(deepcopy(contours), CPU())
        ContourDynamics._device_surgery_pipeline!(state, params, UnboundedDomain(), CPU())
        @test length(cpu_prob.contours) == 4
        @test same_contours(materialize_contours(state), cpu_prob.contours;
                            rtol=1f-4, atol=1f-5)
    end

    @testset "Obtuse-corner demotion keeps corners at degenerate segments" begin
        # Node 3 is a right-angle corner and stays. Node 1 is a corner whose
        # previous segment has length 1e-20: below
        # eps, so its angle is undefined and the CPU keeps the corner even
        # though the dot product is (barely) negative.
        nodes = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(1.0, 1.0),
                 SVector(0.0, 1.0), SVector(-1e-20, 0.0)]
        corners = [true, false, true, false, false]
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)

        cpu_contours = [deepcopy(c)]
        ContourDynamics._demote_obtuse_corners!(cpu_contours)
        state = DeviceContourState([deepcopy(c)], CPU())
        ContourDynamics._demote_obtuse_corners!(state, CPU())
        actual = only(materialize_contours(state))

        @test cpu_contours[1].corners == [true, false, true, false, false]
        @test actual.corners == cpu_contours[1].corners
    end

    @testset "Fixed-corner remesh of a zero-length span matches CPU" begin
        base = circular_patch(1.0, 40, 1.0)
        nodes = copy(base.nodes)
        # Two coincident corners: the span between them has zero length.
        insert!(nodes, 11, nodes[10])
        corners = falses(length(nodes))
        corners[10] = true
        corners[11] = true
        corners[31] = true
        c = PVContour(nodes, 1.0, zero(SVector{2,Float64}), corners)
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)

        cpu_c = remesh(c, params; _density_sources=[c])
        dev_c = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
        @test nnodes(dev_c) == nnodes(cpu_c)
        @test dev_c.corners == cpu_c.corners
        @test all(isapprox.(dev_c.nodes, cpu_c.nodes; atol=1e-10, rtol=1e-10))
    end

    @testset "Closed remesh of a zero-perimeter contour matches CPU" begin
        p = SVector(0.3, -0.2)
        c = PVContour([p, p, p, p], 1.0)
        params = SurgeryParams(0.005, 0.04, 0.16, 1e-6, 10)
        cpu_c = remesh(c, params)
        dev_c = only(ContourDynamics._device_remesh_contours([c], params, CPU()))
        @test nnodes(cpu_c) == 4
        @test nnodes(dev_c) == nnodes(cpu_c)
        @test dev_c.corners == cpu_c.corners
        @test dev_c.nodes == cpu_c.nodes
    end

    @testset "Multi-layer device stall warnings name the layer" begin
        # The device loop must receive the same per-layer label as the CPU
        # path; check the keyword plumbing without triggering a stall.
        layers = ([circular_patch(0.5, 32, 1.0)], [circular_patch(0.3, 24, -0.7)])
        params = SurgeryParams(0.002, 0.01, 0.2, 1e-8, 100)
        states = ntuple(i -> DeviceContourState(deepcopy(layers[i]), CPU()), 2)
        @test_logs min_level=Logging.Warn ContourDynamics._device_multilayer_surgery!(
            states, params, UnboundedDomain(), CPU())
        @test_logs min_level=Logging.Warn ContourDynamics._device_surgery_pipeline!(
            states[1], params, UnboundedDomain(), CPU(); layer_label=" layer 1")
    end
end
